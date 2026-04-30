#!/usr/bin/env python3
"""
gemv_dramsim3_tracer_banked.py

Generate a DRAMsim3-friendly transaction trace for GEMV while optionally
modeling logical bank-level parallel placement for the A matrix.

Default problem:
    (32 x 1536) @ (1536 x 1) = (32 x 1)

Key additions over the simple tracer:
- Logical bank-group / bank aware placement for A
- Support for sharding A across DDR banks either by row or by burst chunk
- Metadata records which logical bank-group / bank each A request targets

Important note:
This script creates a *logical* bank-aware layout. Whether DRAMsim3 maps the
resulting addresses to the exact physical bank-group / bank you intend still
depends on the address mapping policy in your DRAMsim3 configuration.
To make the simulator reflect the intended layout, choose an address mapping
that is compatible with this placement scheme.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_up(x: int, align: int) -> int:
    return ceil_div(x, align) * align

BASE = 0x100000000000

# DDR4_8Gb_x16_3200
def encode_addr(ro: int, ra: int, bg: int, ba: int, co: int) -> int:
    return (
        BASE
        | (ro << 17)
        | (ra << 16)
        | (ba << 14)
        | (bg << 13)
        | (co << 6)
    )

@dataclass
class TraceReq:
    clk: int
    addr: int
    op: str         # READ / WRITE
    tensor: str
    logical_offset: int
    nbytes: int
    row_base: int
    col_idx_in_row: int
    meta: Dict


class AddressSpace:
    def __init__(
        self,
        base: int = 0x1000_0000_0000,
        burst_bytes: int = 64,
        row_bytes: int = 1024,
    ):
        self.cur = base
        self.burst_bytes = burst_bytes
        self.row_bytes = row_bytes
        self.map: Dict[str, int] = {}
        self.size_map: Dict[str, int] = {}

    def alloc(self, name: str, size: int, force_row_align: bool = False) -> int:
        if name in self.map:
            return self.map[name]
        align = self.row_bytes if force_row_align else self.burst_bytes
        addr = align_up(self.cur, align)
        self.map[name] = addr
        self.size_map[name] = size
        self.cur = addr + size
        return addr

    def get(self, name: str) -> int:
        return self.map[name]

    def size(self, name: str) -> int:
        return self.size_map[name]


class TraceWriter:
    def __init__(self, dramsim3_path: Path, jsonl_path: Path):
        self.dramsim3_fp = dramsim3_path.open("w", encoding="utf-8")
        self.jsonl_fp = jsonl_path.open("w", encoding="utf-8")
        self.count = 0
        self.reads = 0
        self.writes = 0

    def write(self, req: TraceReq) -> None:
        self.dramsim3_fp.write(f"{hex(req.addr)} {req.op} {req.clk}\n")
        self.jsonl_fp.write(json.dumps(asdict(req), ensure_ascii=False) + "\n")
        self.count += 1
        if req.op == "READ":
            self.reads += 1
        else:
            self.writes += 1

    def close(self) -> None:
        self.dramsim3_fp.close()
        self.jsonl_fp.close()


class BankedAMatrixLayout:
    """
    Logical layout for A across bank groups and banks.

    Two supported placement modes:
    1) contiguous
       - Original behavior: A placed as one contiguous tensor.
    2) bank_round_robin_rows
       - Whole GEMV rows of A are assigned round-robin to banks.
    3) bank_round_robin_bursts
       - Each 64B burst chunk of a GEMV row is assigned round-robin to banks.
         This is usually the more relevant mode for exposing intra-dot-product
         bank-level parallelism.

    We allocate one memory region per logical bank shard so the trace contains
    bank-distributed addresses instead of one monolithic contiguous A region.
    """

    def __init__(
        self,
        addr: AddressSpace,
        m: int,
        k: int,
        elem_bytes: int,
        burst_bytes: int,
        row_bytes: int,
        bankgroups: int,
        banks_per_group: int,
        layout_mode: str,
    ):
        self.addr = addr
        self.m = m
        self.k = k
        self.elem_bytes = elem_bytes
        self.burst_bytes = burst_bytes
        self.row_bytes = row_bytes
        self.bankgroups = bankgroups
        self.banks_per_group = banks_per_group
        self.total_banks = bankgroups * banks_per_group
        self.layout_mode = layout_mode

        self.k_bytes = self.k * self.elem_bytes
        self.bursts_per_row = ceil_div(self.k_bytes, self.burst_bytes)

        # contiguous 모드면 기존처럼 하나만 할당
        self.contiguous_base: Optional[int] = None

        # bank shard별 base / size
        self.bank_bases: Dict[int, int] = {}
        self.bank_sizes: Dict[int, int] = {}

        # row/burst -> bank 배치 테이블
        self.row_to_bank: Dict[int, int] = {}
        self.burst_owner: Dict[Tuple[int, int], int] = {}
        self.local_offset_by_chunk: Dict[Tuple[int, int], int] = {}

        self._build()

    def _bank_name(self, bank_id: int) -> str:
        bg = bank_id // self.banks_per_group
        b = bank_id % self.banks_per_group
        return f"A_bg{bg}_b{b}"

    def _build(self) -> None:
        a_bytes = self.m * self.k * self.elem_bytes

        if self.layout_mode == "contiguous":
            self.contiguous_base = self.addr.alloc("A", a_bytes, force_row_align=True)
            return

        if self.layout_mode == "bank_round_robin_rows":
            bytes_per_bank = [0 for _ in range(self.total_banks)]
            for i in range(self.m):
                bank_id = i % self.total_banks
                self.row_to_bank[i] = bank_id
                local_off = bytes_per_bank[bank_id]
                self.local_offset_by_chunk[(i, -1)] = local_off
                bytes_per_bank[bank_id] += self.k_bytes

            for bank_id, size in enumerate(bytes_per_bank):
                self.bank_sizes[bank_id] = size
                self.bank_bases[bank_id] = self.addr.alloc(
                    self._bank_name(bank_id),
                    size if size > 0 else self.burst_bytes,
                    force_row_align=True,
                )
            return

        if self.layout_mode == "bank_round_robin_bursts":
            bytes_per_bank = [0 for _ in range(self.total_banks)]
            for i in range(self.m):
                for c in range(self.bursts_per_row):
                    bank_id = c % self.total_banks
                    self.burst_owner[(i, c)] = bank_id
                    local_off = bytes_per_bank[bank_id]
                    self.local_offset_by_chunk[(i, c)] = local_off
                    bytes_per_bank[bank_id] += self.burst_bytes

            for bank_id, size in enumerate(bytes_per_bank):
                self.bank_sizes[bank_id] = size
                self.bank_bases[bank_id] = self.addr.alloc(
                    self._bank_name(bank_id),
                    size if size > 0 else self.burst_bytes,
                    force_row_align=True,
                )
            return

        raise ValueError(f"Unsupported A layout mode: {self.layout_mode}")

    def get_a_access_info(self, row: int, burst_chunk: int) -> Dict:
        logical_offset = row * self.k_bytes + burst_chunk * self.burst_bytes

        if self.layout_mode == "contiguous":
            assert self.contiguous_base is not None
            addr = self.contiguous_base + logical_offset
            return {
                "addr": addr,
                "logical_offset": logical_offset,
                "bankgroup": None,
                "bank": None,
                "bank_id": None,
                "physical_shard_offset": logical_offset,
                "physical_shard_name": "A",
            }

        if self.layout_mode == "bank_round_robin_rows":
            bank_id = self.row_to_bank[row]
            bankgroup = bank_id // self.banks_per_group
            bank = bank_id % self.banks_per_group
            row_base_off = self.local_offset_by_chunk[(row, -1)]
            shard_off = row_base_off + burst_chunk * self.burst_bytes
            return {
                "addr": self.bank_bases[bank_id] + shard_off,
                "logical_offset": logical_offset,
                "bankgroup": bankgroup,
                "bank": bank,
                "bank_id": bank_id,
                "physical_shard_offset": shard_off,
                "physical_shard_name": self._bank_name(bank_id),
            }

        if self.layout_mode == "bank_round_robin_bursts":
            BANK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

            cols_per_bank_row = self.row_bytes // self.burst_bytes  # 8192 / 64 = 128
            bursts_per_ro_all_banks = cols_per_bank_row * self.total_banks  # 128 * 8 = 1024

            global_burst = row * self.bursts_per_row + burst_chunk

            ro = global_burst // bursts_per_ro_all_banks
            slot_in_ro = global_burst % bursts_per_ro_all_banks

            bank_order_idx = slot_in_ro % self.total_banks
            bank_id = BANK_ORDER[bank_order_idx]

            bg = bank_id // self.banks_per_group
            ba = bank_id % self.banks_per_group

            co = slot_in_ro // self.total_banks
            ra = 0

            addr = encode_addr(ro=ro, ra=ra, bg=bg, ba=ba, co=co)

            return {
                "addr": addr,
                "logical_offset": logical_offset,
                "bankgroup": bg,
                "bank": ba,
                "bank_id": bank_id,
                "physical_shard_offset": logical_offset,
                "physical_shard_name": f"A_bg{bg}_b{ba}",
            }
        raise ValueError(f"Unsupported A layout mode: {self.layout_mode}")

    def summary(self) -> Dict:
        if self.layout_mode == "contiguous":
            return {
                "layout_mode": self.layout_mode,
                "total_banks": self.total_banks,
                "A": {
                    "base": hex(self.contiguous_base) if self.contiguous_base is not None else None,
                    "bytes": self.m * self.k * self.elem_bytes,
                },
            }

        banks = []
        for bank_id in range(self.total_banks):
            banks.append(
                {
                    "bank_id": bank_id,
                    "bankgroup": bank_id // self.banks_per_group,
                    "bank": bank_id % self.banks_per_group,
                    "name": self._bank_name(bank_id),
                    "base": hex(self.bank_bases[bank_id]),
                    "bytes": self.bank_sizes[bank_id],
                }
            )

        return {
            "layout_mode": self.layout_mode,
            "bankgroups": self.bankgroups,
            "banks_per_group": self.banks_per_group,
            "total_banks": self.total_banks,
            "bursts_per_row": self.bursts_per_row,
            "bank_shards": banks,
        }


class GemvTracer:
    def __init__(
        self,
        m: int,
        k: int,
        elem_bytes: int,
        burst_bytes: int,
        row_bytes: int,
        interarrival: int,
        reuse_x: str,
        out_dir: Path,
        bankgroups: int,
        banks_per_group: int,
        a_layout: str,
        num_gemv: int,
    ):
        if row_bytes % burst_bytes != 0:
            raise ValueError("row_bytes must be a multiple of burst_bytes")

        self.m = m
        self.k = k
        self.elem_bytes = elem_bytes
        self.burst_bytes = burst_bytes
        self.row_bytes = row_bytes
        self.interarrival = interarrival
        self.reuse_x = reuse_x
        self.out_dir = out_dir
        self.bankgroups = bankgroups
        self.banks_per_group = banks_per_group
        self.total_banks = bankgroups * banks_per_group
        self.a_layout = a_layout

        self.num_gemv = num_gemv

        self.elems_per_burst = burst_bytes // elem_bytes
        if self.elems_per_burst <= 0:
            raise ValueError("burst_bytes must be >= elem_bytes")
        if burst_bytes % elem_bytes != 0:
            raise ValueError("For a clean burst trace, burst_bytes must be divisible by elem_bytes")

        self.addr = AddressSpace(burst_bytes=burst_bytes, row_bytes=row_bytes)
        self.time = 0
        self.trace: List[TraceReq] = []

        self.x_bases = []
        self.y_bases = []
        self.a_layouts = []

        for g in range(self.num_gemv):
            x_base = encode_addr(ro=0, ra=1, bg=0, ba=g%8, co=0) # to fix
            y_base = encode_addr(ro=0, ra=1, bg=0, ba=g%8, co=48) # to fix

            self.x_bases.append(x_base)
            self.y_bases.append(y_base)

            a_layout = BankedAMatrixLayout(
                addr=self.addr,
                m=self.m,
                k=self.k,
                elem_bytes=self.elem_bytes,
                burst_bytes=self.burst_bytes,
                row_bytes=self.row_bytes,
                bankgroups=self.bankgroups,
                banks_per_group=self.banks_per_group,
                layout_mode=self.a_layout,
            )
            self.a_layouts.append(a_layout)

    def _emit(self, op: str, addr: int, tensor: str, logical_offset: int, meta: Dict) -> None:
        row_base = (addr // self.row_bytes) * self.row_bytes
        col_idx_in_row = (addr - row_base) // self.burst_bytes

        req = TraceReq(
            clk=self.time,
            addr=addr,
            op=op,
            tensor=tensor,
            logical_offset=logical_offset,
            nbytes=self.burst_bytes,
            row_base=row_base,
            col_idx_in_row=col_idx_in_row,
            meta=meta,
        )

        self.trace.append(req)
        self.time += self.interarrival

    def _emit_tensor_span(
        self,
        op: str,
        tensor: str,
        base: int,
        offset_bytes: int,
        span_bytes: int,
        meta: Dict,
    ) -> None:
        first = (offset_bytes // self.burst_bytes) * self.burst_bytes
        last = align_up(offset_bytes + span_bytes, self.burst_bytes)

        for off in range(first, last, self.burst_bytes):
            self._emit(
                op=op,
                addr=base + off,
                tensor=tensor,
                logical_offset=off,
                meta=meta,
            )

    def generate(self):
        k_bytes = self.k * self.elem_bytes
        bursts_per_dot = ceil_div(k_bytes, self.burst_bytes)

        for g in range(self.num_gemv):   # 🔥 추가
            x_base = self.x_bases[g]
            y_base = self.y_bases[g]
            a_layout = self.a_layouts[g]

            for i in range(self.m):
                for c in range(bursts_per_dot):
                    burst_off = c * self.burst_bytes
                    a_info = a_layout.get_a_access_info(i, c)

                    self._emit(
                        op="READ",
                        addr=a_info["addr"],
                        tensor=f"A{g}",   # 구분
                        logical_offset=a_info["logical_offset"],
                        meta={
                            "gemv_id": g,   # 🔥 추가
                            "row": i,
                            "burst_chunk": c,
                            "operand": "A",
                        },
                    )

                    self._emit(
                        op="READ",
                        addr=x_base + burst_off,
                        tensor=f"x{g}",
                        logical_offset=burst_off,
                        meta={
                            "gemv_id": g,
                            "row": i,
                            "burst_chunk": c,
                            "operand": "x",
                        },
                    )

                # y write
                y_elem_off = i * self.elem_bytes
                self._emit_tensor_span(
                    op="WRITE",
                    tensor=f"y{g}",
                    base=y_base,
                    offset_bytes=y_elem_off,
                    span_bytes=self.elem_bytes,
                    meta={
                        "gemv_id": g,
                        "row": i,
                    },
                )

        return self.trace

    def summary(self) -> Dict:
        x_bytes = self.k * self.elem_bytes
        y_bytes = self.m * self.elem_bytes

        row_payload_bytes = self.k * self.elem_bytes
        bursts_per_dot = ceil_div(row_payload_bytes, self.burst_bytes)

        bank_histogram: Dict[str, int] = {}
        for t in self.trace:
            if t.tensor == "A":
                bg = t.meta.get("bankgroup")
                bk = t.meta.get("bank")
                if bg is not None and bk is not None:
                    key = f"bg{bg}_b{bk}"
                    bank_histogram[key] = bank_histogram.get(key, 0) + 1

        return {
            "problem": {
                "M": self.m,
                "K": self.k,
                "elem_bytes": self.elem_bytes,
                "equation": f"({self.m}x{self.k}) x ({self.k}x1) = ({self.m}x1)",
            },
            "dram_geometry": {
                "row_bytes": self.row_bytes,
                "burst_bytes": self.burst_bytes,
                "bursts_per_row": self.row_bytes // self.burst_bytes,
                "bankgroups": self.bankgroups,
                "banks_per_group": self.banks_per_group,
                "total_banks": self.total_banks,
            },
            "tensor_layout": {
                    "num_gemv": self.num_gemv,
                    "A": [
                        {
                            "gemv_id": g,
                            "layout": self.a_layouts[g].summary(),
                        }
                        for g in range(self.num_gemv)
                    ],
                    "x": [
                        {
                            "gemv_id": g,
                            "base": hex(self.x_bases[g]),
                            "bytes": x_bytes,
                        }
                        for g in range(self.num_gemv)
                    ],
                    "y": [
                        {
                            "gemv_id": g,
                            "base": hex(self.y_bases[g]),
                            "bytes": y_bytes,
                        }
                        for g in range(self.num_gemv)
                    ],
                },
            "access_pattern": {
                "reuse_x": self.reuse_x,
                "a_layout": self.a_layout,
                "k_bytes_per_dot": row_payload_bytes,
                "burst_requests_per_dot_operand": bursts_per_dot,
                "interarrival_cycles": self.interarrival,
            },
            "trace_stats": {
                "num_requests": len(self.trace),
                "num_reads": sum(1 for t in self.trace if t.op == "READ"),
                "num_writes": sum(1 for t in self.trace if t.op == "WRITE"),
                "first_clk": self.trace[0].clk if self.trace else None,
                "last_clk": self.trace[-1].clk if self.trace else None,
                "a_bank_request_histogram": bank_histogram,
            },
        }


def parse_args():
    p = argparse.ArgumentParser(description="Generate a DRAMsim3 trace for GEMV.")
    p.add_argument("--m", type=int, default=32, help="Number of rows in A")
    p.add_argument("--k", type=int, default=1536, help="Number of columns in A / length of x")
    p.add_argument("--elem-bytes", type=int, default=2, help="Bytes per element (default: fp16)")
    p.add_argument("--dram-row-bytes", type=int, default=8192, help="DRAM row size in bytes")
    p.add_argument("--dram-burst-bytes", type=int, default=64, help="DRAM burst granularity in bytes")
    p.add_argument("--interarrival", type=int, default=1, help="Cycles between requests in the trace")
    p.add_argument("--bankgroups", type=int, default=2, help="Number of DDR bank groups")
    p.add_argument("--banks-per-group", type=int, default=4, help="Banks per bank group")
    p.add_argument(
        "--a-layout",
        choices=["contiguous", "bank_round_robin_rows", "bank_round_robin_bursts"],
        default="bank_round_robin_bursts",
        help=(
            "How to place A in memory. "
            "'contiguous' = original monolithic layout, "
            "'bank_round_robin_rows' = distribute GEMV rows across banks, "
            "'bank_round_robin_bursts' = distribute 64B A bursts across banks."
        ),
    )
    p.add_argument(
        "--reuse-x",
        choices=["none", "per_row", "global"],
        default="none",
        help=(
            "x reuse model. "
            "'none' = read x from DRAM for every row, "
            "'per_row' = same as none for this simple single-row-at-a-time tracer, "
            "'global' = preload x once, then do not issue more x DRAM reads."
        ),
    )
    p.add_argument("--num_gemv", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=Path("./gemv_trace_out"))
    p.add_argument("--prefix", type=str, default="gemv_32x1536_fp16_b64_banked_16_new")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tracer = GemvTracer(
        m=args.m,
        k=args.k,
        elem_bytes=args.elem_bytes,
        burst_bytes=args.dram_burst_bytes,
        row_bytes=args.dram_row_bytes,
        interarrival=args.interarrival,
        reuse_x=args.reuse_x,
        out_dir=args.out_dir,
        bankgroups=args.bankgroups,
        banks_per_group=args.banks_per_group,
        a_layout=args.a_layout,
        num_gemv=args.num_gemv,
    )
    trace = tracer.generate()

    dramsim3_path = args.out_dir / f"{args.prefix}.trace"
    jsonl_path = args.out_dir / f"{args.prefix}.jsonl"
    summary_path = args.out_dir / f"{args.prefix}_summary.json"

    writer = TraceWriter(dramsim3_path, jsonl_path)
    for req in trace:
        writer.write(req)
    writer.close()

    summary = tracer.summary()
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote:")
    print(f"  DRAMsim3 trace : {dramsim3_path}")
    print(f"  JSONL trace    : {jsonl_path}")
    print(f"  Summary        : {summary_path}")


if __name__ == "__main__":
    main()
