#!/usr/bin/env python3
"""
gemv_dramsim3_tracer.py

Generate a DRAMsim3-friendly transaction trace for GEMV:
    (M x K) @ (K x 1) = (M x 1)

Default problem:
    (32 x 1536) @ (1536 x 1) = (32 x 1)

Assumptions:
- DRAM row size = 1KB
- DRAM burst / trace granularity = 64B
- The trace is emitted at 64B granularity (one DRAM request per 64B chunk)
- Trace format follows DRAMsim3's simple trace style:
      <hex_addr> READ <clk>
      <hex_addr> WRITE <clk>

Notes:
- This script does NOT model compute latency. It only emits memory requests.
- To make the simulator behavior consistent with this trace, your DRAMsim3 .ini
  should use a matching transaction / burst granularity and an address mapping
  compatible with your study.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_up(x: int, align: int) -> int:
    # x를 align 단위로 올림해서 가장 가까운 배수로 맞춤
    return ceil_div(x, align) * align


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

        self.elems_per_burst = burst_bytes // elem_bytes
        if self.elems_per_burst <= 0:
            raise ValueError("burst_bytes must be >= elem_bytes")
        if burst_bytes % elem_bytes != 0:
            raise ValueError("For a clean burst trace, burst_bytes must be divisible by elem_bytes")

        self.addr = AddressSpace(burst_bytes=burst_bytes, row_bytes=row_bytes)
        self.time = 0
        self.trace: List[TraceReq] = []
        self.num_gemv = num_gemv
        self._alloc_tensors()

    def _alloc_tensors(self) -> None:
        a_bytes = self.m * self.k * self.elem_bytes
        x_bytes = self.k * self.elem_bytes
        y_bytes = self.m * self.elem_bytes

        for g in range(self.num_gemv):
            self.addr.alloc(f"A{g}", a_bytes, force_row_align=True)
            self.addr.alloc(f"x{g}", x_bytes, force_row_align=True)
            self.addr.alloc(f"y{g}", y_bytes, force_row_align=True)

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
        # 실제 접근 범위를 burst 경계(64B) 기준으로 확장
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

    def generate(self) -> List[TraceReq]:
        k_bytes = self.k * self.elem_bytes
        bursts_per_dot = ceil_div(k_bytes, self.burst_bytes)

        for g in range(self.num_gemv):
            a_base = self.addr.get(f"A{g}")
            x_base = self.addr.get(f"x{g}")
            y_base = self.addr.get(f"y{g}")

            if self.reuse_x == "global":
                self._emit_tensor_span(
                    op="READ",
                    tensor=f"x{g}",
                    base=x_base,
                    offset_bytes=0,
                    span_bytes=k_bytes,
                    meta={
                        "stage": "x_preload_global",
                        "gemv_id": g,
                        "k": self.k,
                    },
                )

            for i in range(self.m):
                row_byte_offset = i * k_bytes

                for c in range(bursts_per_dot):
                    burst_off = c * self.burst_bytes

                    self._emit(
                        op="READ",
                        addr=a_base + row_byte_offset + burst_off,
                        tensor=f"A{g}",
                        logical_offset=row_byte_offset + burst_off,
                        meta={
                            "stage": "gemv",
                            "gemv_id": g,
                            "row": i,
                            "burst_chunk": c,
                            "operand": "A",
                        },
                    )

                    need_x_read = (
                        self.reuse_x == "none" or
                        self.reuse_x == "per_row"
                    )

                    if need_x_read:
                        self._emit(
                            op="READ",
                            addr=x_base + burst_off,
                            tensor=f"x{g}",
                            logical_offset=burst_off,
                            meta={
                                "stage": "gemv",
                                "gemv_id": g,
                                "row": i,
                                "burst_chunk": c,
                                "operand": "x",
                            },
                        )

                y_elem_off = i * self.elem_bytes
                self._emit_tensor_span(
                    op="WRITE",
                    tensor=f"y{g}",
                    base=y_base,
                    offset_bytes=y_elem_off,
                    span_bytes=self.elem_bytes,
                    meta={
                        "stage": "store_output",
                        "gemv_id": g,
                        "row": i,
                    },
                )

        return self.trace
    def summary(self) -> Dict:
        a_bytes = self.m * self.k * self.elem_bytes
        x_bytes = self.k * self.elem_bytes
        y_bytes = self.m * self.elem_bytes

        row_payload_bytes = self.k * self.elem_bytes
        bursts_per_dot = ceil_div(row_payload_bytes, self.burst_bytes)

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
            },
           "tensor_layout": {
                "num_gemv": self.num_gemv,
                "A": [
                    {"gemv_id": g, "base": hex(self.addr.get(f"A{g}")), "bytes": a_bytes}
                    for g in range(self.num_gemv)
                ],
                "x": [
                    {"gemv_id": g, "base": hex(self.addr.get(f"x{g}")), "bytes": x_bytes}
                    for g in range(self.num_gemv)
                ],
                "y": [
                    {"gemv_id": g, "base": hex(self.addr.get(f"y{g}")), "bytes": y_bytes}
                    for g in range(self.num_gemv)
                ],
            },
            "access_pattern": {
                "reuse_x": self.reuse_x,
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
    p.add_argument("--prefix", type=str, default="gemv_32x1536_fp16_b64_128") # to fix
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