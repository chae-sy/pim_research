#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gemv_dramsim3_tracer_newton.py

Newton(PIM)용 DRAMsim3 trace 생성기
- 연산: (M x K) @ (K x 1) = (M x 1)
- 기본값: (32 x 1536) @ (1536 x 1) = (32 x 1)

이 스크립트는 일반 READ/WRITE trace가 아니라
Newton용 PIM 명령을 포함한 trace를 생성한다.

생성 명령:
- GWRITE   : 입력 벡터 chunk를 global buffer에 적재
- G_ACT    : 4-bank cluster 단위 ganged activate
- COMP     : 모든 bank에서 같은 sub-chunk에 대해 ganged compute
- READRES  : 모든 bank의 result latch를 host가 읽음

가정:
- DRAM row size = 1KB
- DRAM column I/O size = 32B
- element = fp16 (2B)
- Newton 논문의 1KB row / 32B column 구조를 따름
- config의 address_mapping과 동일한 방식으로 주소를 구성
- PIM header encoding은 네 configuration.cc의 EncodePIMHeader()를 그대로 옮김

주의:
1. 이 코드는 네가 올린 Config::EncodePIMHeader() 로직을 그대로 Python으로 옮긴 것.
2. 즉, NewtonController가 이 header address를 해석한다는 가정하에 동작함.
3. trace parser가 "<hex_addr> <OP> <clk>" 형태에서
   OP로 GWRITE / G_ACT / COMP / READRES를 받는다는 전제다.
"""

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def ilog2(x: int) -> int:
    if x <= 0 or not is_pow2(x):
        raise ValueError(f"log2 입력은 2의 거듭제곱이어야 합니다. got={x}")
    return int(math.log2(x))


def align_up(x: int, align: int) -> int:
    return ceil_div(x, align) * align


@dataclass
class TraceReq:
    clk: int
    addr: int
    op: str
    meta: Dict


class TraceWriter:
    def __init__(self, dramsim3_path: Path, jsonl_path: Path):
        self.dramsim3_fp = dramsim3_path.open("w", encoding="utf-8")
        self.jsonl_fp = jsonl_path.open("w", encoding="utf-8")

    def write(self, req: TraceReq) -> None:
        # DRAMsim3 trace 형식
        self.dramsim3_fp.write(f"{hex(req.addr)} {req.op} {req.clk}\n")
        self.jsonl_fp.write(json.dumps(asdict(req), ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.dramsim3_fp.close()
        self.jsonl_fp.close()


class NewtonAddressMapper:
    """
    configuration.cc / configuration.h의 주소 생성 로직을 Python으로 옮긴 버전
    """

    def __init__(
        self,
        channels: int,
        ranks: int,
        bankgroups: int,
        banks_per_group: int,
        rows: int,
        columns: int,
        bus_width: int,
        bl: int,
        address_mapping: str,
    ):
        self.channels = channels
        self.ranks = ranks
        self.bankgroups = bankgroups
        self.banks_per_group = banks_per_group
        self.rows = rows
        self.columns = columns
        self.bus_width = bus_width
        self.bl = bl
        self.address_mapping = address_mapping

        self.request_size_bytes = (bus_width // 8) * bl
        self.shift_bits = ilog2(self.request_size_bytes)

        col_low_bits = ilog2(bl)
        self.actual_col_bits = ilog2(columns) - col_low_bits

        self.field_widths = {
            "ch": ilog2(channels) if channels > 1 else 0,
            "ra": ilog2(ranks) if ranks > 1 else 0,
            "bg": ilog2(bankgroups) if bankgroups > 1 else 0,
            "ba": ilog2(banks_per_group) if banks_per_group > 1 else 0,
            "ro": ilog2(rows),
            "co": self.actual_col_bits,
        }

        # ex) rochrababgco -> ["ro","ch","ra","ba","bg","co"]
        fields = [address_mapping[i:i + 2] for i in range(0, len(address_mapping), 2)]

        self.field_pos = {}
        pos = 0
        while fields:
            token = fields.pop()
            width = self.field_widths[token]
            self.field_pos[token] = pos
            pos += width

        self.ch_pos = self.field_pos.get("ch", 0)
        self.ra_pos = self.field_pos.get("ra", 0)
        self.bg_pos = self.field_pos.get("bg", 0)
        self.ba_pos = self.field_pos.get("ba", 0)
        self.ro_pos = self.field_pos.get("ro", 0)
        self.co_pos = self.field_pos.get("co", 0)

        self.ch_mask = (1 << self.field_widths["ch"]) - 1 if self.field_widths["ch"] > 0 else 0
        self.ra_mask = (1 << self.field_widths["ra"]) - 1 if self.field_widths["ra"] > 0 else 0
        self.bg_mask = (1 << self.field_widths["bg"]) - 1 if self.field_widths["bg"] > 0 else 0
        self.ba_mask = (1 << self.field_widths["ba"]) - 1 if self.field_widths["ba"] > 0 else 0
        self.ro_mask = (1 << self.field_widths["ro"]) - 1
        self.co_mask = (1 << self.field_widths["co"]) - 1

    def make_address(
        self,
        channel: int,
        rank: int,
        bankgroup: int,
        bank: int,
        row: int,
        col: int,
    ) -> int:
        addr = 0
        if self.field_widths["ch"] > 0:
            addr |= (channel & self.ch_mask) << self.ch_pos
        if self.field_widths["ra"] > 0:
            addr |= (rank & self.ra_mask) << self.ra_pos
        if self.field_widths["bg"] > 0:
            addr |= (bankgroup & self.bg_mask) << self.bg_pos
        if self.field_widths["ba"] > 0:
            addr |= (bank & self.ba_mask) << self.ba_pos

        addr |= (row & self.ro_mask) << self.ro_pos
        addr |= (col & self.co_mask) << self.co_pos

        addr <<= self.shift_bits
        return addr

    def encode_pim_header(
        self,
        channel: int,
        row: int,
        for_gwrite: bool,
        num_comps: int,
        num_readres: int,
    ) -> int:
        """
        configuration.cc의 EncodePIMHeader()를 그대로 옮김
        """
        if not is_pow2(num_comps):
            raise ValueError(f"num_comps는 2의 거듭제곱이어야 합니다. got={num_comps}")
        if not is_pow2(num_readres):
            raise ValueError(f"num_readres는 2의 거듭제곱이어야 합니다. got={num_readres}")

        gwrite_bit = 1 if for_gwrite else 0

        # column field를 metadata로 재사용
        log_comps = (gwrite_bit << self.actual_col_bits) + ilog2(num_comps)
        log_readres = ilog2(num_readres)

        return self.make_address(
            channel=channel,
            rank=log_readres // 16,
            bankgroup=(log_readres // 4) & 0x3,
            bank=log_readres % 4,
            row=row,
            col=log_comps,
        )


class NewtonGemvTracer:
    def __init__(
        self,
        m: int,
        k: int,
        elem_bytes: int,
        row_bytes: int,
        col_bytes: int,
        interarrival: int,
        channel: int,
        channels: int,
        ranks: int,
        bankgroups: int,
        banks_per_group: int,
        rows: int,
        columns: int,
        bus_width: int,
        bl: int,
        address_mapping: str,
    ):
        self.m = m
        self.k = k
        self.elem_bytes = elem_bytes
        self.row_bytes = row_bytes
        self.col_bytes = col_bytes
        self.interarrival = interarrival
        self.channel = channel

        if row_bytes % col_bytes != 0:
            raise ValueError("row_bytes는 col_bytes의 배수여야 합니다.")
        if col_bytes % elem_bytes != 0:
            raise ValueError("col_bytes는 elem_bytes의 배수여야 합니다.")

        self.mapper = NewtonAddressMapper(
            channels=channels,
            ranks=ranks,
            bankgroups=bankgroups,
            banks_per_group=banks_per_group,
            rows=rows,
            columns=columns,
            bus_width=bus_width,
            bl=bl,
            address_mapping=address_mapping,
        )

        self.num_banks = bankgroups * banks_per_group
        self.cluster_size = 4  # Newton 논문의 G_ACT는 4-bank cluster 기준
        self.cols_per_row = row_bytes // col_bytes
        self.elems_per_dram_row = row_bytes // elem_bytes

        if not is_pow2(self.cols_per_row):
            raise ValueError("Newton header의 num_comps 때문에 row_bytes/col_bytes는 2의 거듭제곱이어야 합니다.")

        self.time = 0
        self.trace: List[TraceReq] = []

    def _emit(self, op: str, addr: int, meta: Dict) -> None:
        self.trace.append(
            TraceReq(
                clk=self.time,
                addr=addr,
                op=op,
                meta=meta,
            )
        )
        self.time += self.interarrival

    def generate(self) -> List[TraceReq]:
        """
        Newton 논문의 tiled MV 스케줄을 단순화해서 반영

        바깥 루프: K 방향 chunk (1 DRAM row = 1KB = 512 fp16)
        안쪽 루프: M 방향 vertical tile (bank 수만큼 한 번에 처리)

        각 K chunk마다:
          1) GWRITE x 32  : global buffer 적재
          2) 각 vertical tile마다
             - G_ACT x (num_banks / 4)
             - COMP x 32
             - READRES x 1
        """

        if self.k % self.elems_per_dram_row != 0:
            raise ValueError(
                f"K={self.k} 는 현재 Newton tracer에서 elems_per_dram_row={self.elems_per_dram_row}의 배수여야 합니다."
            )

        num_k_chunks = self.k // self.elems_per_dram_row
        num_row_tiles = ceil_div(self.m, self.num_banks)

        num_comps = self.cols_per_row   # 1KB / 32B = 32
        num_readres = 1                 # READRES 한 번에 모든 bank 결과를 읽는다고 가정

        for k_chunk in range(num_k_chunks):
            # ---------------------------------------
            # 1) 입력 vector chunk를 global buffer에 적재
            # ---------------------------------------
            gwrite_header = self.mapper.encode_pim_header(
                channel=self.channel,
                row=k_chunk,
                for_gwrite=True,
                num_comps=num_comps,
                num_readres=num_readres,
            )

            for subchunk in range(num_comps):
                self._emit(
                    op="GWRITE",
                    addr=gwrite_header,
                    meta={
                        "stage": "load_global_buffer",
                        "k_chunk": k_chunk,
                        "subchunk": subchunk,
                        "num_comps": num_comps,
                        "num_readres": num_readres,
                    },
                )

            # ---------------------------------------
            # 2) vertical tile 단위로 matrix rows 처리
            # ---------------------------------------
            for row_tile in range(num_row_tiles):
                base_row_idx = row_tile * self.num_banks

                # 이 tile에서 실제로 유효한 matrix row 수
                valid_rows = max(0, min(self.num_banks, self.m - base_row_idx))
                if valid_rows <= 0:
                    continue

                # Newton의 interleaved layout에서는
                # 같은 k_chunk의 matrix chunk들이 각 bank의 같은 DRAM row index에 놓인다고 보는 게 자연스러움
                # 여기서는 row address를 "global tile id"로 단순화
                dram_row_for_tile = row_tile * num_k_chunks + k_chunk

                # ---------------------------------------
                # 2-1) 4-bank cluster 단위 G_ACT
                # ---------------------------------------
                num_clusters = ceil_div(valid_rows, self.cluster_size)
                for cluster in range(num_clusters):
                    cluster_header = self.mapper.make_address(
                        channel=self.channel,
                        rank=0,
                        bankgroup=cluster % max(1, self.mapper.bankgroups),
                        bank=0,
                        row=dram_row_for_tile,
                        col=0,
                    )

                    self._emit(
                        op="G_ACT",
                        addr=cluster_header,
                        meta={
                            "stage": "activate_tile",
                            "k_chunk": k_chunk,
                            "row_tile": row_tile,
                            "dram_row": dram_row_for_tile,
                            "cluster": cluster,
                            "rows_covered_start": base_row_idx + cluster * self.cluster_size,
                            "rows_covered_end": min(
                                base_row_idx + (cluster + 1) * self.cluster_size,
                                self.m
                            ) - 1,
                        },
                    )

                # ---------------------------------------
                # 2-2) row-wide compute
                # ---------------------------------------
                comp_header = self.mapper.encode_pim_header(
                    channel=self.channel,
                    row=dram_row_for_tile,
                    for_gwrite=False,
                    num_comps=num_comps,
                    num_readres=num_readres,
                )

                for subchunk in range(num_comps):
                    self._emit(
                        op="COMP",
                        addr=comp_header,
                        meta={
                            "stage": "compute_tile",
                            "k_chunk": k_chunk,
                            "row_tile": row_tile,
                            "dram_row": dram_row_for_tile,
                            "subchunk": subchunk,
                            "valid_rows": valid_rows,
                            "num_comps": num_comps,
                        },
                    )

                # ---------------------------------------
                # 2-3) bank 결과를 한 번에 읽기
                # ---------------------------------------
                self._emit(
                    op="READRES",
                    addr=comp_header,
                    meta={
                        "stage": "read_results",
                        "k_chunk": k_chunk,
                        "row_tile": row_tile,
                        "dram_row": dram_row_for_tile,
                        "valid_rows": valid_rows,
                        "num_readres": num_readres,
                    },
                )

        return self.trace

    def summary(self) -> Dict:
        counts: Dict[str, int] = {}
        for req in self.trace:
            counts[req.op] = counts.get(req.op, 0) + 1

        return {
            "problem": {
                "M": self.m,
                "K": self.k,
                "elem_bytes": self.elem_bytes,
                "equation": f"({self.m}x{self.k}) x ({self.k}x1) = ({self.m}x1)",
            },
            "newton_geometry": {
                "row_bytes": self.row_bytes,
                "col_bytes": self.col_bytes,
                "cols_per_row": self.cols_per_row,
                "elems_per_dram_row": self.elems_per_dram_row,
                "num_banks": self.num_banks,
                "cluster_size": self.cluster_size,
            },
            "mapping": {
                "address_mapping": self.mapper.address_mapping,
                "request_size_bytes": self.mapper.request_size_bytes,
                "shift_bits": self.mapper.shift_bits,
                "actual_col_bits": self.mapper.actual_col_bits,
            },
            "trace_stats": {
                "num_requests": len(self.trace),
                "per_op": counts,
                "first_clk": self.trace[0].clk if self.trace else None,
                "last_clk": self.trace[-1].clk if self.trace else None,
            },
        }


def parse_args():
    p = argparse.ArgumentParser(description="Generate a Newton DRAMsim3 trace for GEMV.")

    # GEMV 크기
    p.add_argument("--m", type=int, default=32, help="A의 row 수")
    p.add_argument("--k", type=int, default=1536, help="A의 col 수 / x의 길이")
    p.add_argument("--elem-bytes", type=int, default=2, help="원소 크기 byte (fp16=2)")

    # DRAM / Newton 구조
    p.add_argument("--dram-row-bytes", type=int, default=1024, help="DRAM row 크기 byte")
    p.add_argument("--dram-col-bytes", type=int, default=32, help="DRAM column I/O 크기 byte")
    p.add_argument("--interarrival", type=int, default=1, help="명령 간 간격 cycle")

    # config와 맞춰야 하는 값들
    p.add_argument("--channel", type=int, default=0, help="사용할 channel index")
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--ranks", type=int, default=1)
    p.add_argument("--bankgroups", type=int, default=2)
    p.add_argument("--banks-per-group", type=int, default=4)
    p.add_argument("--rows", type=int, default=65536)
    p.add_argument("--columns", type=int, default=1024)
    p.add_argument("--bus-width", type=int, default=64)
    p.add_argument("--bl", type=int, default=8)
    p.add_argument("--address-mapping", type=str, default="rochrababgco")

    p.add_argument("--out-dir", type=Path, default=Path("./newton_trace_out"))
    p.add_argument("--prefix", type=str, default="newton_gemv_32x1536_fp16")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tracer = NewtonGemvTracer(
        m=args.m,
        k=args.k,
        elem_bytes=args.elem_bytes,
        row_bytes=args.dram_row_bytes,
        col_bytes=args.dram_col_bytes,
        interarrival=args.interarrival,
        channel=args.channel,
        channels=args.channels,
        ranks=args.ranks,
        bankgroups=args.bankgroups,
        banks_per_group=args.banks_per_group,
        rows=args.rows,
        columns=args.columns,
        bus_width=args.bus_width,
        bl=args.bl,
        address_mapping=args.address_mapping,
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
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nWrote:")
    print(f"  DRAMsim3 trace : {dramsim3_path}")
    print(f"  JSONL trace    : {jsonl_path}")
    print(f"  Summary        : {summary_path}")


if __name__ == "__main__":
    main()