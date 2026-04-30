#!/usr/bin/env python3
import argparse

CO_SHIFT = 6
BG_SHIFT = 13
BA_SHIFT = 14
RA_SHIFT = 16
RO_SHIFT = 17

CO_MASK = (1 << 7) - 1
BG_MASK = (1 << 1) - 1
BA_MASK = (1 << 2) - 1
RA_MASK = (1 << 1) - 1
RO_MASK = (1 << 16) - 1

BANKS_PER_GROUP = 4


def decode_addr(addr: int):
    co = (addr >> CO_SHIFT) & CO_MASK
    bg = (addr >> BG_SHIFT) & BG_MASK
    ba = (addr >> BA_SHIFT) & BA_MASK
    ra = (addr >> RA_SHIFT) & RA_MASK
    ro = (addr >> RO_SHIFT) & RO_MASK
    ch = 0
    bank = bg * BANKS_PER_GROUP + ba
    return ro, ra, bank, bg, ba, co, ch


def convert_trace(input_path, output_path):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        fout.write("# ro ra bank bg ba co ch cmd clk original_addr\n")

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 3:
                print(f"Skip malformed line {line_no}: {line}")
                continue

            addr_str = parts[0]
            cmd = parts[1]
            clk = parts[2]

            addr = int(addr_str, 16)

            ro, ra, bank, bg, ba, co, ch = decode_addr(addr)

            fout.write(
                f"{ro:5d} {ra:2d} {bank:4d} {bg:2d} {ba:2d} "
                f"{co:4d} {ch:2d} {cmd:5s} {clk:>5s} {addr_str}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_trace")
    parser.add_argument("output_trace")
    args = parser.parse_args()

    convert_trace(args.input_trace, args.output_trace)


if __name__ == "__main__":
    main()

## error
# 같은 파일을 2번 convert하면 짝수번째 줄만 convert됨. 