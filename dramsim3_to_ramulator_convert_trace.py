import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert DRAMsim3 trace to Ramulator2 LoadStoreTrace format"
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Input DRAMsim3 trace file (e.g., *.trace)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output Ramulator2 trace file",
    )

    args = parser.parse_args()

    with args.src.open("r", encoding="utf-8") as fin, args.dst.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Line {lineno}: expected 3 tokens, got {len(parts)} -> {line}"
                )

            addr, op, clk = parts

            if op == "READ":
                fout.write(f"LD {addr}\n")
            elif op == "WRITE":
                fout.write(f"ST {addr}\n")
            else:
                raise ValueError(f"Line {lineno}: unknown op {op}")


if __name__ == "__main__":
    main()