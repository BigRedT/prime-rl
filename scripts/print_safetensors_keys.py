"""Print all keys in a directory of safetensors files."""

import sys
from pathlib import Path

from safetensors import safe_open


def main(ckpt_dir: Path) -> None:
    files = sorted(ckpt_dir.glob("*.safetensors"))
    if not files:
        print(f"No safetensors files in {ckpt_dir}")
        sys.exit(1)

    for sf_path in files:
        print(f"\n=== {sf_path.name} ===")
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                print(key)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <checkpoint_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]))
