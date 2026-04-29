"""Fix Qwen3.5 checkpoint key prefixes for vLLM compatibility.

The training model saves keys as:
  model.language_model.language_model.*  (text LM weights)
  model.language_model.visual.*          (vision encoder weights)

But vLLM expects:
  model.language_model.*                 (text LM weights)
  model.visual.*                         (vision encoder weights)

This script rewrites the safetensors files and index in-place.
"""

import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def fix_key(key: str) -> str:
    if key.startswith("model.language_model.language_model."):
        return key.replace("model.language_model.language_model.", "model.language_model.", 1)
    if key.startswith("model.language_model.visual."):
        return key.replace("model.language_model.visual.", "model.visual.", 1)
    return key


def fix_checkpoint(ckpt_dir: Path) -> None:
    safetensor_files = sorted(ckpt_dir.glob("*.safetensors"))
    if not safetensor_files:
        print(f"No safetensors files found in {ckpt_dir}")
        sys.exit(1)

    for sf_path in safetensor_files:
        print(f"Fixing {sf_path.name} ...")
        tensors = {}
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[fix_key(key)] = f.get_tensor(key)

        tmp_path = sf_path.with_suffix(".tmp")
        save_file(tensors, tmp_path, metadata={"format": "pt"})
        tmp_path.replace(sf_path)

    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        print("Fixing model.safetensors.index.json ...")
        with open(index_path) as f:
            index = json.load(f)
        index["weight_map"] = {fix_key(k): v for k, v in index["weight_map"].items()}
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, sort_keys=True)
            f.write("\n")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <checkpoint_dir>")
        sys.exit(1)
    fix_checkpoint(Path(sys.argv[1]))
