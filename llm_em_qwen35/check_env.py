#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform


def version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    import torch

    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_version={torch.version.cuda}")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gb = props.total_memory / (1024**3)
            print(f"gpu[{idx}]={props.name} vram_gb={gb:.1f}")
    print(f"transformers={version('transformers')}")
    print(f"datasets={version('datasets')}")
    print(f"trl={version('trl')}")
    print(f"peft={version('peft')}")
    print(f"unsloth={version('unsloth')}")


if __name__ == "__main__":
    main()

