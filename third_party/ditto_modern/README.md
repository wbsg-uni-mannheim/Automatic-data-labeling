# ditto_modern

This module provides the Ditto training and evaluation runtime used for all Ditto students of the paper. `ORIGIN.md` describes its relation to the original Ditto implementation.

Scope:
- Keep Ditto-style pair-text formulation (`COL <attr> VAL <value>`)
- Use current Hugging Face / PyTorch APIs
- Support single-GPU and DDP (`torchrun`)
- Keep WDC json.gz schema compatibility

