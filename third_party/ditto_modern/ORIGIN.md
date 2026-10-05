# Origin

This directory reimplements the Ditto matcher of Li et al. (PVLDB 2020, https://github.com/megagonlabs/ditto) on current Hugging Face Transformers and PyTorch APIs. It keeps Ditto's pair serialization (`COL <attribute> VAL <value>`), its data augmentation, and its fine-tuning of a pretrained encoder (RoBERTa-base in the paper). All Ditto students of the paper were trained with this implementation through `scripts/training/train_ditto.py`.
