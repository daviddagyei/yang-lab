# Original RDBM training code

This folder preserves the repository's original image-restoration trainer and
dataset loader. The current multicoil MRI experiments do not use these entry
points. They remain here for architecture provenance and reproducibility.

The shared `rdbm.py` and `networks.py` modules remain at the project root because
the active MRI models import them directly.
