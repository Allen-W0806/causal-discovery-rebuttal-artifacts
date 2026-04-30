# DYNOTEARS Setup

This folder contains a minimal local vendor drop of the DYNOTEARS implementation from the CausalNex project.

## Source

- Upstream project: `mckinsey/causalnex`
- Upstream branch used as reference: `develop`
- DYNOTEARS reference file:
  `https://github.com/mckinsey/causalnex/blob/develop/causalnex/structure/dynotears.py`
- Paper:
  `DYNOTEARS: Structure Learning from Time-Series Data` (AISTATS 2020)

## Main Implementation

- DYNOTEARS is implemented in:
  `causalnex/structure/dynotears.py`

## Included Files

- `causalnex/structure/dynotears.py`
  Upstream DYNOTEARS implementation.
- `causalnex/structure/structuremodel.py`
  Upstream `StructureModel` dependency used by DYNOTEARS.
- `causalnex/structure/transformers.py`
  Upstream `DynamicDataTransformer` dependency used by DYNOTEARS.
- `causalnex/__init__.py`
  Minimal local package shim for this trimmed vendor drop.
- `causalnex/structure/__init__.py`
  Minimal local package shim that exposes `StructureModel` without pulling unrelated CausalNex modules.
- `LICENSE.md`
  Upstream license notice copied from CausalNex.

## Dependencies Likely Needed Later

For this trimmed DYNOTEARS subset, the main Python dependencies are:

- `numpy`
- `pandas`
- `scipy`
- `networkx`
- `scikit-learn`

Notes:

- The full CausalNex repository has additional optional dependencies, but they are not required for the vendored files in this folder.
- The local `__init__.py` shims are intentionally minimal so this folder does not pull in unrelated modules such as the PyTorch DAG estimators.

## Folder Organization

- `causalnex/`
  Minimal package root for the vendored subset.
- `causalnex/structure/`
  Contains the DYNOTEARS implementation and the two direct support modules it imports.
- `LICENSE.md`
  Upstream license notice.

No experiments, benchmarks, result folders, or generated outputs are included here.
