# Changelog

All notable user-facing changes to linum-basic are documented here.

## Unreleased

### Performance

- **Device-resident solver for the Torch backend.** The per-pixel sort, the weights and every reweighting pass now stay on the device; only the working-size flat-field and dark-field come back to the host. The ALM iteration carries the multiplier in scaled form and no longer forms the residual inside the loop, so it reads and writes the large arrays about seven times instead of about twenty-five. A per-z mosaic fit of 16 planes of 496 tiles at working size 128 takes about 3 s on one RTX A6000, down from about 19 s, with flat-fields equal to within 3e-5.
- The solver is eager, so nothing is compiled when a process starts, and results no longer depend on whether the step was compiled: the compiled step enabled TF32 matmul and its flat-fields differed from the eager ones by up to 3 %.
- **Multi-GPU fits use one thread per GPU** in the calling process, taking z-levels from a shared queue, instead of one worker process per GPU.
- `apply_fit` corrects the volume in place on one private copy; seam metrics gather the overlap strips instead of whole tiles; small tile stacks are resized without a thread pool.
- Set `LINUM_BASIC_ALM_SOLVER=legacy` to run the previous Torch step (optionally compiled). The NumPy backend is unchanged.

### Fixes

- **Dark-field level.** The offset update averaged the bright-minus-dim contrast over the images before fitting a line through it, which left the line with one point: the denominator was zero up to rounding and the estimate landed on its cap, the minimum of the stack, whatever the data said. The contrast is now kept per image, on every backend. On synthetic stacks with a true offset of 0, 8 and 20 the estimate goes from 4.1, 12.1 and 24.1 (the stack minimum each time) to 0.6, 8.8 and 20.8, and it no longer changes when the input is scaled by 1 + 1e-6.
- **Dark-field convergence.** A dark-field that is zero up to rounding kept the reweighting loop running to `max_reweighting_iterations`, because its change was measured against itself. A dark-field below `darkfield_convergence_floor` (default 1 %) of the mean image intensity is now measured against that fraction of the image level. Set the floor to 0 for the old test.
- **Matmul precision.** Compiling the ALM step no longer switches the process to TF32 matmul. A compiled single-GPU fit used to converge to flat-fields up to 3 % away from an eager multi-GPU fit of the same data.
- **Auto strategy below working size 128.** `strategy="auto"` no longer selects the batched solver there; it uses the per-z paths, which are several times faster with the device-resident solver. `strategy="batched"` and `force_batched_cuda` still select it.
- A compiled dark-field step is no longer reused for a stack with a different minimum (the cap of the offset was fixed when the step was built).

## 2.0.0 — Performance and maturity release

This release delivers faster per-z mosaic fitting, a clearer GPU backend policy, structured library warnings, and stronger documentation and CI coverage. Version remains **2.0.0**; no PyPI publish is included in this milestone summary.

### Mosaic fitting

- **`fit_mosaic`** runs BaSiC independently on each z-level of an OME-Zarr mosaic, with optional global or per-z field aggregation.
- **Auto strategy selection** (`strategy="auto"`) picks sequential single-GPU, multi-GPU fan-out, or batched CUDA execution from workload shape, visible hardware, and explicit overrides.
- Seam-quality metrics (`seam_l1`, `seam_curvature`) support self-supervised evaluation without ground-truth references.
- **Optuna tuning** (`tune`) searches hyperparameters against combined seam metrics on subsampled z-levels.

### Performance

- **Batched CUDA path** fuses multiple z-planes into a single GPU ALM solve when resolution and hardware favour it; z-chunk sizing limits peak memory on large volumes.
- **Multi-GPU concurrency** fans out per-z fits across visible CUDA devices when batched execution is not selected.
- At **working size ≥ 128**, auto strategy never selects batched CUDA without an explicit override — production UAT showed sequential or multi-GPU paths preserve quality at full resolution.
- **`scripts/benchmark_speedup.py`** provides baseline/candidate/compare/concurrency subcommands, quality gates on seam metrics, and JSON artifact bundles (including fast-path configuration files) for reproducible speed/quality A/B workflows.

### GPU and CUDA backend

- **CUDA-only GPU policy:** PyTorch MPS (Apple Silicon Metal) is **not supported** for BaSiC fitting — float64 SVD requirements are unmet on MPS.
- **`backend="auto"`** on machines without CUDA resolves to **NumPy**; use `--backend numpy` on Apple Silicon or `--device cuda:0` on CUDA servers.
- **`is_gpu_backend()`** detects CUDA only, preventing false multi-process fan-out on Mac.
- The CLI rejects `--device mps` early with a clear error message.
- **`torch.compile`** is used when available; compile failures emit a **filterable `UserWarning`** and fall back to eager mode instead of failing silently.

### CLI

- Entry point renamed to **`basic`** with subcommands: `correct`, `fit`, `tune`, `preview`.
- OME-Zarr mosaic workflows are first-class via `basic fit` and `basic tune`.

### Warnings and diagnostics

- ALM **max-iteration convergence notices** now emit **`UserWarning`** (once per process for scalar fits, once per z-batch size for batched CUDA) instead of printing to stdout — consistent with compile-fallback warnings and filterable in notebooks and pipelines.
- Verbose progress remains available via `tqdm` when `verbose=True`.

### CI and documentation

- **`sbh-simulator`** moved from core dependencies to the optional **`validation`** extra; vignette integration tests install via `uv sync --extra validation`.
- **Docs CI** runs the same strict Sphinx build as `make docs` (`-W --keep-going -n`) on every pull request, executing all notebooks during the build.
- Documentation sweep aligned GPU/MPS guidance across README, parameters, CLI help, and API docstrings with the canonical policy in `docs/gpu.md`.
- **`AGENTS.md`** reconciled with `pyproject.toml` (version 2.0.0, GPL-3.0-or-later, `basic` CLI, all optional extras, benchmark harness, expanded invariants).

### Dependencies

- Python **≥ 3.14** required.
- Core stack: NumPy, SciPy, OpenCV, tqdm, zarr, ome-zarr, Optuna, joblib.
- Optional extras: `gpu`, `viz`, `docs`, `dev`, `validation`, `notebooks`.
