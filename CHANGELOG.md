# Changelog

All notable changes to `scitex-cv` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`ocr_surya()` — a second OCR engine, served by llama.cpp's `llama-server`.** Reads a page through the Surya-2 GGUF plus its multimodal projector over HTTP, so it pulls in **no torch**. That is the point: current torch wheels ship no kernels for Pascal cards (a GTX 1070 is `sm_61`; torch 2.13.0+cu130 covers `sm_75`+), while llama.cpp built with `CMAKE_CUDA_ARCHITECTURES=61` runs on the same card. Returns structured layout — text with per-block bounding boxes and semantic labels — rather than a flat string.
- **`normalize_page()`** — resamples a page so its long edge is 1755px (`SURYA_PAGE_HEIGHT`), preserving aspect. It is a *normalizer*, not an upscaler: it scales **down** as readily as up. Measured 2026-08-10 on three real scanned documents, the model reads only near one working resolution and otherwise answers a different question entirely. A soft scan that failed at its native 753×1073 read correctly once resampled to 1242×1770, and **broke again** when upscaled to 1506×2146 — so "bigger is better" is wrong. The lower bound is universal; the upper bound depends on how sharp the source is (at the same 3205 image tokens a crisp page reads and a soft scan does not). 1755 is the one operating point that produced text for every input tested — chosen for that reason, not as a demonstrated optimum.
- **`is_layout_only()`** — detects the silent failure above. When Surya cannot read a page it does not error: it returns a JSON array of `{label, bbox, count}` blocks with no characters, arriving with `finish_reason="stop"` and a plausible byte count, so it reads as success to anything that only checks the request completed. `SuryaReading.status` is three-valued (`text` / `layout-only` / `empty`) and `text` is `None` rather than `""` on failure, because an empty string is indistinguishable from a genuinely blank page. The reading reports the size actually sent and deliberately does **not** assert a cause — overshooting and undershooting produce the identical symptom.
- **`torch_build_has_kernels_for_local_gpu()`** — asks whether the *installed torch build* ships kernels for this machine's GPU, which is not the question `torch.cuda.is_available()` answers. Named after torch on purpose so the answer cannot be misapplied to engines on other runtimes (CTranslate2, ONNX Runtime, llama.cpp) whose kernel coverage is their own.

### Fixed

- **`ocr()` no longer selects a GPU that cannot run the model.** `easyocr.Reader` defaults to `gpu=True` and trusts `torch.cuda.is_available()`, which returns **True** on a GTX 1070 even though this torch build has no `sm_61` kernels — the driver and runtime are fine, only the kernels are missing. `ocr()` now takes `gpu: Optional[bool]`, defaulting to the capability probe above; pass `True`/`False` to override. The reader cache is keyed on `gpu` as well as the language tuple, so an override cannot return the other device's reader.

## [0.2.0] — 2026-06-27

- **BREAKING(deps): switch `opencv-python` → `opencv-python-headless`.** The headless wheel is self-contained (bundles ffmpeg/libpng/openblas/…) and pulls in no X11/OpenGL/GLib system libraries, so scitex-cv installs and imports on a minimal/headless base with **no apt packages required**. scitex-cv uses no GUI cv2 (`imshow` etc.); if you need it, install `opencv-python` yourself.
- feat(system-deps): register a `scitex_dev.system_deps` provider so the SciTeX container build discovers scitex-cv's OS-level needs via the ecosystem aggregator instead of hardcoding them. With headless OpenCV the verified apt set is **empty**, so the provider returns no packages (earlier dev revisions declared `libxcb1`/`libgl1`/`libglib2.0-0` for full opencv-python; headless drops all three).
- perf(import): defer the cv2 import — `import scitex_cv` (and the system-deps provider module) no longer pulls in cv2 at package load (PEP 562 lazy attributes); public functions import cv2 on first use, unchanged.
- test(integration): add a lazy-import gate proving the package and the provider module import without cv2.

## [0.1.5] — 2026-05-26

- test(quality): rewrite tests for PA-307 TQ001/002/003/007 conformance
- ci(codecov): disable PR comments to stop email noise
- ci(quality): replace broken ecosystem-clone template with single-package audit-all
- ci(docs): make sphinx_html commit-back step non-fatal
- docs(sphinx_html): refresh from CI build

## [0.1.4] — 2026-05-19

- quality: subprocess coverage wiring + [dev] completeness
- fix(workflows): resync integrated release pipeline from scitex-dev v0.11.20
- fix(workflows): standardize to scitex-dev canonical set
- ci(release): sync publish-pypi.yml fix
- ci: sync GitHub Releases with PyPI publish
- ci: sync-main.yml — auto-FF main on v\* tag push
- chore(deps): bump scitex-dev pin floor to 0.11.7
- docs: add CHANGELOG.md + CONTRIBUTING.md
- docs(readme): add Architecture + Demo sections
- docs: add skills leaves per SK105-107 standard template
- docs: various documentation improvements

## [0.1.3] — 2025-11-15

- audit: clear all 11 audit warnings
- fix(release-safety): opt-in publish-pypi.yml (workflow_dispatch only)
- fix(skills): strip trailing `<!-- EOF -->` (SK211)
- fix(api): PA501/PA201/PA203 hygiene
- chore(version): switch `__version__` to importlib.metadata

## [0.1.2]

- Initial CHANGELOG entry — see git log for prior history.
