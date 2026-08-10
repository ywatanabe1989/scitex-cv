# Changelog

All notable changes to `scitex-cv` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] — 2026-08-10

### Added

- **`read_document()` — a scanned PDF in, ordered per-page results out, with a resumable JSON sidecar.** Written beside the input under a stable name, so a consumer reads a result without running Python and an interrupted overnight batch resumes where it stopped rather than at page 1. The sidecar is written after *every* page and renamed atomically; one written only at completion would be useless in exactly the case it exists for. New `pdf` extra (`pip install 'scitex-cv[pdf]'`) supplies pymupdf.
- **Four page states — `text` / `blank` / `unreadable` / `error` — because three could not express the requirement.** Only `text` carries characters; the others are `None`, never `""`, since an empty string is precisely the value that cannot distinguish "nothing was written here" from "nothing was recovered". `unreadable` (every rung tried, all layout-only) is a fact about the **page**; `error` (unreachable endpoint, corrupt page) is a fact about the **run**, where the page may be perfectly readable. `attempts` records how many rungs actually ran, so "fell off the end of the ladder" and "died on rung one" never blur — they call for opposite next actions.
- **Blankness is measured from pixels, not asked of the model.** Surya returns layout-only JSON for a blank page *identically* to an unreadable one, so asking it reported a genuinely blank page as `unreadable` after six wasted rungs. Ink coverage settles it before the first inference call: real documents measure 0.049/0.073/0.051 against a blank page at 0.00000, with the threshold at 0.002 — three orders of magnitude of separation, so a measurement rather than a tuned knob. A blank page now costs zero inference calls instead of six, which compounds on scan stacks full of separator sheets.
- **PDF pages with a real text layer skip OCR entirely**, recording `source="pdf-text-layer"` so a caller can tell which path produced the text — faster and more accurate than rendering and re-reading it.

### Changed

- **Each ladder rung rasterizes afresh from the vector source** rather than rasterizing once and resizing per rung. Resampling was measured to change whether the model reads a page *at an identical image-token count*, so this removes a variable rather than adding one. A page is never resampled twice.
- **Resume policy is asymmetric on purpose.** `text` and `blank` are kept; `error` is **always** retried, because the failure described the run and not the page; `unreadable` is kept only under an unchanged ladder, since a stored "cannot be read" that outlives the renderings that produced it is a negative frozen past its evidence — and caching it permanently would mean improving the ladder had no effect on exactly the pages the improvement was for. The sidecar records a ladder fingerprint to make that decidable.

### Fixed

- **A false ownership claim in `_ocr.py`, corrected in place rather than quietly deleted.** It stated "PDF handling deliberately lives elsewhere (scitex-io owns PDF -> image)". Verified against that package's source: scitex-io carries PDF *metadata* only (XMP embed/read) and has no rasterization at all. The claim deferred to an owner that does not exist — it did not merely mislead, it manufactured a handoff nobody performed, with both sides able to believe the other held it. Kept visible as a correction because reading it was enough to stop the work it named.

## [0.3.0] — 2026-08-10

### Added

- **`ocr_surya()` — a second OCR engine, served by llama.cpp's `llama-server`.** Reads a page through the Surya-2 GGUF plus its multimodal projector over HTTP, so it pulls in **no torch**. That is the point: current torch wheels ship no kernels for Pascal cards (a GTX 1070 is `sm_61`; torch 2.13.0+cu130 covers `sm_75`+), while llama.cpp built with `CMAKE_CUDA_ARCHITECTURES=61` runs on the same card. Returns structured layout — text with per-block bounding boxes and semantic labels — rather than a flat string.
- **A render ladder (`SURYA_RENDER_LADDER`), because a single rendering's failure is not the page's failure.** Measured 2026-08-10 across twelve renderings (four resampling filters × three target heights) of two real scanned documents: a crisp page produced text on **12 of 12**, while a soft ScanSnap scan produced text on only **2 of 12**. Neither the filter nor the target size predicts the outcome alone — at a fixed size and *identical* image-token count, `cubic` and `area` recovered the soft scan while `lanczos4` and `linear` did not. So `ocr_surya()` walks the ladder and stops at the first rendering that is not layout-only, reporting which rung won (`rendering`) and how many were tried (`attempts`). This exploits a measured redundancy instead of claiming to understand where the model's OCR/layout boundary lies. The ladder's *order* rests on one soft page and should be re-derived as more arrive; its *structure* does not depend on that order being optimal.
- **`normalize_page(image, height, filter_name)`** — resamples a page so its long edge is `height`, preserving aspect. It is a *normalizer*, not an upscaler: it scales **down** as readily as up, since upscaling a soft scan too far was measured to break recognition that worked smaller. `filter_name` is not cosmetic — see above.
- **`is_layout_only()`** — detects the silent failure above. When Surya cannot read a page it does not error: it returns a JSON array of `{label, bbox, count}` blocks with no characters, arriving with `finish_reason="stop"` and a plausible byte count, so it reads as success to anything that only checks the request completed. `SuryaReading.status` is three-valued (`text` / `layout-only` / `empty`) and `text` is `None` rather than `""` on failure, because an empty string is indistinguishable from a genuinely blank page. The reading reports the size actually sent and deliberately does **not** assert a cause — overshooting and undershooting produce the identical symptom.
- **`torch_build_has_kernels_for_local_gpu()`** — asks whether the *installed torch build* ships kernels for this machine's GPU, which is not the question `torch.cuda.is_available()` answers. Named after torch on purpose so the answer cannot be misapplied to engines on other runtimes (CTranslate2, ONNX Runtime, llama.cpp) whose kernel coverage is their own.

### Fixed

- **`pip install scitex-cv[all]` now actually installs everything public.** The `ocr` extra was missing from `all` (PS-221 §3), so the documented give-me-everything install silently omitted EasyOCR and `scitex_cv.ocr` then raised `ImportError` — a pre-existing gap since the extra was introduced, surfaced by CI on this branch. `all` now references `scitex-cv[dev,docs,ocr]`. This makes `[all]` heavier, since easyocr pulls torch; an `[all]` that does not mean all is the worse surprise. The new Surya engine needs no extra at all — it is an HTTP call.
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
