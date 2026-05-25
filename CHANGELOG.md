# Changelog

All notable changes to `scitex-cv` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
