# Gemini Agent & Developer Guide: `pathways-cli`

This document details the codebase design, architecture, key lessons, and integration verification guides for the `pwy` CLI tool, serving as a developer-facing companion to the user-facing `README.md`.

---

## 1. Codebase Architecture

The project is structured around a standard PEP 621 package layout:

```
/Users/stoelinga/workspace/pathways-cli/
├── pyproject.toml         # Package configurations, CLI scripts, and Pytest options
├── .gitignore             # Excludes local environments, caches, and secrets
├── README.md              # User documentation and example verification steps
├── GEMINI.md              # Codebase design and developer/agent context (this file)
├── src/
│   └── pwy/
│       ├── __init__.py    # Exposes cli entry points
│       ├── cli.py         # click CLI definition: up, down commands
│       ├── generator.py   # Topology math, spot VM toggles, colocated python configurations
│       ├── templates.py   # Complete GKE JobSet multi-line YAML manifest template
│       └── kubernetes.py  # Wrapper invoking kubectl subprocesses
└── tests/
    ├── __init__.py
    ├── test_cli.py        # CLI option validations & mocks
    ├── test_generator.py  # Mappings and string-formatting unit tests
    └── test_e2e.py        # Real GKE cluster integration execution verifying JAX setup
```

---

## 2. Testing Workflows

Verify changes using one of the three testing scopes:

### 1. Unit Tests (Mocked)
Tests calculations and YAML generation without cluster access.
```bash
uv run pytest tests/test_generator.py tests/test_cli.py
```

### 2. End-to-End Integration Tests (Active Cluster)
Runs actual deployments on a running TPU nodepool, installs JAX, executes verification scripts, and tears the setup down.
1. Configure your GCS path in a local `.env` file:
   ```env
   PWY_E2E_GCS_SCRATCH_LOCATION=gs://my-staging-bucket/pathways
   ```
2. Run pytest targeting the `e2e` mark:
   ```bash
   uv run pytest tests/test_e2e.py -m e2e -s
   ```
