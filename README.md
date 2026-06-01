# `pwy`: Standalone Pathways GKE Cluster CLI Tool

`pwy` is a lightweight, standalone Python CLI utility designed to generate, apply, and manage interactive Pathways workloads on Google Kubernetes Engine (GKE) using Kubernetes JobSets.

---

## Features

- **Automated TPU Topology Calculations**: Translates simple TPU resource types (`v6e-4`, `v6e-16`, etc.) into GKE topologies, VM counts, and instance settings.
- **Spot VM Support**: Dynamically injects GKE node selectors and tolerations for running workloads on cost-effective Spot VMs.
- **Colocated Python Support**: Simplifies distributed checkpointing (e.g. via Orbax) by configuring and enabling colocated host CPU sidecars and proxy endpoints automatically.
- **Interactive & Batch Execution**: Supports spinning up pathways servers with infinite sleep drivers for interactive debugging, or executing training scripts directly.
- **Dry-run Manifest Generation**: Preview and inspect the GKE JobSet manifest without applying it to the cluster.

---

## Installation

This project utilizes [uv](https://github.com/astral-sh/uv) for fast, modern Python package and dependency management.

To sync the environment and install `pwy`:

```bash
uv sync
```

---

## Usage

You can invoke `pwy` commands directly using `uv run`:

### 1. Provision / Preview a Cluster (`pwy up`)

Starts a Pathways JobSet or dry-runs the configuration.

```bash
uv run pwy up \
  --tpu-type v6e-16 \
  --gcs-scratch-location gs://my-bucket/pathways-staging \
  --num-slices 1 \
  --dry-run
```

#### Key Options:
- `--tpu-type`: **(Required)** TPU type (e.g., `v6e-4`, `v6e-8`, `v6e-16`, `v6e-32`, `v6e-64`).
- `--gcs-scratch-location`: **(Required)** GCS scratch path for pathways synchronization.
- `--num-slices`: Number of TPU slices to run (default: `1`).
- `--jax-client-image`: Custom client container image (default: `python:3.12-slim`).
- `--command`: Run a custom training/eval script in the client container. If omitted, defaults to `sleep infinity` (interactive mode).
- `--enable-spot`: Add node affinity and toleration settings for Spot VMs.
- `--colocated-python`: Enables colocated CPU Python sidecar/init containers on GKE workers and enables external proxy routing.
- `--dry-run`: Prints the generated YAML to stdout instead of calling `kubectl apply`.
- `--name`: Name of the Kubernetes JobSet resource (default: `pathways-interactive`).
- `--namespace`: Target Kubernetes namespace (default: `default`).

---

### 2. Teardown a Cluster (`pwy down`)

Deletes the running Pathways JobSet.

```bash
uv run pwy down --name pathways-interactive --namespace default
```

---

### 3. Verification Example

Once the interactive cluster is running, you can verify execution by `exec`ing into the client container:

1. **Find the client pod name**:
   ```bash
   POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=pathways-interactive -o jsonpath='{.items[?(@.metadata.labels.jobset\\.sigs\\.k8s\\.io/replicatedjob-name=="pwhd")].metadata.name}')
   ```

2. **Install JAX and Pathways utils**:
   ```bash
   kubectl exec $POD_NAME -c client -- pip install jax pathwaysutils
   ```

3. **Run a Python snippet to initialize and list devices**:
   ```bash
   kubectl exec $POD_NAME -c client -- python3 -c "import pathwaysutils; pathwaysutils.initialize(); import jax; print(jax.devices())"
   ```

   The command output should print the available virtual TPU devices (e.g., coordinates and memory spaces of the allocated chips).

---

## TPU Type Mappings

`pwy` handles all resource-limit math and topologies automatically according to the following matrix:

| TPU Type | GKE Topology | VMs Per Slice | RM Instance Type |
| :--- | :--- | :--- | :--- |
| `v6e-4` | `2x2` | 1 | `tpuv6e:2x2` |
| `v6e-8` | `2x4` | 2 | `tpuv6e:2x4` |
| `v6e-16` | `4x4` | 4 | `tpuv6e:4x4` |
| `v6e-32` | `4x8` | 8 | `tpuv6e:4x8` |
| `v6e-64` | `8x8` | 16 | `tpuv6e:8x8` |

---

## Running Tests

To execute the unit test suite:

```bash
uv run pytest
```
