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

Install `pathways-cli` from PyPI using your preferred package manager:

```bash
# Using pip
pip install pathways-cli

# Or using uv (recommended for fast tool management)
uv tool install pathways-cli
```

---

## Usage

Once installed, you can invoke the `pwy` CLI directly:

### 1. Provision / Preview a Cluster (`pwy up`)

Starts a Pathways JobSet on v6e.

```bash
pwy up \
  --tpu-type v6e-16 \
  --gcs-scratch-location gs://my-bucket/pathways-staging
```

Note the above assumes you have a GKE cluster created with a v6e-16 nodepool already provisioned.

#### Key Options:
- `--tpu-type`: **(Required)** TPU type (e.g., `v6e-4`, `v6e-8`, `v6e-16`, `v6e-32`, `v6e-64`).
- `--gcs-scratch-location`: **(Required)** GCS scratch path for pathways synchronization.
- `--num-slices`: Number of TPU slices to run (default: `1`).
- `--jax-client-image`: Custom client container image (default: `python:3.12-slim`).
- `--command`: Run a custom training/eval script in the client container. If omitted, defaults to `sleep infinity` (interactive mode).
- `--spot`: Add node affinity and toleration settings for Spot VMs.
- `--colocated-python`: Enables colocated CPU Python sidecar/init containers on GKE workers and enables external proxy routing.
- `--dry-run`: Prints the generated YAML to stdout instead of calling `kubectl apply`.
- `--name`: Name of the Kubernetes JobSet resource (default: `$USER-pw`).
- `--namespace`: Target Kubernetes namespace (default: `default`).

---

### 2. Teardown a Cluster (`pwy down`)

Deletes the running Pathways JobSet.

```bash
pwy down --name pathways-interactive --namespace default
```

---

### 3. Verification Example

Once the interactive cluster is running, you can verify execution by `exec`ing into the client container:

1. **Find the client pod name**:
   ```bash
   POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$USER-pw,jobset.sigs.k8s.io/replicatedjob-name=pwhd -o jsonpath='{.items[0].metadata.name}')
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

### 4. Running Jupyter Notebook (Interactive Development)

You can spin up a Jupyter Notebook directly inside the JAX client container using the `--command` override:

1. **Launch the cluster with Jupyter Lab**:
   ```bash
   pwy up \
     --tpu-type v6e-4 \
     --gcs-scratch-location gs://my-bucket/pathways-staging \
     --command "pip install jax pathwaysutils jupyterlab && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
   ```

2. **Find the client pod name**:
   ```bash
   POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$USER-pw,jobset.sigs.k8s.io/replicatedjob-name=pwhd -o jsonpath='{.items[0].metadata.name}')
   ```

3. **Port forward to the Jupyter server**:
   ```bash
   kubectl port-forward $POD_NAME 8888:8888
   ```

4. **Access Jupyter Lab** in your browser at `http://localhost:8888`. Create a new notebook and run a JAX device check:
   ```python
   import pathwaysutils
   pathwaysutils.initialize()
   import jax
   print(jax.devices())
   ```

---

## TPU Type Mappings

`pwy` handles all resource-limit math and topologies automatically. It supports a wide range of TPU generations, including:

- **TPU v6e**: `v6e-4` up to `v6e-256` (including `v6e-8-1` with 8 chips per VM)
- **TPU v5p**: `v5p-8` up to `v5p-17920`
- **TPU v5e (v5LitePod)**: `v5litepod-8` up to `v5litepod-256`
- **TPU v4**: `v4-8` up to `v4-4096`
- **TPU 7x**: `7x-8` up to `7x-8192`

---

## Running Tests

To execute the unit test suite:

```bash
uv run pytest
```
