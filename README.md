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

### 5. Running vLLM (Multi-Host TPU Serving) via Pathways

You can deploy and run `vllm-tpu` in multi-host mode using the Pathways backend in a single step by passing the startup command via `--command`. The JAX client container executes the server process, communicating with the worker TPUs over the Pathways proxy.

1. **Launch the Pathways cluster and run vLLM serving in one command**:
   ```bash
   pwy up \
     --tpu-type v6e-16 \
     --gcs-scratch-location gs://my-bucket/pathways-staging \
     --name vllm-pw \
     --command 'until pip install uv && uv pip install --system vllm-tpu pathwaysutils; do echo "Pip failed, retrying in 5s..."; sleep 5; done && JAX_PLATFORMS="proxy,cpu" VLLM_TPU_USING_PATHWAYS=1 TPU_BACKEND_TYPE=jax MODEL_IMPL_TYPE=vllm VLLM_ENABLE_V1_MULTIPROCESSING=0 python3 -m vllm.entrypoints.cli.main serve "Qwen/Qwen3.6-35B-A3B" --load-format dummy --tensor-parallel-size 16 --max-model-len 8192 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.80'
   ```
   *Note: This provisions a JobSet named `vllm-pw` requesting a single slice of TPU v6e-16 (composed of 4 TPU VMs / 16 total chips). The `--command` override handles package installation robustly and launches the server. `--max-model-len` is set to `8192`, `--max-num-batched-tokens` is set to `16384` (required by multimodal validation logic when `--disable-chunked-mm-input` is forced on Qwen 3.6 MoE), and `--gpu-memory-utilization` is restricted to `0.80` to reserve headroom for compile-time allocations.*

2. **Monitor the server installation and logs**:
   Track the package installation progress and JAX model compilation directly from the client container logs:
   ```bash
   POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=vllm-pw,jobset.sigs.k8s.io/replicatedjob-name=pwhd -o jsonpath='{.items[0].metadata.name}')
   kubectl logs -f $POD_NAME -c client
   ```

3. **Verify the server is serving requests**:
   From a separate terminal on your local machine, forward the server port:
   ```bash
   kubectl port-forward $POD_NAME 8000:8000
   ```
   Send a query to the model:
   ```bash
   curl http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "Qwen/Qwen3.6-35B-A3B",
           "prompt": "Pathways is a",
           "max_tokens": 50,
           "temperature": 0.0
       }'
   ```


---

### 6. Running sglang-jax (Multi-Host TPU Serving) via Pathways

You can deploy and run `sglang-jax` in multi-host mode using the Pathways backend by syncing the cloned repository to the client container and starting the server process.

1. **Clone the `sglang-jax` repository**:
   ```bash
   git clone https://github.com/sgl-project/sglang-jax
   ```

2. **Launch the Pathways cluster**:
   ```bash
   pwy up \
     --tpu-type v6e-16 \
     --gcs-scratch-location gs://my-bucket/pathways-staging \
     --name sglang-pw
   ```

3. **Install dependencies and launch the server**:
   Use `pwy run` with the `--source` option pointing to the cloned repository to sync code and launch serving:
   ```bash
   pwy run \
     --name sglang-pw \
     --source sglang-jax \
     --dest /app \
     bash -c 'until pip install uv && uv pip install --system -e /app/python[cpu] pathwaysutils; do echo "Pip failed, retrying in 5s..."; sleep 5; done && JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 JAX_USE_SHARDY_PARTITIONER=0 python3 -u -m sgl_jax.launch_server --model-path Qwen/Qwen2.5-3B-Instruct --load-format dummy --trust-remote-code --tp-size=16 --mem-fraction-static=0.8 --chunked-prefill-size=2048 --download-dir=/tmp --dtype=bfloat16 --max-running-requests 8 --skip-server-warmup --page-size=64 --max-total-tokens=257536 --random-seed=27 --precompile-token-paddings=2048 --precompile-bs-paddings=8 --enable-single-process --attention-backend native'
   ```
   *Note: Using `--attention-backend native` is currently recommended on Pathways to bypass custom Pallas FlashAttention compilation issues due to JAX version mismatches between the sglang-jax package (defaulting to JAX 0.8.1) and the Pathways server (JAX 0.10.0). Once sglang-jax officially upgrades its dependencies to JAX 0.10.0, the optimized attention backend (`--attention-backend fa`, the default) will run out-of-the-box. Setting `--load-format dummy` runs the model with randomly-initialized weights for quick verification without downloading weight files.*

4. **Verify the server is serving requests**:
   Find the client pod name:
   ```bash
   POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=sglang-pw,jobset.sigs.k8s.io/replicatedjob-name=pwhd -o jsonpath='{.items[0].metadata.name}')
   ```
   Forward the server port (default `30000`):
   ```bash
   kubectl port-forward pod/$POD_NAME 30000:30000
   ```
   Send a query to the model using either `/generate` or the OpenAI-compatible `/v1/chat/completions` API:
   ```bash
   # Option A: SGLang native generate endpoint
   curl -X POST 'http://127.0.0.1:30000/generate' \
     -H 'Content-Type: application/json' \
     -d '{"text": "the capital of France is", "sampling_params": {"max_new_tokens": 10, "temperature": 0.6}}'

   # Option B: OpenAI-compatible Chat Completions API
   curl -s -d '{
     "model": "Qwen/Qwen2.5-3B-Instruct",
     "messages": [{"role": "user", "content": "Hello!"}],
     "max_tokens": 16
   }' -H "Content-Type: application/json" http://127.0.0.1:30000/v1/chat/completions
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
