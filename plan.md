# Implementation Plan: Standalone Pathways Cluster CLI Tool (`pwy`)

This document outlines the final design and detailed implementation specifications for a standalone Python CLI tool to generate, apply, and manage interactive Pathways GKE cluster manifests.

---

## 1. CLI Commands & Arguments

The CLI binary/entry point will be named `pwy`.

### Commands

#### 1. `pwy up`
Starts the cluster or dry-runs the configuration.

```bash
pwy up \
  --tpu-type=v6e-4 \
  --gcs-scratch-location=gs://my-bucket/staging \
  [--num-slices=1] \
  [--jax-client-image=python:3.12-slim] \
  [--command="python my_script.py"] \
  [--enable-spot] \
  [--colocated-python] \
  [--dry-run] \
  [--name=pathways-interactive] \
  [--namespace=default]
```

* **Behavior**:
  * Calculates cluster configuration based on `--tpu-type` and `--num-slices`.
  * Generates the JobSet YAML.
  * If `--dry-run` is set: Prints the generated YAML to stdout and exits.
  * Otherwise: Pipes the YAML directly to `kubectl apply -f -`.

#### 2. `pwy down`
Tears down the cluster.

```bash
pwy down [--name=pathways-interactive] [--namespace=default]
```
* **Behavior**: Runs `kubectl delete jobset <name> --namespace=<namespace>`.

---

## 2. Automated TPU Mappings & Configuration Math

The tool automatically maps the user-provided `--tpu-type` to GKE resources, topologies, and arguments.

### Mappings Database

| TPU Type | GKE Topology | VMs Per Slice (`vms_per_slice`) | RM Instance Type (`rm_instance_type`) |
| :--- | :--- | :--- | :--- |
| `v6e-4` | `2x2` | 1 | `tpuv6e:2x2` |
| `v6e-8` | `2x4` | 2 | `tpuv6e:2x4` |
| `v6e-16` | `4x4` | 4 | `tpuv6e:4x4` |
| `v6e-32` | `4x8` | 8 | `tpuv6e:4x8` |
| `v6e-64` | `8x8` | 16 | `tpuv6e:8x8` |

### Configuration Math

For any given run with `tpu-type` and `num-slices`:

* **Pathways Head (`pwhd` replicated job)**:
  * `replicas` is always `1`.
* **Pathways Worker (`pwwk` replicated job)**:
  * `replicas` (number of slices) = `--num-slices`
  * `spec.parallelism` (VMs per slice) = `vms_per_slice`
  * `spec.completions` (VMs per slice) = `vms_per_slice`
  * `resources.limits["google.com/tpu"]` = `4` (always 4 for v6e)
  * `nodeSelector["cloud.google.com/gke-tpu-topology"]` = `GKE Topology`
  * `nodeSelector["cloud.google.com/gke-tpu-accelerator"]` = `tpu-v6e-slice`
* **Resource Manager container (`pathways-rm` sidecar)**:
  * `--instance_count` = `--num-slices`
  * `--instance_type` = `rm_instance_type`

---

## 3. Client Command & Image Execution Logic

The client container `command` field is generated dynamically based on the `--jax-client-image`, `--command`, and `--colocated-python` flags:

1. **JAX Client Image**: 
   * Uses `--jax-client-image` (defaulting to `python:3.12-slim`).
2. **Command Executed**:
   * **If `--command` is NOT provided**:
     ```bash
     bash -c "sleep infinity"
     ```
   * **If `--command` IS provided** (e.g. `--command="python training.py"`):
     The tool boots up the environment, initializes pathways, and then executes the custom command directly:
     ```bash
     bash -c "python training.py"
     ```

---

## 4. Full Manifest Reference Template

Below is the complete reference JobSet YAML template populated with default/baseline variables for a `v6e-4` single-slice run. The CLI generator code should output a structure identical to this:

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: {NAME}
  namespace: {NAMESPACE}
spec:
  failurePolicy:
    maxRestarts: 0
    restartStrategy: BlockingRecreate
  replicatedJobs:
    # -------------------------------------------------------------------------
    # 1. Pathways Head (Client Pod)
    # -------------------------------------------------------------------------
    - name: pwhd
      replicas: 1
      template:
        spec:
          parallelism: 1
          completions: 1
          backoffLimit: 32
          template:
            metadata:
              annotations:
                cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
            spec:
              terminationGracePeriodSeconds: 60
              restartPolicy: Never
              hostAliases:
                - ip: 169.254.169.254
                  hostnames:
                    - metadata
                    - metadata.google.internal
              tolerations:
                - key: google.com/tpu
                  operator: Equal
                  value: "present"
                  effect: NoSchedule
                # Spot toleration only added if --enable-spot is True
                {SPOT_TOLERATION_HEAD}
              containers:
                - name: client
                  image: {CLIENT_IMAGE}
                  command:
                    - bash
                    - -c
                    - |
                      {CLIENT_EXECUTION_COMMAND}
                  resources:
                    requests:
                      cpu: "1000m"
                      memory: "16Gi"
                    limits:
                      cpu: "1000m"
                      memory: "16Gi"
                  env:
                    - name: TPU_TYPE
                      value: {TPU_TYPE}
                    - name: NUM_TPU_SLICES
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']
                    - name: JAX_BACKEND_TARGET
                      value: grpc://localhost:29000
                    - name: XCLOUD_ENVIRONMENT
                      value: GCP
                    - name: JAX_PLATFORMS
                      value: proxy
                    - name: ENABLE_PATHWAYS_PERSISTENCE
                      value: "1"
                    - name: TPU_SKIP_MDS_QUERY
                      value: "true"
                    - name: PYTHONUNBUFFERED
                      value: "1"
                    - name: TEST_UNDECLARED_OUTPUTS_DIR
                      value: "true"
                    - name: IFRT_PROXY_LARGE_TRANSFER_THRESHOLD
                      value: "1"
                    - name: IFRT_PROXY_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY
                      value: /tmp/ifrt_proxy
                  volumeMounts:
                    - name: shared-memory
                      mountPath: /tmp/ifrt_proxy
                  imagePullPolicy: Always
              initContainers:
                - name: pathways-proxy
                  image: us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:jax-0.9.2
                  restartPolicy: Always
                  ports:
                    - containerPort: 29000
                  env:
                    - name: IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS
                      value: "true"
                    - name: IFRT_PROXY_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY
                      value: /tmp/ifrt_proxy
                  args:
                    - --resource_manager_address=localhost:29001
                    - --server_port=29000
                  volumeMounts:
                    - name: shared-memory
                      mountPath: /tmp/ifrt_proxy
                - name: pathways-rm
                  image: us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.2
                  restartPolicy: Always
                  env:
                    - name: TPU_SKIP_MDS_QUERY
                      value: "true"
                  args:
                    - --server_port=29001
                    - --node_type=resource_manager
                    - --instance_count={NUM_SLICES}
                    - --instance_type={RM_INSTANCE_TYPE}
                    - --gcs_scratch_location={GCS_SCRATCH_LOCATION}
              volumes:
                - name: shared-memory
                  emptyDir:
                    medium: Memory
              serviceAccountName: default
              dnsPolicy: ClusterFirstWithHostNet

    # -------------------------------------------------------------------------
    # 2. Pathways Workers (TPU Pods)
    # -------------------------------------------------------------------------
    - name: pwwk
      replicas: {NUM_SLICES}
      template:
        metadata:
          annotations:
            alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
        spec:
          parallelism: {VMS_PER_SLICE}
          completions: {VMS_PER_SLICE}
          backoffLimit: 32
          template:
            spec:
              terminationGracePeriodSeconds: 60
              hostAliases:
                - ip: 169.254.169.254
                  hostnames:
                    - metadata
                    - metadata.google.internal
              nodeSelector:
                cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
                cloud.google.com/gke-tpu-topology: {GKE_TOPOLOGY}
                {SPOT_NODE_SELECTOR_WORKER}
              tolerations:
                # Spot toleration only added if --enable-spot is True
                {SPOT_TOLERATION_WORKER}
              containers:
                - name: worker
                  image: us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.9.2
                  imagePullPolicy: Always
                  ports:
                    - containerPort: 8471
                    - containerPort: 8080
                    - containerPort: 8431
                    - containerPort: 9000
                    - containerPort: 29001
                  securityContext:
                    privileged: true
                  resources:
                    limits:
                      google.com/tpu: 4
                  env:
                    - name: TPU_TYPE
                      value: {TPU_TYPE}
                    - name: NUM_TPU_SLICES
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']
                    - name: MEGASCALE_COORDINATOR_ADDRESS
                      value: {NAME}-pwhd-0-0.{NAME}
                    - name: MEGASCALE_NUM_SLICES
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']
                    - name: MEGASCALE_SLICE_ID
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/job-index']
                  args:
                    - --server_port=29001
                    - --resource_manager_address={NAME}-pwhd-0-0.{NAME}:29001
                    - --gcs_scratch_location={GCS_SCRATCH_LOCATION}
                    - --tpu_pinned_host_allocation_recycle=true
                    - --tpu_premapped_buffer_size=274877906944
              serviceAccountName: default
              dnsPolicy: ClusterFirstWithHostNet
  successPolicy:
    operator: All
    targetReplicatedJobs:
      - pwhd
```

---

## 5. Repository Layout & Target Files

A new standalone directory structure will be created under the workspace (mocking a new repository context):

```
/Users/stoelinga/workspace/pathways-cli/
├── pyproject.toml
├── README.md
├── pwy/
│   ├── __init__.py
│   ├── cli.py             # Entry point (Click CLI commands: up, down)
│   ├── generator.py       # Topology mapping and dictionary interpolation
│   ├── templates.py       # Text template holding the YAML manifest structure
│   └── kubernetes.py      # Subprocess module executing "kubectl apply -f" or "kubectl delete"
└── tests/
    ├── __init__.py
    ├── test_generator.py
    └── test_cli.py
```

### File Implementation Details

#### `pwy/generator.py`
Contains the lookup dictionaries and mapping functions:
```python
TPU_MAPPINGS = {
    "v6e-4": {"topology": "2x2", "vms_per_slice": 1, "rm_type": "tpuv6e:2x2"},
    "v6e-8": {"topology": "2x4", "vms_per_slice": 2, "rm_type": "tpuv6e:2x4"},
    "v6e-16": {"topology": "4x4", "vms_per_slice": 4, "rm_type": "tpuv6e:4x4"},
    "v6e-32": {"topology": "4x8", "vms_per_slice": 8, "rm_type": "tpuv6e:4x8"},
    "v6e-64": {"topology": "8x8", "vms_per_slice": 16, "rm_type": "tpuv6e:8x8"},
}

def generate_yaml(
    name: str,
    namespace: str,
    tpu_type: str,
    gcs_scratch_location: str,
    num_slices: int = 1,
    jax_client_image: str = "python:3.12-slim",
    command: str = None,
    enable_spot: bool = False,
) -> str:
    # 1. Look up TPU type mappings
    # 2. Format client container commands
    # 3. Handle spot nodeSelector and tolerations formatting
    # 4. Interpolate templates.YAML_TEMPLATE with final string variables
    ...
```

#### `pwy/cli.py`
Handles options parsing and commands:
* Imports `generate_yaml`.
* Runs `kubectl apply` or `kubectl delete` using Python's `subprocess.run(..., input=yaml_content.encode())`.

