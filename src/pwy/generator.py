from pwy.templates import YAML_TEMPLATE

TPU_MAPPINGS = {
    "v6e-4": {"topology": "2x2", "vms_per_slice": 1, "rm_type": "tpuv6e:2x2"},
    "v6e-8": {"topology": "2x4", "vms_per_slice": 2, "rm_type": "tpuv6e:2x4"},
    "v6e-16": {"topology": "4x4", "vms_per_slice": 4, "rm_type": "tpuv6e:4x4"},
    "v6e-32": {"topology": "4x8", "vms_per_slice": 8, "rm_type": "tpuv6e:4x8"},
    "v6e-64": {"topology": "8x8", "vms_per_slice": 16, "rm_type": "tpuv6e:8x8"},
}

def get_colocated_python_image(client_image: str) -> str:
    if "/" in client_image and ":" in client_image:
        try:
            path, tag = client_image.rsplit(":", 1)
            repo, _ = path.rsplit("/", 1)
            return f"{repo}/colocated-python:{tag}"
        except Exception:
            pass
    return "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/colocated-python:jax-0.9.2"

def generate_yaml(
    name: str,
    namespace: str,
    tpu_type: str,
    gcs_scratch_location: str,
    num_slices: int = 1,
    jax_client_image: str = "python:3.12-slim",
    command: str = None,
    enable_spot: bool = False,
    colocated_python: bool = False,
) -> str:
    if tpu_type not in TPU_MAPPINGS:
        raise ValueError(
            f"Unsupported TPU type: {tpu_type}. Supported types: {list(TPU_MAPPINGS.keys())}"
        )
    
    mapping = TPU_MAPPINGS[tpu_type]
    gke_topology = mapping["topology"]
    vms_per_slice = mapping["vms_per_slice"]
    rm_instance_type = mapping["rm_type"]
    
    # Format client execution command
    if not command:
        client_command = "sleep infinity"
    else:
        client_command = command
    
    # Format Spot VM Node Selector and Tolerations
    if enable_spot:
        spot_toleration_head = (
            '                - key: "cloud.google.com/gke-spot"\\n'
            '                  operator: "Equal"\\n'
            '                  value: "true"\\n'
            '                  effect: "NoSchedule"'
        )
        spot_node_selector_worker = '                cloud.google.com/gke-spot: "true"'
        spot_toleration_worker = (
            '                - key: "cloud.google.com/gke-spot"\\n'
            '                  operator: "Equal"\\n'
            '                  value: "true"\\n'
            '                  effect: "NoSchedule"'
        )
    else:
        spot_toleration_head = ""
        spot_node_selector_worker = ""
        spot_toleration_worker = ""
    
    # Format colocated python options
    if colocated_python:
        proxy_sidecar_arg = "\\n                    - --sidecar_name=external"
        tpu_premapped_buffer_size = 34359738368  # 32 GiB
        colocated_img = get_colocated_python_image(jax_client_image)
        worker_init_containers = (
            "              initContainers:\\n"
            "                - name: colocated-python\\n"
            f"                  image: {colocated_img}\\n"
            "                  imagePullPolicy: Always\\n"
            "                  restartPolicy: Always\\n"
            "                  ports:\\n"
            "                    - containerPort: 50051\\n"
            "                      protocol: TCP\\n"
            "                  env:\\n"
            "                    - name: CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY\\n"
            "                      value: /tmp/ifrt_proxy\\n"
            "                    - name: GRPC_SERVER_ADDRESS\\n"
            "                      value: 0.0.0.0:50051\\n"
            "                  volumeMounts:\\n"
            "                    - name: shared-memory\\n"
            "                      mountPath: /tmp/ifrt_proxy"
        )
    else:
        proxy_sidecar_arg = ""
        tpu_premapped_buffer_size = 274877906944  # 256 GiB
        worker_init_containers = ""
    
    # Interpolate variables in the template
    yaml_content = YAML_TEMPLATE.format(
        NAME=name,
        NAMESPACE=namespace,
        CLIENT_IMAGE=jax_client_image,
        CLIENT_EXECUTION_COMMAND=client_command,
        TPU_TYPE=tpu_type,
        NUM_SLICES=num_slices,
        RM_INSTANCE_TYPE=rm_instance_type,
        GCS_SCRATCH_LOCATION=gcs_scratch_location,
        GKE_TOPOLOGY=gke_topology,
        VMS_PER_SLICE=vms_per_slice,
        SPOT_TOLERATION_HEAD=spot_toleration_head,
        SPOT_NODE_SELECTOR_WORKER=spot_node_selector_worker,
        SPOT_TOLERATION_WORKER=spot_toleration_worker,
        PROXY_SIDECAR_ARG=proxy_sidecar_arg,
        TPU_PREMAPPED_BUFFER_SIZE=tpu_premapped_buffer_size,
        WORKER_INIT_CONTAINERS=worker_init_containers,
    )
    
    # Clean up empty lines caused by optional block placeholders
    # (specifically ensuring there are no lines with only whitespace or empty lines where placeholders were)
    lines = []
    for line in yaml_content.splitlines():
        if line.strip() or line == "":
            lines.append(line)
    return "\\n".join(lines) + "\\n"
