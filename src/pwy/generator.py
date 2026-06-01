from pwy.templates import YAML_TEMPLATE

TPU_MAPPINGS = {
    # v6e
    "v6e-4": {
        "topology": "2x2",
        "vms_per_slice": 1,
        "gke_accelerator": "tpu-v6e-slice",
        "rm_type": "tpuv6e:2x2",
        "chips_per_vm": 4,
    },
    "v6e-8": {
        "topology": "2x4",
        "vms_per_slice": 2,
        "gke_accelerator": "tpu-v6e-slice",
        "rm_type": "tpuv6e:2x4",
        "chips_per_vm": 4,
    },
    "v6e-8-1": {
        "topology": "2x4",
        "vms_per_slice": 1,
        "gke_accelerator": "tpu-v6e-slice",
        "rm_type": "tpuv6e:2x4",
        "chips_per_vm": 8,
    },
    "v6e-16": {
        "topology": "4x4",
        "vms_per_slice": 4,
        "gke_accelerator": "tpu-v6e-slice",
        "rm_type": "tpuv6e:4x4",
        "chips_per_vm": 4,
    },
    "v6e-32": {
        "topology": "4x8",
        "vms_per_slice": 8,
        "gke_accelerator": "tpu-v6e-slice",
        "rm_type": "tpuv6e:4x8",
        "chips_per_vm": 4,
    },
    "v6e-64": {
        "topology": "8x8",
        "vms_per_slice": 16,
        "gke_accelerator": "tpu-v6e-slice",
        "rm_type": "tpuv6e:8x8",
        "chips_per_vm": 4,
    },
    "v6e-128": {
        "topology": "8x16",
        "vms_per_slice": 32,
        "gke_accelerator": "tpu-v6e-slice",
        "rm_type": "tpuv6e:8x16",
        "chips_per_vm": 4,
    },
    "v6e-256": {
        "topology": "16x16",
        "vms_per_slice": 64,
        "gke_accelerator": "tpu-v6e-slice",
        "rm_type": "tpuv6e:16x16",
        "chips_per_vm": 4,
    },
    # v5litepod
    "v5litepod-8": {
        "topology": "2x4",
        "vms_per_slice": 2,
        "gke_accelerator": "tpu-v5-lite-podslice",
        "rm_type": "tpuv5litepod:2x4",
        "chips_per_vm": 4,
    },
    "v5litepod-16": {
        "topology": "4x4",
        "vms_per_slice": 4,
        "gke_accelerator": "tpu-v5-lite-podslice",
        "rm_type": "tpuv5litepod:4x4",
        "chips_per_vm": 4,
    },
    "v5litepod-32": {
        "topology": "4x8",
        "vms_per_slice": 8,
        "gke_accelerator": "tpu-v5-lite-podslice",
        "rm_type": "tpuv5litepod:4x8",
        "chips_per_vm": 4,
    },
    "v5litepod-64": {
        "topology": "8x8",
        "vms_per_slice": 16,
        "gke_accelerator": "tpu-v5-lite-podslice",
        "rm_type": "tpuv5litepod:8x8",
        "chips_per_vm": 4,
    },
    "v5litepod-128": {
        "topology": "8x16",
        "vms_per_slice": 32,
        "gke_accelerator": "tpu-v5-lite-podslice",
        "rm_type": "tpuv5litepod:8x16",
        "chips_per_vm": 4,
    },
    "v5litepod-256": {
        "topology": "16x16",
        "vms_per_slice": 64,
        "gke_accelerator": "tpu-v5-lite-podslice",
        "rm_type": "tpuv5litepod:16x16",
        "chips_per_vm": 4,
    },
    # 7x
    "7x-8": {
        "topology": "2x2x1",
        "vms_per_slice": 1,
        "gke_accelerator": "tpu7x",
        "rm_type": "tpu7x:2x2x1",
        "chips_per_vm": 4,
    },
    "7x-16": {
        "topology": "2x2x2",
        "vms_per_slice": 2,
        "gke_accelerator": "tpu7x",
        "rm_type": "tpu7x:2x2x2",
        "chips_per_vm": 4,
    },
    "7x-32": {
        "topology": "2x2x4",
        "vms_per_slice": 4,
        "gke_accelerator": "tpu7x",
        "rm_type": "tpu7x:2x2x4",
        "chips_per_vm": 4,
    },
    "7x-64": {
        "topology": "2x4x4",
        "vms_per_slice": 8,
        "gke_accelerator": "tpu7x",
        "rm_type": "tpu7x:2x4x4",
        "chips_per_vm": 4,
    },
    "7x-128": {
        "topology": "4x4x4",
        "vms_per_slice": 16,
        "gke_accelerator": "tpu7x",
        "rm_type": "tpu7x:4x4x4",
        "chips_per_vm": 4,
    },
    "7x-256": {
        "topology": "4x4x8",
        "vms_per_slice": 32,
        "gke_accelerator": "tpu7x",
        "rm_type": "tpu7x:4x4x8",
        "chips_per_vm": 4,
    },
    "7x-512": {
        "topology": "4x8x8",
        "vms_per_slice": 64,
        "gke_accelerator": "tpu7x",
        "rm_type": "tpu7x:4x8x8",
        "chips_per_vm": 4,
    },
    "7x-1024": {
        "topology": "8x8x8",
        "vms_per_slice": 128,
        "gke_accelerator": "tpu7x",
        "rm_type": "tpu7x:8x8x8",
        "chips_per_vm": 4,
    },
    "7x-2048": {
        "topology": "8x8x16",
        "vms_per_slice": 256,
        "gke_accelerator": "tpu7x",
        "rm_type": "tpu7x:8x8x16",
        "chips_per_vm": 4,
    },
    "7x-4096": {
        "topology": "8x16x16",
        "vms_per_slice": 512,
        "gke_accelerator": "tpu7x",
        "rm_type": "tpu7x:8x16x16",
        "chips_per_vm": 4,
    },
    "7x-8192": {
        "topology": "16x16x16",
        "vms_per_slice": 1024,
        "gke_accelerator": "tpu7x",
        "rm_type": "tpu7x:16x16x16",
        "chips_per_vm": 4,
    },
    # v4
    "v4-8": {
        "topology": "2x2x1",
        "vms_per_slice": 1,
        "gke_accelerator": "tpu-v4-podslice",
        "rm_type": "tpuv4:2x2x1",
        "chips_per_vm": 4,
    },
    "v4-16": {
        "topology": "2x2x2",
        "vms_per_slice": 2,
        "gke_accelerator": "tpu-v4-podslice",
        "rm_type": "tpuv4:2x2x2",
        "chips_per_vm": 4,
    },
    "v4-32": {
        "topology": "2x2x4",
        "vms_per_slice": 4,
        "gke_accelerator": "tpu-v4-podslice",
        "rm_type": "tpuv4:2x2x4",
        "chips_per_vm": 4,
    },
    "v4-64": {
        "topology": "2x4x4",
        "vms_per_slice": 8,
        "gke_accelerator": "tpu-v4-podslice",
        "rm_type": "tpuv4:2x4x4",
        "chips_per_vm": 4,
    },
    "v4-128": {
        "topology": "4x4x4",
        "vms_per_slice": 16,
        "gke_accelerator": "tpu-v4-podslice",
        "rm_type": "tpuv4:4x4x4",
        "chips_per_vm": 4,
    },
    "v4-256": {
        "topology": "4x4x8",
        "vms_per_slice": 32,
        "gke_accelerator": "tpu-v4-podslice",
        "rm_type": "tpuv4:4x4x8",
        "chips_per_vm": 4,
    },
    "v4-512": {
        "topology": "4x8x8",
        "vms_per_slice": 64,
        "gke_accelerator": "tpu-v4-podslice",
        "rm_type": "tpuv4:4x8x8",
        "chips_per_vm": 4,
    },
    "v4-1024": {
        "topology": "8x8x8",
        "vms_per_slice": 128,
        "gke_accelerator": "tpu-v4-podslice",
        "rm_type": "tpuv4:8x8x8",
        "chips_per_vm": 4,
    },
    "v4-1536": {
        "topology": "8x8x12",
        "vms_per_slice": 192,
        "gke_accelerator": "tpu-v4-podslice",
        "rm_type": "tpuv4:8x8x12",
        "chips_per_vm": 4,
    },
    "v4-2048": {
        "topology": "8x8x16",
        "vms_per_slice": 256,
        "gke_accelerator": "tpu-v4-podslice",
        "rm_type": "tpuv4:8x8x16",
        "chips_per_vm": 4,
    },
    "v4-4096": {
        "topology": "8x16x16",
        "vms_per_slice": 512,
        "gke_accelerator": "tpu-v4-podslice",
        "rm_type": "tpuv4:8x16x16",
        "chips_per_vm": 4,
    },
}

# Dynamically populate v5p topologies
_V5P_DATA = [
    ("v5p-8", "2x2x1", 1),
    ("v5p-16", "2x2x2", 2),
    ("v5p-32", "2x2x4", 4),
    ("v5p-64", "2x4x4", 8),
    ("v5p-128", "4x4x4", 16),
    ("v5p-256", "4x4x8", 32),
    ("v5p-384", "4x4x12", 48),
    ("v5p-512", "4x8x8", 64),
    ("v5p-640", "4x4x20", 80),
    ("v5p-768", "4x8x12", 96),
    ("v5p-896", "4x4x28", 112),
    ("v5p-1024", "8x8x8", 128),
    ("v5p-1152", "4x12x12", 144),
    ("v5p-1280", "4x8x20", 160),
    ("v5p-1408", "4x4x44", 176),
    ("v5p-1536", "8x8x12", 192),
    ("v5p-1664", "4x4x52", 208),
    ("v5p-1792", "4x8x28", 224),
    ("v5p-1920", "4x12x20", 240),
    ("v5p-2048", "8x8x16", 256),
    ("v5p-2176", "4x4x68", 272),
    ("v5p-2304", "8x12x12", 288),
    ("v5p-2432", "4x4x76", 304),
    ("v5p-2560", "8x8x20", 320),
    ("v5p-2688", "4x12x28", 336),
    ("v5p-2816", "4x8x44", 352),
    ("v5p-2944", "4x4x92", 368),
    ("v5p-3072", "8x12x16", 384),
    ("v5p-3200", "4x20x20", 400),
    ("v5p-3328", "4x8x52", 416),
    ("v5p-3456", "12x12x12", 432),
    ("v5p-3584", "8x8x28", 448),
    ("v5p-3712", "4x4x116", 464),
    ("v5p-3840", "8x12x20", 480),
    ("v5p-3968", "4x4x124", 496),
    ("v5p-4096", "8x16x16", 512),
    ("v5p-4224", "4x12x44", 528),
    ("v5p-4352", "4x8x68", 544),
    ("v5p-4480", "4x20x28", 560),
    ("v5p-4608", "12x12x16", 576),
    ("v5p-4736", "4x4x148", 592),
    ("v5p-4864", "4x8x76", 608),
    ("v5p-4992", "4x12x52", 624),
    ("v5p-5120", "8x16x20", 640),
    ("v5p-5248", "4x4x164", 656),
    ("v5p-5376", "8x12x28", 672),
    ("v5p-5504", "4x4x172", 688),
    ("v5p-5632", "8x8x44", 704),
    ("v5p-5760", "12x12x20", 720),
    ("v5p-5888", "4x8x92", 736),
    ("v5p-6016", "4x4x188", 752),
    ("v5p-6144", "12x16x16", 768),
    ("v5p-6272", "4x28x28", 784),
    ("v5p-6400", "8x20x20", 800),
    ("v5p-6528", "4x12x68", 816),
    ("v5p-6656", "8x8x52", 832),
    ("v5p-6784", "4x4x212", 848),
    ("v5p-6912", "12x12x24", 864),
    ("v5p-7040", "4x20x44", 880),
    ("v5p-7168", "8x16x28", 896),
    ("v5p-7296", "4x12x76", 912),
    ("v5p-7424", "4x8x116", 928),
    ("v5p-7552", "4x4x236", 944),
    ("v5p-7680", "12x16x20", 960),
    ("v5p-7808", "4x4x244", 976),
    ("v5p-7936", "4x8x124", 992),
    ("v5p-8064", "12x12x28", 1008),
    ("v5p-8192", "16x16x16", 1024),
    ("v5p-8320", "4x20x52", 1040),
    ("v5p-8448", "8x12x44", 1056),
    ("v5p-8704", "8x8x68", 1088),
    ("v5p-8832", "4x12x92", 1104),
    ("v5p-8960", "8x20x28", 1120),
    ("v5p-9216", "12x16x24", 1152),
    ("v5p-9472", "4x8x148", 1184),
    ("v5p-9600", "12x20x20", 1200),
    ("v5p-9728", "8x8x76", 1216),
    ("v5p-9856", "4x28x44", 1232),
    ("v5p-9984", "8x12x52", 1248),
    ("v5p-10240", "16x16x20", 1280),
    ("v5p-10368", "12x12x36", 1296),
    ("v5p-10496", "4x8x164", 1312),
    ("v5p-10752", "12x16x28", 1344),
    ("v5p-10880", "4x20x68", 1360),
    ("v5p-11008", "4x8x172", 1376),
    ("v5p-11136", "4x12x116", 1392),
    ("v5p-11264", "8x16x44", 1408),
    ("v5p-11520", "12x20x24", 1440),
    ("v5p-11648", "4x28x52", 1456),
    ("v5p-11776", "8x8x92", 1472),
    ("v5p-11904", "4x12x124", 1488),
    ("v5p-12032", "4x8x188", 1504),
    ("v5p-12160", "4x20x76", 1520),
    ("v5p-12288", "16x16x24", 1536),
    ("v5p-13824", "12x24x24", 1728),
    ("v5p-17920", "16x20x28", 2240),
]
for key, topo, vms in _V5P_DATA:
    TPU_MAPPINGS[key] = {
        "topology": topo,
        "vms_per_slice": vms,
        "gke_accelerator": "tpu-v5p-slice",
        "rm_type": f"tpuv5p:{topo}",
        "chips_per_vm": 4,
    }


def get_colocated_python_image(client_image: str) -> str:
    if "/" in client_image and ":" in client_image:
        try:
            path, tag = client_image.rsplit(":", 1)
            repo, _ = path.rsplit("/", 1)
            return f"{repo}/colocated-python:{tag}"
        except Exception:
            pass
    return "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/colocated-python:jax-0.10.0"


def generate_yaml(
    name: str,
    namespace: str,
    tpu_type: str,
    gcs_scratch_location: str,
    num_slices: int = 1,
    jax_client_image: str = "python:3.12-slim",
    command: str = None,
    spot: bool = False,
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
    if spot:
        spot_toleration_head = (
            '                - key: "cloud.google.com/gke-spot"\n'
            '                  operator: "Equal"\n'
            '                  value: "true"\n'
            '                  effect: "NoSchedule"'
        )
        spot_node_selector_worker = '                cloud.google.com/gke-spot: "true"'
        spot_toleration_worker = (
            '                - key: "cloud.google.com/gke-spot"\n'
            '                  operator: "Equal"\n'
            '                  value: "true"\n'
            '                  effect: "NoSchedule"'
        )
    else:
        spot_toleration_head = ""
        spot_node_selector_worker = ""
        spot_toleration_worker = ""

    # Format colocated python options
    if colocated_python:
        proxy_sidecar_arg = "\n                    - --sidecar_name=external"
        tpu_premapped_buffer_size = 34359738368  # 32 GiB
        colocated_img = get_colocated_python_image(jax_client_image)
        worker_init_containers = (
            "              initContainers:\n"
            "                - name: colocated-python\n"
            f"                  image: {colocated_img}\n"
            "                  imagePullPolicy: Always\n"
            "                  restartPolicy: Always\n"
            "                  ports:\n"
            "                    - containerPort: 50051\n"
            "                      protocol: TCP\n"
            "                  env:\n"
            "                    - name: CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY\n"
            "                      value: /tmp/ifrt_proxy\n"
            "                    - name: GRPC_SERVER_ADDRESS\n"
            "                      value: 0.0.0.0:50051\n"
            "                  volumeMounts:\n"
            "                    - name: shared-memory\n"
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
        GKE_ACCELERATOR=mapping["gke_accelerator"],
        CHIPS_PER_VM=mapping["chips_per_vm"],
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
    return "\n".join(lines) + "\n"
