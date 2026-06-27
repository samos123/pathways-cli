"""
JobSet manifest generator for Pathways GKE clusters.
Constructs the JobSet Custom Resource specification as a structured Python dictionary.
"""


def build_jobset_dict(
    name: str,
    namespace: str,
    tpu_type: str,
    gcs_scratch_location: str,
    gke_topology: str,
    vms_per_slice: int,
    rm_instance_type: str,
    gke_accelerator: str,
    chips_per_vm: int,
    num_slices: int = 1,
    jax_client_image: str = "python:3.12-slim",
    command: str = None,
    spot: bool = False,
    colocated_python: bool = False,
    head_on_tpu: bool = True,
    sync: bool = False,
    remote_path: str = "/app",
    colocated_img: str = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/colocated-python:jax-0.10.0",
) -> dict:
    """Builds the complete Python dictionary for the JobSet resource."""
    # Format client execution command
    if not command:
        client_command = "sleep infinity"
    else:
        client_command = command

    if sync:
        client_command = (
            f"mkdir -p {remote_path} && cd {remote_path} && {client_command}"
        )

    tpu_premapped_buffer_size = 4294967296

    # -------------------------------------------------------------------------
    # 1. Pathways Head (Client Pod)
    # -------------------------------------------------------------------------
    head_tolerations = []
    if head_on_tpu:
        head_tolerations.append(
            {
                "key": "google.com/tpu",
                "operator": "Equal",
                "value": "present",
                "effect": "NoSchedule",
            }
        )
    if spot:
        head_tolerations.append(
            {
                "key": "cloud.google.com/gke-spot",
                "operator": "Equal",
                "value": "true",
                "effect": "NoSchedule",
            }
        )

    head_pod_spec = {
        "terminationGracePeriodSeconds": 60,
        "restartPolicy": "Never",
        "hostAliases": [
            {
                "ip": "169.254.169.254",
                "hostnames": ["metadata", "metadata.google.internal"],
            }
        ],
        "serviceAccountName": "default",
        "dnsPolicy": "ClusterFirstWithHostNet",
    }

    if head_tolerations:
        head_pod_spec["tolerations"] = head_tolerations

    if head_on_tpu:
        head_pod_spec["affinity"] = {
            "podAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {
                            "matchExpressions": [
                                {
                                    "key": "jobset.sigs.k8s.io/jobset-name",
                                    "operator": "In",
                                    "values": [name],
                                },
                                {
                                    "key": "jobset.sigs.k8s.io/replicatedjob-name",
                                    "operator": "In",
                                    "values": ["pwwk"],
                                },
                            ]
                        },
                        "topologyKey": "kubernetes.io/hostname",
                    }
                ]
            }
        }

    client_container = {
        "name": "client",
        "image": jax_client_image,
        "command": ["bash", "-c", client_command],
        "resources": {
            "requests": {"cpu": "1000m", "memory": "16Gi"},
            "limits": {"cpu": "1000m", "memory": "16Gi"},
        },
        "env": [
            {"name": "TPU_TYPE", "value": tpu_type},
            {
                "name": "NUM_TPU_SLICES",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']"
                    }
                },
            },
            {"name": "JAX_BACKEND_TARGET", "value": "grpc://localhost:29000"},
            {"name": "XCLOUD_ENVIRONMENT", "value": "GCP"},
            {"name": "JAX_PLATFORMS", "value": "proxy"},
            {"name": "ENABLE_PATHWAYS_PERSISTENCE", "value": "1"},
            {"name": "TPU_SKIP_MDS_QUERY", "value": "true"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "TEST_UNDECLARED_OUTPUTS_DIR", "value": "true"},
            {"name": "IFRT_PROXY_LARGE_TRANSFER_THRESHOLD", "value": "1"},
            {
                "name": "IFRT_PROXY_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY",
                "value": "/tmp/ifrt_proxy",
            },
        ],
        "volumeMounts": [{"name": "shared-memory", "mountPath": "/tmp/ifrt_proxy"}],
        "imagePullPolicy": "Always",
    }
    head_pod_spec["containers"] = [client_container]

    proxy_args = [
        "--resource_manager_address=localhost:29001",
        "--server_port=29000",
    ]
    if colocated_python:
        proxy_args.append("--sidecar_name=external")

    proxy_container = {
        "name": "pathways-proxy",
        "image": "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:jax-0.10.0",
        "restartPolicy": "Always",
        "ports": [{"containerPort": 29000}],
        "env": [
            {"name": "IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS", "value": "true"},
            {
                "name": "IFRT_PROXY_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY",
                "value": "/tmp/ifrt_proxy",
            },
        ],
        "args": proxy_args,
        "volumeMounts": [{"name": "shared-memory", "mountPath": "/tmp/ifrt_proxy"}],
    }

    rm_container = {
        "name": "pathways-rm",
        "image": "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.10.0",
        "restartPolicy": "Always",
        "env": [{"name": "TPU_SKIP_MDS_QUERY", "value": "true"}],
        "args": [
            "--server_port=29001",
            "--node_type=resource_manager",
            f"--instance_count={num_slices}",
            f"--instance_type={rm_instance_type}",
            f"--gcs_scratch_location={gcs_scratch_location}",
            "--enforce_kernel_ipv6_support=false",
        ],
    }
    head_pod_spec["initContainers"] = [proxy_container, rm_container]
    head_pod_spec["volumes"] = [
        {"name": "shared-memory", "emptyDir": {"medium": "Memory"}}
    ]

    pwhd_job = {
        "name": "pwhd",
        "replicas": 1,
        "template": {
            "spec": {
                "parallelism": 1,
                "completions": 1,
                "backoffLimit": 32,
                "template": {
                    "metadata": {
                        "annotations": {
                            "cluster-autoscaler.kubernetes.io/safe-to-evict": "false"
                        }
                    },
                    "spec": head_pod_spec,
                },
            }
        },
    }

    # -------------------------------------------------------------------------
    # 2. Pathways Workers (TPU Pods)
    # -------------------------------------------------------------------------
    worker_node_selector = {
        "cloud.google.com/gke-tpu-accelerator": gke_accelerator,
        "cloud.google.com/gke-tpu-topology": gke_topology,
    }
    worker_tolerations = [
        {
            "key": "google.com/tpu",
            "operator": "Equal",
            "value": "present",
            "effect": "NoSchedule",
        }
    ]
    if spot:
        worker_node_selector["cloud.google.com/gke-spot"] = "true"
        worker_tolerations.append(
            {
                "key": "cloud.google.com/gke-spot",
                "operator": "Equal",
                "value": "true",
                "effect": "NoSchedule",
            }
        )

    worker_container = {
        "name": "worker",
        "image": "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.10.0",
        "imagePullPolicy": "Always",
        "ports": [
            {"containerPort": 8471},
            {"containerPort": 8080},
            {"containerPort": 8431},
            {"containerPort": 9000},
            {"containerPort": 29001},
        ],
        "securityContext": {"privileged": True},
        "resources": {"limits": {"google.com/tpu": chips_per_vm}},
        "env": [
            {"name": "TPU_TYPE", "value": tpu_type},
            {
                "name": "NUM_TPU_SLICES",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']"
                    }
                },
            },
            {
                "name": "MEGASCALE_COORDINATOR_ADDRESS",
                "value": f"{name}-pwhd-0-0.{name}",
            },
            {
                "name": "MEGASCALE_NUM_SLICES",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']"
                    }
                },
            },
            {
                "name": "MEGASCALE_SLICE_ID",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.labels['jobset.sigs.k8s.io/job-index']"
                    }
                },
            },
        ],
        "args": [
            "--server_port=29001",
            f"--resource_manager_address={name}-pwhd-0-0.{name}:29001",
            f"--gcs_scratch_location={gcs_scratch_location}",
            "--tpu_pinned_host_allocation_recycle=true",
            f"--tpu_premapped_buffer_size={tpu_premapped_buffer_size}",
            "--enforce_kernel_ipv6_support=false",
        ],
    }

    worker_pod_spec = {
        "terminationGracePeriodSeconds": 60,
        "hostAliases": [
            {
                "ip": "169.254.169.254",
                "hostnames": ["metadata", "metadata.google.internal"],
            }
        ],
        "nodeSelector": worker_node_selector,
        "tolerations": worker_tolerations,
        "containers": [worker_container],
        "serviceAccountName": "default",
        "dnsPolicy": "ClusterFirstWithHostNet",
    }

    if colocated_python:
        colocated_container = {
            "name": "colocated-python",
            "image": colocated_img,
            "imagePullPolicy": "Always",
            "restartPolicy": "Always",
            "ports": [{"containerPort": 50051, "protocol": "TCP"}],
            "env": [
                {
                    "name": "CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY",
                    "value": "/tmp/ifrt_proxy",
                },
                {"name": "GRPC_SERVER_ADDRESS", "value": "0.0.0.0:50051"},
            ],
            "volumeMounts": [{"name": "shared-memory", "mountPath": "/tmp/ifrt_proxy"}],
        }
        worker_pod_spec["initContainers"] = [colocated_container]
        worker_container["volumeMounts"] = [
            {"name": "shared-memory", "mountPath": "/tmp/ifrt_proxy"}
        ]
        worker_pod_spec["volumes"] = [
            {"name": "shared-memory", "emptyDir": {"medium": "Memory"}}
        ]

    worker_job_template = {"spec": worker_pod_spec}
    if not head_on_tpu:
        worker_job_template["metadata"] = {
            "annotations": {
                "alpha.jobset.sigs.k8s.io/exclusive-topology": "cloud.google.com/gke-nodepool"
            }
        }

    pwwk_job = {
        "name": "pwwk",
        "replicas": num_slices,
        "template": {
            "spec": {
                "parallelism": vms_per_slice,
                "completions": vms_per_slice,
                "backoffLimit": 32,
                "template": worker_job_template,
            }
        },
    }

    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "failurePolicy": {
                "maxRestarts": 0,
                "restartStrategy": "BlockingRecreate",
            },
            "replicatedJobs": [pwhd_job, pwwk_job],
            "successPolicy": {
                "operator": "All",
                "targetReplicatedJobs": ["pwhd"],
            },
        },
    }
