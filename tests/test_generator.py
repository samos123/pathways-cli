import pytest
from pwy.generator import (
    generate_yaml,
    generate_jobset_dict,
    get_colocated_python_image,
)


def test_get_colocated_python_image():
    # Registry path
    assert (
        get_colocated_python_image("us-docker.pkg.dev/my-project/my-repo/client:latest")
        == "us-docker.pkg.dev/my-project/my-repo/colocated-python:latest"
    )
    assert (
        get_colocated_python_image("gcr.io/another-project/image:v1.0")
        == "gcr.io/another-project/colocated-python:v1.0"
    )
    # Fallback/invalid path
    assert (
        get_colocated_python_image("python:3.12-slim")
        == "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/colocated-python:jax-0.10.0"
    )


def test_generate_yaml_default():
    yaml_content = generate_yaml(
        name="test-run",
        namespace="my-namespace",
        tpu_type="v6e-4",
        gcs_scratch_location="gs://my-bucket/staging",
    )

    # Assert name and namespace
    assert "name: test-run" in yaml_content
    assert "namespace: my-namespace" in yaml_content

    # Assert TPU Type and mappings
    assert "value: v6e-4" in yaml_content
    assert "cloud.google.com/gke-tpu-topology: 2x2" in yaml_content
    assert "parallelism: 1" in yaml_content
    assert "completions: 1" in yaml_content
    assert "--instance_type=tpuv6e:2x2" in yaml_content

    # Default commands
    assert "sleep infinity" in yaml_content

    # Premapped buffer size (default 4 GiB)
    assert "--tpu_premapped_buffer_size=4294967296" in yaml_content

    # Spot elements should NOT be present
    assert "cloud.google.com/gke-spot" not in yaml_content

    # Colocated python elements should NOT be present
    assert "colocated-python" not in yaml_content
    assert "--sidecar_name=external" not in yaml_content

    # Head on TPU elements should be present by default
    assert "affinity:" in yaml_content
    assert "podAffinity:" in yaml_content
    assert "jobset.sigs.k8s.io/jobset-name" in yaml_content
    assert "- test-run" in yaml_content
    assert "topologyKey: kubernetes.io/hostname" in yaml_content
    assert "exclusive-topology" not in yaml_content


def test_generate_yaml_head_on_tpu_disabled():
    yaml_content = generate_yaml(
        name="test-run",
        namespace="my-namespace",
        tpu_type="v6e-4",
        gcs_scratch_location="gs://my-bucket/staging",
        head_on_tpu=False,
    )
    # Head on TPU elements should NOT be present
    assert "affinity:" not in yaml_content
    assert "podAffinity:" not in yaml_content
    assert "jobset.sigs.k8s.io/jobset-name" not in yaml_content

    pwhd_part, _ = yaml_content.split("- name: pwwk", 1)
    assert "google.com/tpu" not in pwhd_part
    assert "exclusive-topology" in yaml_content


def test_generate_yaml_v6e_16_multi_slice():
    yaml_content = generate_yaml(
        name="test-run",
        namespace="default",
        tpu_type="v6e-16",
        gcs_scratch_location="gs://my-bucket/staging",
        num_slices=2,
    )
    # VMs per slice = 4, rm_type = tpuv6e:4x4, num_slices = 2
    assert "cloud.google.com/gke-tpu-topology: 4x4" in yaml_content
    assert "replicas: 2" in yaml_content
    assert "parallelism: 4" in yaml_content
    assert "completions: 4" in yaml_content
    assert "--instance_count=2" in yaml_content
    assert "--instance_type=tpuv6e:4x4" in yaml_content


def test_generate_yaml_spot_enabled():
    yaml_content = generate_yaml(
        name="test-run",
        namespace="default",
        tpu_type="v6e-4",
        gcs_scratch_location="gs://my-bucket/staging",
        spot=True,
    )

    # Assert spot VM node selector and tolerations are present
    assert 'cloud.google.com/gke-spot: "true"' in yaml_content
    assert "key: cloud.google.com/gke-spot" in yaml_content
    assert "operator: Equal" in yaml_content
    assert 'value: "true"' in yaml_content
    assert "effect: NoSchedule" in yaml_content


def test_generate_yaml_colocated_python():
    client_img = "us-docker.pkg.dev/my-project/my-repo/client:latest"
    yaml_content = generate_yaml(
        name="test-run",
        namespace="default",
        tpu_type="v6e-4",
        gcs_scratch_location="gs://my-bucket/staging",
        jax_client_image=client_img,
        colocated_python=True,
    )

    # Colocated python image derived from client image
    expected_colocated_img = (
        "us-docker.pkg.dev/my-project/my-repo/colocated-python:latest"
    )
    assert f"image: {expected_colocated_img}" in yaml_content
    assert "name: colocated-python" in yaml_content
    assert "- --sidecar_name=external" in yaml_content
    assert "mountPath: /tmp/ifrt_proxy" in yaml_content
    assert "emptyDir:" in yaml_content

    # Premapped buffer size should be 4 GiB
    assert "--tpu_premapped_buffer_size=4294967296" in yaml_content


def test_generate_yaml_custom_command():
    yaml_content = generate_yaml(
        name="test-run",
        namespace="default",
        tpu_type="v6e-4",
        gcs_scratch_location="gs://my-bucket/staging",
        command="python run_my_training.py --batch-size=32",
    )
    assert "python run_my_training.py --batch-size=32" in yaml_content
    assert "sleep infinity" not in yaml_content


def test_generate_yaml_invalid_tpu_type():
    with pytest.raises(ValueError) as excinfo:
        generate_yaml(
            name="test-run",
            namespace="default",
            tpu_type="invalid-tpu",  # not in the mappings
            gcs_scratch_location="gs://my-bucket/staging",
        )
    assert "Unsupported TPU type: invalid-tpu" in str(excinfo.value)


def test_generate_yaml_v5p_8():
    yaml_content = generate_yaml(
        name="test-run",
        namespace="default",
        tpu_type="v5p-8",
        gcs_scratch_location="gs://my-bucket/staging",
    )
    assert "cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice" in yaml_content
    assert "cloud.google.com/gke-tpu-topology: 2x2x1" in yaml_content
    assert "--instance_type=tpuv5:2x2x1" in yaml_content
    assert "google.com/tpu: 4" in yaml_content


def test_generate_yaml_v6e_8_1_eight_chips():
    yaml_content = generate_yaml(
        name="test-run",
        namespace="default",
        tpu_type="v6e-8-1",
        gcs_scratch_location="gs://my-bucket/staging",
    )
    assert "cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice" in yaml_content
    assert "cloud.google.com/gke-tpu-topology: 2x4" in yaml_content
    assert "--instance_type=tpuv6e:2x4" in yaml_content
    assert "google.com/tpu: 8" in yaml_content


def test_generate_yaml_7x_8():
    yaml_content = generate_yaml(
        name="test-run",
        namespace="default",
        tpu_type="7x-8",
        gcs_scratch_location="gs://my-bucket/staging",
    )
    assert "cloud.google.com/gke-tpu-accelerator: tpu7x" in yaml_content
    assert "cloud.google.com/gke-tpu-topology: 2x2x1" in yaml_content
    assert "--instance_type=tpu7x:2x2x1" in yaml_content


def test_generate_yaml_sync_enabled_default_path():
    yaml_content = generate_yaml(
        name="test-run",
        namespace="default",
        tpu_type="v6e-4",
        gcs_scratch_location="gs://my-bucket/staging",
        sync=True,
    )
    assert "mkdir -p /app && cd /app && sleep infinity" in yaml_content


def test_generate_yaml_sync_enabled_custom_path():
    yaml_content = generate_yaml(
        name="test-run",
        namespace="default",
        tpu_type="v6e-4",
        gcs_scratch_location="gs://my-bucket/staging",
        sync=True,
        remote_path="/workspace",
        command="python run_my_training.py",
    )
    assert (
        "mkdir -p /workspace && cd /workspace && python run_my_training.py"
        in yaml_content
    )


def test_generate_yaml_invalid_names():
    # Empty name
    with pytest.raises(ValueError) as excinfo:
        generate_yaml(
            name="",
            namespace="default",
            tpu_type="v6e-4",
            gcs_scratch_location="gs://my-bucket/staging",
        )
    assert "Name cannot be empty" in str(excinfo.value)

    # Name too long (> 63 characters)
    long_name = "a" * 64
    with pytest.raises(ValueError) as excinfo:
        generate_yaml(
            name=long_name,
            namespace="default",
            tpu_type="v6e-4",
            gcs_scratch_location="gs://my-bucket/staging",
        )
    assert "Name must be 63 characters or less" in str(excinfo.value)

    # Name containing invalid characters (uppercase)
    with pytest.raises(ValueError) as excinfo:
        generate_yaml(
            name="MyCluster",
            namespace="default",
            tpu_type="v6e-4",
            gcs_scratch_location="gs://my-bucket/staging",
        )
    assert "Name must consist of lowercase alphanumeric" in str(excinfo.value)


def test_generate_jobset_dict_structure():
    jobset_dict = generate_jobset_dict(
        name="dict-test",
        namespace="test-ns",
        tpu_type="v6e-4",
        gcs_scratch_location="gs://my-bucket/staging",
        spot=True,
    )
    assert isinstance(jobset_dict, dict)
    assert jobset_dict["apiVersion"] == "jobset.x-k8s.io/v1alpha2"
    assert jobset_dict["kind"] == "JobSet"
    assert jobset_dict["metadata"]["name"] == "dict-test"
    assert jobset_dict["metadata"]["namespace"] == "test-ns"
    replicated_jobs = jobset_dict["spec"]["replicatedJobs"]
    assert len(replicated_jobs) == 2
    assert replicated_jobs[0]["name"] == "pwhd"
    assert replicated_jobs[1]["name"] == "pwwk"
