import subprocess
import time


def apply_manifest(yaml_content: str) -> subprocess.CompletedProcess:
    """Applies the YAML manifest using kubectl apply -f -."""
    process = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=yaml_content.encode("utf-8"),
        capture_output=True,
    )
    return process


def delete_jobset(name: str, namespace: str) -> subprocess.CompletedProcess:
    """Deletes the JobSet using kubectl delete jobset <name> --namespace=<namespace>."""
    process = subprocess.run(
        ["kubectl", "delete", "jobset", name, f"--namespace={namespace}"],
        capture_output=True,
    )
    return process


def get_client_pod_name(name: str, namespace: str) -> str:
    """Finds the name of the JAX client (head) pod in the JobSet."""
    cmd = [
        "kubectl",
        "get",
        "pods",
        "--namespace",
        namespace,
        "-l",
        f"jobset.sigs.k8s.io/jobset-name={name},jobset.sigs.k8s.io/replicatedjob-name=pwhd",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ]
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0 or not process.stdout.strip():
        raise RuntimeError(
            f"Failed to find client pod for JobSet '{name}' in namespace '{namespace}': {process.stderr.strip()}"
        )
    return process.stdout.strip()


def wait_for_client_pod(name: str, namespace: str, timeout: int = 60) -> str:
    """Waits for the client pod to be created and returns its name."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            return get_client_pod_name(name, namespace)
        except RuntimeError:
            pass
        time.sleep(2)
    raise RuntimeError(
        f"Timed out waiting for client pod to be created for JobSet '{name}'"
    )


def wait_for_pod_ready(pod_name: str, namespace: str, timeout: int = 180) -> bool:
    """Polls a pod until all of its containers are Running and Ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        phase_cmd = [
            "kubectl",
            "get",
            "pod",
            pod_name,
            "--namespace",
            namespace,
            "-o",
            "jsonpath={.status.phase}",
        ]
        proc_phase = subprocess.run(phase_cmd, capture_output=True, text=True)
        if proc_phase.returncode == 0 and proc_phase.stdout.strip() == "Running":
            ready_cmd = [
                "kubectl",
                "get",
                "pod",
                pod_name,
                "--namespace",
                namespace,
                "-o",
                "jsonpath={.status.containerStatuses[*].ready}",
            ]
            proc_ready = subprocess.run(ready_cmd, capture_output=True, text=True)
            if proc_ready.returncode == 0:
                readiness = proc_ready.stdout.strip().split()
                if readiness and all(r == "true" for r in readiness):
                    return True
        time.sleep(3)
    return False
