import subprocess

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
