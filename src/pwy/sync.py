import os
import sys
import tempfile
import stat
import subprocess


def install_rsync_if_needed(pod_name: str, namespace: str) -> bool:
    """Checks if rsync is available in the remote container, and attempts to install it if not."""
    # Check if rsync exists
    check_cmd = [
        "kubectl",
        "exec",
        "-n",
        namespace,
        pod_name,
        "-c",
        "client",
        "--",
        "which",
        "rsync",
    ]
    process = subprocess.run(check_cmd, capture_output=True)
    if process.returncode == 0:
        return True

    # Try installing rsync
    print("rsync not found in client container. Attempting to install...")
    install_cmd = [
        "kubectl",
        "exec",
        "-n",
        namespace,
        pod_name,
        "-c",
        "client",
        "--",
        "bash",
        "-c",
        "apt-get update && apt-get install -y rsync",
    ]
    process = subprocess.run(install_cmd, capture_output=True, text=True)
    if process.returncode == 0:
        print("Successfully installed rsync in the client container.")
        return True
    else:
        print(
            f"Warning: Failed to install rsync dynamically (likely because the container image lacks apt or isn't root): {process.stderr.strip()}"
        )
        return False


def run_rsync_sync(source: str, dest: str, pod_name: str, namespace: str) -> bool:
    """Performs an incremental sync using rsync over kubectl."""
    src_dir = os.path.abspath(source)
    if os.path.isdir(src_dir) and not src_dir.endswith("/"):
        src_dir += "/"

    # Create a temporary shell script wrapper for kubectl exec to correctly position '--'
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(f"""#!/bin/bash
pod_name="$1"
shift
exec kubectl exec -i -n {namespace} -c client "$pod_name" -- "$@"
""")
        helper_path = f.name

    st = os.stat(helper_path)
    os.chmod(helper_path, st.st_mode | stat.S_IEXEC)

    try:
        cmd = [
            "rsync",
            "-avz",
            "--blocking-io",
            "-e",
            helper_path,
            "--exclude",
            ".git",
            "--exclude",
            ".venv",
            "--exclude",
            "__pycache__",
            "--exclude",
            ".pytest_cache",
            src_dir,
            f"{pod_name}:{dest}",
        ]
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode == 0:
            print("Sync complete.")
            return True
        else:
            print(f"rsync failed: {process.stderr.strip()}", file=sys.stderr)
            return False
    finally:
        try:
            os.remove(helper_path)
        except OSError:
            pass


def run_fallback_sync(source: str, dest: str, pod_name: str, namespace: str) -> bool:
    """Performs sync using kubectl cp as a fallback."""
    print("Falling back to kubectl cp...")
    # First, make sure the destination folder exists on the pod
    mkdir_cmd = [
        "kubectl",
        "exec",
        "-n",
        namespace,
        pod_name,
        "-c",
        "client",
        "--",
        "mkdir",
        "-p",
        dest,
    ]
    subprocess.run(mkdir_cmd, capture_output=True)

    # Use kubectl cp to copy the source files
    cp_cmd = [
        "kubectl",
        "cp",
        source,
        f"{namespace}/{pod_name}:{dest}",
        "-c",
        "client",
    ]
    process = subprocess.run(cp_cmd, capture_output=True, text=True)
    if process.returncode == 0:
        print("Sync complete (via kubectl cp).")
        return True
    else:
        print(f"kubectl cp failed: {process.stderr.strip()}", file=sys.stderr)
        return False


def sync_directory(source: str, dest: str, pod_name: str, namespace: str):
    """Orchestrates syncing directories, using rsync if available, otherwise falling back to kubectl cp."""
    if install_rsync_if_needed(pod_name, namespace):
        success = run_rsync_sync(source, dest, pod_name, namespace)
        if not success:
            run_fallback_sync(source, dest, pod_name, namespace)
    else:
        run_fallback_sync(source, dest, pod_name, namespace)
