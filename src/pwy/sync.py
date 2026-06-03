import os
import sys
import time
import tempfile
import stat
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


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


class SyncEventHandler(FileSystemEventHandler):
    def __init__(self, source: str, dest: str, pod_name: str, namespace: str):
        super().__init__()
        self.source = source
        self.dest = dest
        self.pod_name = pod_name
        self.namespace = namespace
        self.last_sync = 0
        self.debounce_interval = 1.0  # seconds

    def on_any_event(self, event):
        if event.is_directory:
            return

        # Exclude common dev build/cache folders
        parts = event.src_path.split(os.sep)
        if any(
            p in parts
            for p in (".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache")
        ):
            return

        current_time = time.time()
        if current_time - self.last_sync > self.debounce_interval:
            self.last_sync = current_time
            print(f"Change detected: {event.src_path}. Syncing...")
            sync_directory(self.source, self.dest, self.pod_name, self.namespace)


def watch_directory(source: str, dest: str, pod_name: str, namespace: str):
    """Starts a continuous file watching loop to sync files on change."""
    print(f"Starting file watcher on '{source}' syncing to '{pod_name}:{dest}'...")
    # Initial sync
    sync_directory(source, dest, pod_name, namespace)

    event_handler = SyncEventHandler(source, dest, pod_name, namespace)
    observer = Observer()
    observer.schedule(event_handler, path=source, recursive=True)
    observer.start()

    print("Watcher started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        observer.stop()
    observer.join()
