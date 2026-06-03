import os
import time
import subprocess
import pytest
from click.testing import CliRunner
from pwy.cli import main
import dotenv

dotenv.load_dotenv()


@pytest.mark.e2e
def test_e2e_pathways_interactive_up_down():
    runner = CliRunner()
    gcs_location = os.getenv("PWY_E2E_GCS_SCRATCH_LOCATION")
    assert (
        gcs_location is not None
    ), "PWY_E2E_GCS_SCRATCH_LOCATION env variable must be set in .env"

    # 1. Clean up any leftover JobSet first just in case
    print("Pre-test cleanup...")
    runner.invoke(main, ["down"])
    time.sleep(10)

    # 2. Deploy JobSet using pwy up
    print("Launching pathways cluster via pwy up...")
    result = runner.invoke(
        main,
        [
            "up",
            "--tpu-type",
            "v6e-4",
            "--gcs-scratch-location",
            gcs_location,
        ],
    )

    assert result.exit_code == 0, f"pwy up failed: {result.output}"
    assert "Successfully applied JobSet" in result.output

    # 3. Poll pods until they are READY/Running (timeout 180s)
    head_pod_name = None
    timeout = 180
    start_time = time.time()

    print("Waiting for pods to be Running...")
    while time.time() - start_time < timeout:
        # Find head pod name
        cmd = [
            "kubectl",
            "get",
            "pods",
            "-l",
            "jobset.sigs.k8s.io/jobset-name=pathways-interactive",
            "-o",
            'jsonpath={.items[?(@.metadata.labels.jobset\\.sigs\\.k8s\\.io/replicatedjob-name=="pwhd")].metadata.name}',
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0 and proc.stdout.strip():
            head_pod_name = proc.stdout.decode("utf-8").strip()

        # Check pod statuses
        proc_status = subprocess.run(
            [
                "kubectl",
                "get",
                "pods",
                "-l",
                "jobset.sigs.k8s.io/jobset-name=pathways-interactive",
                "-o",
                "jsonpath={.items[*].status.phase}",
            ],
            capture_output=True,
        )

        if proc_status.returncode == 0:
            phases = proc_status.stdout.decode("utf-8").strip().split()
            # If all pods are in Running phase
            if phases and all(p == "Running" for p in phases):
                # Also check container readiness of head pod
                proc_ready = subprocess.run(
                    [
                        "kubectl",
                        "get",
                        "pod",
                        head_pod_name,
                        "-o",
                        "jsonpath={.status.containerStatuses[*].ready}",
                    ],
                    capture_output=True,
                )
                if proc_ready.returncode == 0:
                    readiness = proc_ready.stdout.decode("utf-8").strip().split()
                    if readiness and all(r == "true" for r in readiness):
                        print("All pods are Running and Ready!")
                        break
        time.sleep(5)
    else:
        pytest.fail("Timed out waiting for pods to be Running and Ready.")

    # 4. Install dependencies inside the client container
    print("Installing JAX inside the client container...")
    pip_proc = subprocess.run(
        [
            "kubectl",
            "exec",
            head_pod_name,
            "-c",
            "client",
            "--",
            "pip",
            "install",
            "jax",
            "pathwaysutils",
        ],
        capture_output=True,
    )

    assert (
        pip_proc.returncode == 0
    ), f"pip install failed: {pip_proc.stderr.decode('utf-8')}"

    # 5. Execute python JAX validation snippet
    print("Running JAX verification snippet...")
    python_cmd = "import pathwaysutils; pathwaysutils.initialize(); import jax; print(jax.devices())"
    jax_proc = subprocess.run(
        [
            "kubectl",
            "exec",
            head_pod_name,
            "-c",
            "client",
            "--",
            "python3",
            "-c",
            python_cmd,
        ],
        capture_output=True,
    )

    output = jax_proc.stdout.decode("utf-8")
    stderr = jax_proc.stderr.decode("utf-8")
    print(f"JAX snippet stdout:\n{output}")
    print(f"JAX snippet stderr:\n{stderr}")

    assert jax_proc.returncode == 0, f"JAX execution failed: {stderr}"
    assert "TPU_DEVICE" in output, "TPU devices not found in output"
    assert "device(0,TPU_DEVICE" in output, "Device 0 not found"

    # 5b. Test pwy run with sync and command execution
    print("Testing pwy run execution...")
    # Write a temporary script to the local workspace
    temp_script_path = "tests/test_e2e_run_snippet.py"
    with open(temp_script_path, "w") as f:
        f.write('print("SUCCESSFUL_E2E_RUN_COMMAND")\n')

    try:
        # Execute via pwy run in a subprocess
        pwy_run_proc = subprocess.run(
            [
                "uv",
                "run",
                "pwy",
                "run",
                "python3",
                temp_script_path,
            ],
            capture_output=True,
            text=True,
        )
        print(f"pwy run stdout:\n{pwy_run_proc.stdout}")
        print(f"pwy run stderr:\n{pwy_run_proc.stderr}")
        assert pwy_run_proc.returncode == 0, f"pwy run failed: {pwy_run_proc.stderr}"
        assert "SUCCESSFUL_E2E_RUN_COMMAND" in pwy_run_proc.stdout
    finally:
        # Clean up local temporary file
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

    # 6. Teardown JobSet using pwy down
    print("Cleaning up cluster via pwy down...")
    down_result = runner.invoke(main, ["down"])
    assert down_result.exit_code == 0, f"pwy down failed: {down_result.output}"
    assert "Successfully deleted JobSet" in down_result.output
