import subprocess
from unittest.mock import patch
from click.testing import CliRunner
from pwy.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "pwy: Standalone Pathways GKE Cluster CLI Tool" in result.output
    assert "up" in result.output
    assert "down" in result.output


def test_cli_up_dry_run():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "up",
            "--tpu-type",
            "v6e-4",
            "--gcs-scratch-location",
            "gs://my-bucket/staging",
            "--dry-run",
            "--name",
            "my-test-cluster",
        ],
    )
    assert result.exit_code == 0
    assert "apiVersion: jobset.x-k8s.io/v1alpha2" in result.output
    assert "name: my-test-cluster" in result.output
    assert "cloud.google.com/gke-tpu-topology: 2x2" in result.output


def test_cli_up_validation_error():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "up",
            "--tpu-type",
            "invalid-tpu",
            "--gcs-scratch-location",
            "gs://my-bucket/staging",
        ],
    )
    assert result.exit_code == 1
    assert "Error: Unsupported TPU type: invalid-tpu" in result.output


@patch("pwy.cli.apply_manifest")
def test_cli_up_apply_success(mock_apply):
    # Mocking subprocess completed process with success
    mock_apply.return_value = subprocess.CompletedProcess(
        args=["kubectl", "apply"],
        returncode=0,
        stdout=b"jobset.jobset.x-k8s.io/pathways-interactive created",
        stderr=b"",
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "up",
            "--tpu-type",
            "v6e-4",
            "--gcs-scratch-location",
            "gs://my-bucket/staging",
        ],
    )

    assert result.exit_code == 0
    assert (
        "Applying Pathways JobSet manifest for 'pathways-interactive'" in result.output
    )
    assert "Successfully applied JobSet 'pathways-interactive'!" in result.output
    mock_apply.assert_called_once()


@patch("pwy.cli.apply_manifest")
def test_cli_up_apply_failure(mock_apply):
    # Mocking subprocess completed process with failure
    mock_apply.return_value = subprocess.CompletedProcess(
        args=["kubectl", "apply"],
        returncode=1,
        stdout=b"",
        stderr=b"Error from server (Forbidden): user cannot apply jobsets",
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "up",
            "--tpu-type",
            "v6e-4",
            "--gcs-scratch-location",
            "gs://my-bucket/staging",
        ],
    )

    assert result.exit_code == 1
    assert "Failed to apply JobSet manifest." in result.output
    assert "Error from server (Forbidden)" in result.output
    mock_apply.assert_called_once()


@patch("pwy.cli.delete_jobset")
def test_cli_down_success(mock_delete):
    mock_delete.return_value = subprocess.CompletedProcess(
        args=["kubectl", "delete"],
        returncode=0,
        stdout=b'jobset.jobset.x-k8s.io "pathways-interactive" deleted',
        stderr=b"",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["down"])

    assert result.exit_code == 0
    assert "Deleting Pathways JobSet 'pathways-interactive'" in result.output
    assert "Successfully deleted JobSet 'pathways-interactive'!" in result.output
    mock_delete.assert_called_once()


def test_cli_up_with_env_variables(monkeypatch):
    monkeypatch.setenv("PWY_TPU_TYPE", "v6e-8")
    monkeypatch.setenv("PWY_GCS_SCRATCH_LOCATION", "gs://my-env-bucket/staging")
    monkeypatch.setenv("PWY_NAME", "env-cluster")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "up",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "apiVersion: jobset.x-k8s.io/v1alpha2" in result.output
    assert "name: env-cluster" in result.output
    assert "cloud.google.com/gke-tpu-topology: 2x4" in result.output


def test_cli_up_with_dotenv_file():
    import subprocess
    import os

    dotenv_content = (
        "PWY_TPU_TYPE=v6e-8\n"
        "PWY_GCS_SCRATCH_LOCATION=gs://test-dotenv-bucket/staging\n"
        "PWY_NAME=test-dotenv-cluster\n"
    )
    # To be safe, we can create the temporary directory inside the workspace itself.
    # Let's create a scratch dir inside the workspace.
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scratch_dir = os.path.join(workspace_root, ".test_scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    try:
        with open(os.path.join(scratch_dir, ".env"), "w") as f:
            f.write(dotenv_content)

        result = subprocess.run(
            ["uv", "run", "pwy", "up", "--dry-run"],
            cwd=scratch_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "name: test-dotenv-cluster" in result.stdout
        assert "cloud.google.com/gke-tpu-topology: 2x4" in result.stdout
    finally:
        # Cleanup
        if os.path.exists(os.path.join(scratch_dir, ".env")):
            os.remove(os.path.join(scratch_dir, ".env"))
        if os.path.exists(scratch_dir):
            os.rmdir(scratch_dir)


def test_cli_up_spot_dry_run():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "up",
            "--tpu-type",
            "v6e-4",
            "--gcs-scratch-location",
            "gs://my-bucket/staging",
            "--dry-run",
            "--spot",
        ],
    )
    assert result.exit_code == 0
    assert 'cloud.google.com/gke-spot: "true"' in result.output
    assert '- key: "cloud.google.com/gke-spot"' in result.output
