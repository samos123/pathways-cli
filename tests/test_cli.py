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
    result = runner.invoke(main, [
        "up",
        "--tpu-type", "v6e-4",
        "--gcs-scratch-location", "gs://my-bucket/staging",
        "--dry-run",
        "--name", "my-test-cluster",
    ])
    assert result.exit_code == 0
    assert "apiVersion: jobset.x-k8s.io/v1alpha2" in result.output
    assert "name: my-test-cluster" in result.output
    assert "cloud.google.com/gke-tpu-topology: 2x2" in result.output

def test_cli_up_validation_error():
    runner = CliRunner()
    result = runner.invoke(main, [
        "up",
        "--tpu-type", "invalid-tpu",
        "--gcs-scratch-location", "gs://my-bucket/staging",
    ])
    assert result.exit_code == 1
    assert "Error: Unsupported TPU type: invalid-tpu" in result.output

@patch("pwy.cli.apply_manifest")
def test_cli_up_apply_success(mock_apply):
    # Mocking subprocess completed process with success
    mock_apply.return_value = subprocess.CompletedProcess(
        args=["kubectl", "apply"],
        returncode=0,
        stdout=b"jobset.jobset.x-k8s.io/pathways-interactive created",
        stderr=b""
    )
    
    runner = CliRunner()
    result = runner.invoke(main, [
        "up",
        "--tpu-type", "v6e-4",
        "--gcs-scratch-location", "gs://my-bucket/staging",
    ])
    
    assert result.exit_code == 0
    assert "Applying Pathways JobSet manifest for 'pathways-interactive'" in result.output
    assert "Successfully applied JobSet 'pathways-interactive'!" in result.output
    mock_apply.assert_called_once()

@patch("pwy.cli.apply_manifest")
def test_cli_up_apply_failure(mock_apply):
    # Mocking subprocess completed process with failure
    mock_apply.return_value = subprocess.CompletedProcess(
        args=["kubectl", "apply"],
        returncode=1,
        stdout=b"",
        stderr=b"Error from server (Forbidden): user cannot apply jobsets"
    )
    
    runner = CliRunner()
    result = runner.invoke(main, [
        "up",
        "--tpu-type", "v6e-4",
        "--gcs-scratch-location", "gs://my-bucket/staging",
    ])
    
    assert result.exit_code == 1
    assert "Failed to apply JobSet manifest." in result.output
    assert "Error from server (Forbidden)" in result.output
    mock_apply.assert_called_once()

@patch("pwy.cli.delete_jobset")
def test_cli_down_success(mock_delete):
    mock_delete.return_value = subprocess.CompletedProcess(
        args=["kubectl", "delete"],
        returncode=0,
        stdout=b"jobset.jobset.x-k8s.io \"pathways-interactive\" deleted",
        stderr=b""
    )
    
    runner = CliRunner()
    result = runner.invoke(main, ["down"])
    
    assert result.exit_code == 0
    assert "Deleting Pathways JobSet 'pathways-interactive'" in result.output
    assert "Successfully deleted JobSet 'pathways-interactive'!" in result.output
    mock_delete.assert_called_once()
