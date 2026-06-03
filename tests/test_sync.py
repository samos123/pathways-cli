from unittest.mock import patch
import subprocess
from pwy.sync import (
    get_client_pod_name,
    install_rsync_if_needed,
    run_rsync_sync,
    run_fallback_sync,
    sync_directory,
)


@patch("subprocess.run")
def test_get_client_pod_name(mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kubectl"], returncode=0, stdout="my-sync-pod\n", stderr=""
    )
    pod = get_client_pod_name("my-jobset", "my-ns")
    assert pod == "my-sync-pod"
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_install_rsync_if_needed_already_installed(mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kubectl"], returncode=0, stdout=b"/usr/bin/rsync", stderr=b""
    )
    assert install_rsync_if_needed("my-pod", "my-ns") is True
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_install_rsync_if_needed_install_success(mock_run):
    # 1st call (which rsync) returns 1, 2nd call (apt-get install) returns 0
    mock_run.side_effect = [
        subprocess.CompletedProcess(
            args=["kubectl", "exec"], returncode=1, stdout=b"", stderr=b""
        ),
        subprocess.CompletedProcess(
            args=["kubectl", "exec"],
            returncode=0,
            stdout=b"rsync installed",
            stderr=b"",
        ),
    ]
    assert install_rsync_if_needed("my-pod", "my-ns") is True
    assert mock_run.call_count == 2


@patch("subprocess.run")
def test_install_rsync_if_needed_install_fail(mock_run):
    # 1st call returns 1, 2nd call returns 1
    mock_run.side_effect = [
        subprocess.CompletedProcess(
            args=["kubectl", "exec"], returncode=1, stdout=b"", stderr=b""
        ),
        subprocess.CompletedProcess(
            args=["kubectl", "exec"],
            returncode=1,
            stdout=b"",
            stderr=b"Failed to install",
        ),
    ]
    assert install_rsync_if_needed("my-pod", "my-ns") is False
    assert mock_run.call_count == 2


@patch("subprocess.run")
def test_run_rsync_sync_success(mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["rsync"], returncode=0, stdout=b"", stderr=b""
    )
    assert run_rsync_sync(".", "/app", "my-pod", "my-ns") is True
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_run_fallback_sync_success(mock_run):
    # 1st call (mkdir) returns 0, 2nd call (kubectl cp) returns 0
    mock_run.side_effect = [
        subprocess.CompletedProcess(
            args=["kubectl", "exec"], returncode=0, stdout=b"", stderr=b""
        ),
        subprocess.CompletedProcess(
            args=["kubectl", "cp"], returncode=0, stdout=b"", stderr=b""
        ),
    ]
    assert run_fallback_sync(".", "/app", "my-pod", "my-ns") is True
    assert mock_run.call_count == 2


@patch("pwy.sync.install_rsync_if_needed")
@patch("pwy.sync.run_rsync_sync")
@patch("pwy.sync.run_fallback_sync")
def test_sync_directory_prefer_rsync(mock_fallback, mock_rsync, mock_install):
    mock_install.return_value = True
    mock_rsync.return_value = True

    sync_directory(".", "/app", "my-pod", "my-ns")

    mock_install.assert_called_once_with("my-pod", "my-ns")
    mock_rsync.assert_called_once_with(".", "/app", "my-pod", "my-ns")
    mock_fallback.assert_not_called()


@patch("pwy.sync.install_rsync_if_needed")
@patch("pwy.sync.run_rsync_sync")
@patch("pwy.sync.run_fallback_sync")
def test_sync_directory_fallback_on_rsync_failure(
    mock_fallback, mock_rsync, mock_install
):
    mock_install.return_value = True
    # Rsync fails, should trigger fallback
    mock_rsync.return_value = False

    sync_directory(".", "/app", "my-pod", "my-ns")

    mock_install.assert_called_once_with("my-pod", "my-ns")
    mock_rsync.assert_called_once_with(".", "/app", "my-pod", "my-ns")
    mock_fallback.assert_called_once_with(".", "/app", "my-pod", "my-ns")
