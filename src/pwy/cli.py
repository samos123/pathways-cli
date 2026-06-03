import sys
import os
import shlex
import click
import getpass
import re
from dotenv import load_dotenv, find_dotenv
from pwy.generator import generate_yaml
from pwy.kubernetes import (
    apply_manifest,
    delete_jobset,
    wait_for_client_pod,
    wait_for_pod_ready,
    get_client_pod_name,
)
from pwy.sync import sync_directory

# Load environment variables from .env file if present
load_dotenv(find_dotenv(usecwd=True))


def get_default_name() -> str:
    """Returns the default JobSet name dynamically based on the current user."""
    try:
        username = getpass.getuser()
    except Exception:
        username = "user"

    # Sanitize username to conform to DNS subdomain / JobSet name rules
    username = username.lower()
    username = re.sub(r"[^a-z0-9]+", "-", username)
    username = username[:60].strip("-")

    if not username:
        username = "user"

    return f"{username}-pw"


def validate_jobset_name(ctx, param, value):
    """Validates that a string is a valid Kubernetes JobSet name."""
    if not value:
        raise click.BadParameter("Name cannot be empty.")
    if len(value) > 63:
        raise click.BadParameter("Name must be 63 characters or less.")
    pattern = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
    if not re.match(pattern, value):
        raise click.BadParameter(
            "Name must consist of lowercase alphanumeric characters or '-', and must start and end with an alphanumeric character."
        )
    return value


@click.group()
def main():
    """pwy: Standalone Pathways GKE Cluster CLI Tool"""
    pass


@main.command()
@click.option(
    "--tpu-type",
    required=True,
    envvar="PWY_TPU_TYPE",
    help="TPU type (e.g., v6e-4, v6e-8, v6e-16, v6e-32, v6e-64)",
)
@click.option(
    "--gcs-scratch-location",
    required=True,
    envvar="PWY_GCS_SCRATCH_LOCATION",
    help="GCS scratch location (e.g., gs://bucket/staging)",
)
@click.option(
    "--num-slices",
    default=1,
    type=int,
    show_default=True,
    envvar="PWY_NUM_SLICES",
    help="Number of TPU slices",
)
@click.option(
    "--jax-client-image",
    default="python:3.12-slim",
    show_default=True,
    envvar="PWY_JAX_CLIENT_IMAGE",
    help="Image for the JAX client container",
)
@click.option(
    "--command",
    default=None,
    envvar="PWY_COMMAND",
    help="Command to run in the JAX client container (defaults to sleep infinity)",
)
@click.option(
    "--spot",
    is_flag=True,
    default=False,
    envvar="PWY_SPOT",
    help="Enable spot VM scheduling",
)
@click.option(
    "--colocated-python",
    is_flag=True,
    default=False,
    envvar="PWY_COLOCATED_PYTHON",
    help="Enable colocated python sidecars",
)
@click.option(
    "--head-on-tpu/--no-head-on-tpu",
    is_flag=True,
    default=True,
    envvar="PWY_HEAD_ON_TPU",
    help="Co-schedule the JAX client (head) pod on the same TPU VM slice as the workers",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    envvar="PWY_DRY_RUN",
    help="Dry run: print generated YAML to stdout instead of applying it",
)
@click.option(
    "--name",
    default=get_default_name,
    callback=validate_jobset_name,
    show_default="$USER-pw",
    envvar="PWY_NAME",
    help="Name of the JobSet resource",
)
@click.option(
    "--namespace",
    default="default",
    show_default=True,
    envvar="PWY_NAMESPACE",
    help="Kubernetes namespace",
)
@click.option(
    "--sync",
    default=None,
    envvar="PWY_SYNC",
    help="Local directory path to sync to the JAX client container",
)
@click.option(
    "--remote-path",
    default="/app",
    show_default=True,
    envvar="PWY_REMOTE_PATH",
    help="Destination path in the remote JAX client container",
)
def up(
    tpu_type,
    gcs_scratch_location,
    num_slices,
    jax_client_image,
    command,
    spot,
    colocated_python,
    head_on_tpu,
    dry_run,
    name,
    namespace,
    sync,
    remote_path,
):
    """Starts the Pathways cluster or dry-runs the configuration."""
    try:
        yaml_content = generate_yaml(
            name=name,
            namespace=namespace,
            tpu_type=tpu_type,
            gcs_scratch_location=gcs_scratch_location,
            num_slices=num_slices,
            jax_client_image=jax_client_image,
            command=command,
            spot=spot,
            colocated_python=colocated_python,
            head_on_tpu=head_on_tpu,
            sync=bool(sync),
            remote_path=remote_path,
        )
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    if dry_run:
        click.echo(yaml_content)
        return

    click.echo(
        f"Applying Pathways JobSet manifest for '{name}' in namespace '{namespace}'..."
    )
    process = apply_manifest(yaml_content)
    if process.returncode != 0:
        click.secho("Failed to apply JobSet manifest.", fg="red", err=True)
        click.echo(process.stderr.decode("utf-8"), err=True)
        sys.exit(process.returncode)

    click.secho(f"Successfully applied JobSet '{name}'!", fg="green")

    if sync:
        click.echo("Waiting for JAX client pod to be created...")
        try:
            pod_name = wait_for_client_pod(name, namespace)
            click.echo(
                f"Client pod created: {pod_name}. Waiting for pod to be Running and Ready..."
            )
            if wait_for_pod_ready(pod_name, namespace):
                click.echo(
                    f"Syncing local path '{sync}' to '{pod_name}:{remote_path}'..."
                )
                sync_directory(sync, remote_path, pod_name, namespace)
            else:
                click.secho(
                    "Timed out waiting for client pod to become ready. Skipping initial sync.",
                    fg="yellow",
                    err=True,
                )
        except Exception as e:
            click.secho(f"Failed to perform initial sync: {e}", fg="red", err=True)


@main.command()
@click.option(
    "--name",
    default=get_default_name,
    callback=validate_jobset_name,
    show_default="$USER-pw",
    envvar="PWY_NAME",
    help="Name of the JobSet resource",
)
@click.option(
    "--namespace",
    default="default",
    show_default=True,
    envvar="PWY_NAMESPACE",
    help="Kubernetes namespace",
)
def down(name, namespace):
    """Tears down the Pathways cluster JobSet resource."""
    click.echo(f"Deleting Pathways JobSet '{name}' in namespace '{namespace}'...")
    process = delete_jobset(name, namespace)
    if process.returncode != 0:
        click.secho("Failed to delete JobSet.", fg="red", err=True)
        click.echo(process.stderr.decode("utf-8"), err=True)
        sys.exit(process.returncode)

    click.secho(f"Successfully deleted JobSet '{name}'!", fg="green")


@main.command()
@click.option(
    "--name",
    default=get_default_name,
    callback=validate_jobset_name,
    show_default="$USER-pw",
    envvar="PWY_NAME",
    help="Name of the JobSet resource",
)
@click.option(
    "--namespace",
    default="default",
    show_default=True,
    envvar="PWY_NAMESPACE",
    help="Kubernetes namespace",
)
@click.option(
    "--source",
    default=".",
    show_default=True,
    help="Local source path to sync",
)
@click.option(
    "--dest",
    default="/app",
    show_default=True,
    help="Destination path in the remote container",
)
def sync(name, namespace, source, dest):
    """Syncs local files to the JAX client container."""
    try:
        pod_name = get_client_pod_name(name, namespace)
    except RuntimeError as e:
        click.secho(str(e), fg="red", err=True)
        sys.exit(1)

    click.echo(f"Syncing local path '{source}' to '{pod_name}:{dest}'...")
    sync_directory(source, dest, pod_name, namespace)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.option(
    "--name",
    default=get_default_name,
    callback=validate_jobset_name,
    show_default="$USER-pw",
    envvar="PWY_NAME",
    help="Name of the JobSet resource",
)
@click.option(
    "--namespace",
    default="default",
    show_default=True,
    envvar="PWY_NAMESPACE",
    help="Kubernetes namespace",
)
@click.option(
    "--sync/--no-sync",
    default=True,
    show_default=True,
    help="Sync current working directory before running the command",
)
@click.option(
    "--source",
    default=".",
    show_default=True,
    help="Local source path to sync",
)
@click.option(
    "--dest",
    default="/app",
    show_default=True,
    help="Destination path in the remote container",
)
@click.argument("command_args", nargs=-1, required=True, type=click.UNPROCESSED)
def run(name, namespace, sync, source, dest, command_args):
    """Syncs local directory and runs a command in the JAX client container."""
    try:
        pod_name = get_client_pod_name(name, namespace)
    except RuntimeError as e:
        click.secho(str(e), fg="red", err=True)
        sys.exit(1)

    if sync:
        click.echo(f"Syncing local path '{source}' to '{pod_name}:{dest}'...")
        sync_directory(source, dest, pod_name, namespace)

    # Wrap the command to run within the destination directory
    sh_command = f"mkdir -p {shlex.quote(dest)} && cd {shlex.quote(dest)} && exec {shlex.join(command_args)}"

    exec_args = ["kubectl", "exec"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        exec_args.append("-it")
    else:
        exec_args.append("-i")

    exec_args.extend(
        ["-n", namespace, "-c", "client", pod_name, "--", "bash", "-c", sh_command]
    )

    click.echo(f"Executing command: {' '.join(command_args)}")

    try:
        os.execvp("kubectl", exec_args)
    except OSError as e:
        click.secho(f"Failed to execute kubectl command: {e}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
