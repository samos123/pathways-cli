import sys
import click
from pwy.generator import generate_yaml
from pwy.kubernetes import apply_manifest, delete_jobset

@click.group()
def main():
    """pwy: Standalone Pathways GKE Cluster CLI Tool"""
    pass

@main.command()
@click.option("--tpu-type", required=True, help="TPU type (e.g., v6e-4, v6e-8, v6e-16, v6e-32, v6e-64)")
@click.option("--gcs-scratch-location", required=True, help="GCS scratch location (e.g., gs://bucket/staging)")
@click.option("--num-slices", default=1, type=int, show_default=True, help="Number of TPU slices")
@click.option("--jax-client-image", default="python:3.12-slim", show_default=True, help="Image for the JAX client container")
@click.option("--command", default=None, help="Command to run in the JAX client container (defaults to sleep infinity)")
@click.option("--enable-spot", is_flag=True, default=False, help="Enable spot VM scheduling")
@click.option("--colocated-python", is_flag=True, default=False, help="Enable colocated python sidecars")
@click.option("--dry-run", is_flag=True, default=False, help="Dry run: print generated YAML to stdout instead of applying it")
@click.option("--name", default="pathways-interactive", show_default=True, help="Name of the JobSet resource")
@click.option("--namespace", default="default", show_default=True, help="Kubernetes namespace")
def up(tpu_type, gcs_scratch_location, num_slices, jax_client_image, command, enable_spot, colocated_python, dry_run, name, namespace):
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
            enable_spot=enable_spot,
            colocated_python=colocated_python,
        )
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    if dry_run:
        click.echo(yaml_content)
        return

    click.echo(f"Applying Pathways JobSet manifest for '{name}' in namespace '{namespace}'...")
    process = apply_manifest(yaml_content)
    if process.returncode != 0:
        click.secho("Failed to apply JobSet manifest.", fg="red", err=True)
        click.echo(process.stderr.decode("utf-8"), err=True)
        sys.exit(process.returncode)

    click.secho(f"Successfully applied JobSet '{name}'!", fg="green")

@main.command()
@click.option("--name", default="pathways-interactive", show_default=True, help="Name of the JobSet resource")
@click.option("--namespace", default="default", show_default=True, help="Kubernetes namespace")
def down(name, namespace):
    """Tears down the Pathways cluster JobSet resource."""
    click.echo(f"Deleting Pathways JobSet '{name}' in namespace '{namespace}'...")
    process = delete_jobset(name, namespace)
    if process.returncode != 0:
        click.secho("Failed to delete JobSet.", fg="red", err=True)
        click.echo(process.stderr.decode("utf-8"), err=True)
        sys.exit(process.returncode)

    click.secho(f"Successfully deleted JobSet '{name}'!", fg="green")

if __name__ == "__main__":
    main()
