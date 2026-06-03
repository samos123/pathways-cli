#!/usr/bin/env python3
import sys
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding

# 1. Initialize Pathways if pathwaysutils is available
try:
    import pathwaysutils

    print("Detected pathwaysutils. Initializing Pathways...")
    pathwaysutils.initialize()
    print("Pathways initialized successfully.")
except ImportError:
    print("pathwaysutils not found, running on default JAX backend.")

# 2. Print JAX devices
devices = jax.devices()
print(f"Available JAX devices: {devices}")
num_devices = len(devices)
if num_devices == 0:
    print("Error: No JAX devices found.")
    sys.exit(1)

# 3. Create a 1D Mesh over all available JAX devices
mesh = Mesh(devices, ("x",))
print(f"Created JAX mesh: {mesh}")

# 4. Define NamedSharding to partition the first axis (axis 0) of the array across the mesh
sharding = NamedSharding(mesh, P("x", None))

# 5. Create a large array on the devices with the specified sharding
# Shape is (num_devices, 1024, 1024) to ensure each device gets a (1024, 1024) slice
global_shape = (num_devices, 1024, 1024)
print(
    f"Creating a global array of shape {global_shape} sharded over {num_devices} devices..."
)
x = jax.device_put(jnp.ones(global_shape, dtype=jnp.float32), sharding)

# Verify the array shape and sharding
print(f"Array shape: {x.shape}")
print(f"Array sharding: {x.sharding}")


# 6. Define a JIT-compiled computation
@jax.jit
def simple_computation(arr):
    # Element-wise operations which preserve the sharding
    return arr * 3.5 + 4.2


# 7. Run the calculation
print("Running JIT calculation...")
result = simple_computation(x)

# 8. Force execution and verify results
result.block_until_ready()
print("Calculation complete!")
print(f"Result shape: {result.shape}")
print(f"Result sharding: {result.sharding}")
print(f"Result slice sample (device 0, indices [0, 0, 0:5]): {result[0, 0, :5]}")

# 9. Assert expected values
expected_val = 1.0 * 3.5 + 4.2
actual_val = float(result[0, 0, 0])
assert (
    abs(actual_val - expected_val) < 1e-5
), f"Expected {expected_val}, got {actual_val}"
print("Verification check passed successfully!")
