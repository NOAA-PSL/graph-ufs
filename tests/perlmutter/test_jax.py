import jax
import jax.numpy as jnp

@jax.jit
def foo(arr):
    return arr.sum()

a = jnp.ones((3, 3))
result = foo(a)

print(result)
