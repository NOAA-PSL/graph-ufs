# Perlmutter Environment Tests

## 1. Test JAX

```
python test_jax.py
```

## 2. Test mpi4py

```
srun python test_mpi4py.py
```

Expected output (ordering might be different, but there should be 8 lines total
if using 4 processes/GPUs)

```
None
{'key1': [7, 2.72, (2+3j)], 'key2': ('abc', 'xyz')}
{'key1': [7, 2.72, (2+3j)], 'key2': ('abc', 'xyz')}
{'key1': [7, 2.72, (2+3j)], 'key2': ('abc', 'xyz')}
None
{'key1': [7, 2.72, (2+3j)], 'key2': ('abc', 'xyz')}
None
{'key1': [7, 2.72, (2+3j)], 'key2': ('abc', 'xyz')}
```

## 3. Test mpi4jax

```
export MPI4JAX_USE_CUDA_MPI=1
srun ./select_gpu_device python test_mpi4jax.py
```

Expected output

```
[[6. 6. 6.]
 [6. 6. 6.]
 [6. 6. 6.]]
```

## 4. Test our usage of mpi4jax

```
srun ./select_gpu_device python test_mpitopo.py
```

Expected output

```
[[1.5 1.5 1.5]
 [1.5 1.5 1.5]
 [1.5 1.5 1.5]]
grid2mesh_gnn/~_networks_builder/encoder_edges_grid2mesh_layer_norm
```

## 5. Test multithreading + mpi4py

```
srun python test_threaded_mpi4py.py
```

Expected output: errors that look like this

```
Assertion failed in file ../src/mpi/init/initthread.c at line 361: MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX.count >= 0
/opt/cray/pe/lib64/libmpi_gnu_123.so.12(MPL_backtrace_show+0x26) [0x7f4480ce42bb]
/opt/cray/pe/lib64/libmpi_gnu_123.so.12(+0x1b434b4) [0x7f44807434b4]
/opt/cray/pe/lib64/libmpi_gnu_123.so.12(PMPI_Init_thread+0x158) [0x7f447f205658]
/global/common/software/nersc9/darshan/default/lib/libdarshan.so.0(PMPI_Init_thread+0x46) [0x7f4481322326]
/global/common/software/m4718/timothys/gnugraphufs/lib/python3.11/site-packages/mpi4py/MPI.cpython-311-x86_64-linux-gnu.so(+0x169ae6) [0x7f448315fae6]
python(PyObject_Vectorcall+0x2c) [0x55fcffb1b6ac]
python(_PyEval_EvalFrameDefault+0x694) [0x55fcffb0ef94]
python(+0x29ac5d) [0x55fcffbc5c5d]
python(PyEval_EvalCode+0x9f) [0x55fcffbc539f]
python(+0x2b831a) [0x55fcffbe331a]
python(+0x2b3f93) [0x55fcffbdef93]
python(+0x2c9540) [0x55fcffbf4540]
python(_PyRun_SimpleFileObject+0x1bc) [0x55fcffbf3ecc]
python(_PyRun_AnyFileObject+0x44) [0x55fcffbf3c64]
python(Py_RunMain+0x383) [0x55fcffbee233]
python(Py_BytesMain+0x37) [0x55fcffbb5617]
/lib64/libc.so.6(__libc_start_main+0xef) [0x7f448352624d]
python(+0x28a4ca) [0x55fcffbb54ca]
MPICH ERROR [Rank 1] [job id 33190617.7] [Fri Nov 22 08:45:32 2024] [nid008200] - Abort(1): Internal error
```
