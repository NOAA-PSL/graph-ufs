import os
import logging
import jax

try:
    from mpi4py import MPI
    import mpi4jax
    _has_mpi = True

except:
    _has_mpi = False
    logging.warning(f"graphufs.mpi: Unable to import mpi4py or mpi4jax, cannot use this module")


class MPITopology():
    """Note that all of this usage assumes that we have one process per GPU"""

    @property
    def is_root(self):
        return self.rank == self.root

    def __init__(self, logdir=None):

        assert _has_mpi, f"MPITopology.__init__: Unable to import mpi4py or mpi4jax, cannot use this class"
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.local_size = len(jax.local_devices())
        self.node = self.rank // self.local_size
        self.local_rank = (self.rank - self.node*self.local_size) % self.local_size
        self.local_devices = jax.local_devices()
        self.rank_device = self.local_devices[self.local_rank]
        self.root = 0
        self.friends = tuple(ii for ii in range(self.size) if ii!=self.root)

        self.logdir = "./" if logdir is None else logdir
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        self.logfile = f"{self.logdir}/log.{self.rank:02d}.{self.size:02d}.out"
        self.log(str(self), mode="w")


    def __str__(self):
        msg = "MPITopology Summary\n" +\
            f"comm: {self.comm.Get_name()}\n"
        for key in ["node", "local_rank", "rank", "local_size", "size"]:
            msg += f"{key:<16s}: {getattr(self, key):02d}\n"
        msg += f"local_devices   : {str(self.local_devices)}\n" +\
            f"rank_device     : {str(self.rank_device)}\n"
        return msg

    def log(self, msg, mode="a"):
        with open(self.logfile, mode=mode) as f:
            print(msg, file=f)

    def bcast(self, x):
        return self.comm.bcast(x, root=self.root)



    def device_put(self, x, **kwargs):
        device = kwargs.get("device", self.rank_device)
        return jax.device_put(x, **kwargs)

    def device_bcast(self, x, token=None):
        return mpi4jax.bcast(
            x=x,
            root=self.root,
            comm=self.comm,
            token=token,
        )

    def device_mean(self, array):
        def local_mean(local):
            local_avg, _ = mpi4jax.allreduce(
                local,
                op=MPI.SUM,
                comm=self.comm,
            )
            return local / self.size
        return jax.tree_util.tree_map(
            lambda x: local_mean(x),
            array,
        )

    def _tree_flatten(self):
        children = tuple()
        aux_data = dict(logdir=self.logdir)
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

jax.tree_util.register_pytree_node(
    MPITopology,
    MPITopology._tree_flatten,
    MPITopology._tree_unflatten,
)
