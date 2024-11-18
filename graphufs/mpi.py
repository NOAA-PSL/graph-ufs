import os
import logging
import jax

from graphufs.log import SimpleFormatter

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

    def __init__(self, log_dir=None, log_level=logging.INFO):

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

        self._init_log(log_dir=log_dir, level=log_level)
        logging.info(str(self))


    def __str__(self):
        msg = "\nEnvironment Info\n" +\
            "----------------\n" +\
            jax.print_environment_info(return_string=True) +\
            "\n\n" +\
            f"MPITopology Summary\n" +\
            f"-------------------\n" +\
            f"comm: {self.comm.Get_name()}\n"
        for key in ["node", "local_rank", "rank", "local_size", "size"]:
            msg += f"{key:<18s}: {getattr(self, key):02d}\n"
        msg += f"{'local_devices':<18s}: {str(self.local_devices)}\n" +\
            f"{'rank_device':<18s}: {str(self.rank_device)}\n"
        return msg

    def _init_log(self, log_dir, level=logging.INFO):
        self.log_dir = "./" if log_dir is None else log_dir
        if self.is_root:
            if not os.path.isdir(self.log_dir):
                os.makedirs(self.log_dir)
        self.comm.Barrier()
        self.logfile = f"{self.log_dir}/log.{self.rank:02d}.{self.size:02d}.out"

        logging.basicConfig(
            level=level,
            filename=self.logfile,
            filemode="w",
        )
        logger = logging.getLogger()
        formatter = SimpleFormatter(fmt="[%(relativeCreated)-7d s] [%(levelname)-7s] %(message)s")
        for handler in logger.handlers:
            handler.setFormatter(formatter)

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
        """Take an average across all devices"""
        def local_mean(local):
            local_avg, _ = mpi4jax.allreduce(
                local,
                op=MPI.SUM,
                comm=self.comm,
            )
            return local_avg / self.size
        return jax.tree_util.tree_map(
            lambda x: local_mean(x),
            array,
        )

    def _tree_flatten(self):
        children = tuple()
        aux_data = dict(log_dir=self.log_dir, log_level=self.log_level)
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

jax.tree_util.register_pytree_node(
    MPITopology,
    MPITopology._tree_flatten,
    MPITopology._tree_unflatten,
)
