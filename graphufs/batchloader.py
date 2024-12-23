from math import ceil
import numpy as np

import logging
import threading
import concurrent
import queue

from .mpi import MPITopology, _has_mpi

class BatchLoader():
    """

    Usage
        # --- Method 1
        ds = LocalDataset (or just Dataset)
        trainer = BatchLoader(ds, ...) # queue gets filled immediately
        for k, (x,y) in enumerate(trainer): # restarts counter, but not the thread queue, and does not re-shuffle
            loss = train(x,y)

        # recommended to restart threads to fill queue, e.g. while doing validation
        # if shuffle=True it happens here
        trainer.restart()

        # --- Method 2
        trainer = BatchLoader(ds, ...) # queue gets filled immediately
        for k in range(len(trainer)):
            x,y = trainer.get_data() # pull next item from queue
            loss = train(x,y)

        # again recommended to restart here
        trainer.restart()



    """
    counter = 0
    data_counter = 0

    stop_event = None
    executor = None
    futures = None

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        drop_last=True,
        num_workers=0,
        max_queue_size=1,
        rng_seed=None,
        sample_stride=None,
        start=0,
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.counter = start
        self.data_counter = start

        self.sample_indices = list(int(idx) for idx in np.arange(len(self.dataset)))
        self.sample_stride = sample_stride
        if sample_stride is not None:
            self.sample_indices = self.sample_indices[::sample_stride]
        self.rstate = np.random.RandomState(rng_seed)

        self.num_workers = num_workers
        assert max_queue_size > 0
        max_queue_size = min(max_queue_size, len(self))
        self.max_queue_size = max_queue_size

        # create a separate lock for each of the attributes
        # that get changed, so threads don't bump into each other
        # It's important to have separate locks so we can lock
        # the state of each attribute separately
        self.counter_lock = threading.Lock()
        self.data_counter_lock = threading.Lock()
        self.executor_lock = threading.Lock()
        if self.num_workers > 0:
            self.data_queue = queue.Queue(maxsize=max_queue_size)
            self.stop_event = threading.Event()

        self.restart(idx=start)

    @property
    def initial_times(self) -> list[np.datetime64]:
        """Returns dates of all initial conditions"""
        return self.dataset.initial_times[::self.sample_stride]

    @property
    def mode(self):
        return self.dataset.mode

    @property
    def emulator(self):
        return self.dataset.emulator

    @property
    def preload_batch(self):
        return self.dataset.preload_batch

    @property
    def xds(self):
        return self.dataset.xds

    @property
    def name(self):
        return str(type(self).__name__)

    def __len__(self) -> int:
        n_samples = len(self.sample_indices)
        if self.drop_last:
            n_batches = n_samples // self.batch_size
        else:
            n_batches = ceil(n_samples / self.batch_size)
        return n_batches

    def __iter__(self):
        with self.counter_lock:
            self.counter = 0

        # Always restart in the serial case
        if self.num_workers == 0:
            self.restart()
        else:
            # in the parallel case, we don't want to unnecessarily clear the queue,
            # so we only restart if we've been grabbing data willy nilly
            # and we've exceeded the queue size
            # Also we restart if the BatchLoader was previously shutdown and needs a kick start
            if self.stop_event.is_set() or self.data_counter > self.max_queue_size:
                self.restart()
        return self

    def __next__(self):
        """Note that self.counter is the counter for looping with e.g. enumerate
        (i.e., how much has been removed from the queue)
        whereas self.data_counter is keeping track of how many data items have been put in the queue
        """
        if self.counter < len(self):
            data = self.get_data()
            with self.counter_lock:
                self.counter += 1
            return data
        else:
            logging.debug(f"{self.mode} BatchLoader.__next__: counter > len(self)")
            raise StopIteration

    def _next_data(self):
        logging.debug(f"{self.mode} BatchLoader._next_data[{self.data_counter}]")

        if self.data_counter < len(self):
            st = self.data_counter * self.batch_size
            ed = st + self.batch_size
            batch_indices = self.sample_indices[st:ed]
            x, y = self.dataset[batch_indices]
            return x.values, y.values
        else:
            logging.debug(f"{self.mode} BatchLoader._next_data: data_counter > len(self)")
            raise StopIteration

    def generate(self):
        while not self.stop_event.is_set():
            try:
                data = self._next_data()
                self.data_queue.put(data)
                with self.data_counter_lock:
                    self.data_counter += 1
                logging.debug(f"{self.dataset.mode} done putting")
            except StopIteration:
                self.shutdown()

    def get_data(self):
        """Pull a batch of data from the queue"""
        logging.debug(f"{self.dataset.mode} BatchLoader.get_data")
        if self.num_workers > 0:
            data = self.data_queue.get()
            self.task_done()
            return data
        else:
            data = self._next_data()
            with self.data_counter_lock:
                self.data_counter += 1
            return data

    def task_done(self):
        self.data_queue.task_done()
        logging.debug(f"{self.dataset.mode} BatchLoader: marked task_done")

    def restart(self, idx=0, cancel=False, **kwargs):
        """Restart the :attr:`data_counter` and ThreadPoolExecutor to get ready for the pass through the data

        Args:
            cancel (bool): if True, cancel any remaining queue items/tasks with :meth:`.cancel`
        """
        logging.debug(f"{self.dataset.mode} BatchLoader.restart")

        if self.shuffle:
            self.rstate.shuffle(self.sample_indices)
            self.sample_indices = list(int(i) for i in self.sample_indices)

        # start filling the queue
        if self.num_workers > 0:

            if self.executor is not None:
                self.shutdown(cancel=cancel, **kwargs)
                self.stop_event.clear()

            with self.data_counter_lock:
                self.data_counter = idx

            with self.executor_lock:
                self.executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.num_workers,
                )
                self.futures = [
                    self.executor.submit(self.generate) for _ in range(self.num_workers)
                ]
        else:
            self.data_counter = idx


    def cancel(self):
        """Cancel any remaining workers/queue items by calling :meth:`get_data` until they
        can recognize that the stop_event has been set
        """
        # cancel the existing workers/queue to force a startover
        i = 1
        if self.num_workers > 0:
            while not self.data_queue.empty():
                logging.debug(f"{self.mode} BatchLoader.cancel: Queue not empty. (count, data_count) = ({self.counter}, {self.data_counter})... getting data {i}")
                self.get_data()
                i+=1

    def shutdown(self, cancel=False, **kwargs):
        """Shutdown the ThreadPoolExecutor.

        Args:
            cancel (bool): If true, cancel any remaining tasks...
                Don't do this right after a for loop though, since the for loop may not finish due to a deadlock
        """
        logging.debug(f"{self.dataset.mode} BatchLoader.shutdown")
        if self.num_workers > 0:
            self.stop_event.set()
            if cancel:
                self.cancel()
            with self.executor_lock:
                self.executor.shutdown(**kwargs)
                # set executor to None, so that if shutdown is called from within a loop
                # and the BatchLoader is restarted immediately after, we don't get a double shutdown call
                self.executor = None

class XBatchLoader(BatchLoader):
    """Returns xarray DataArrays with __getitem__ instead of numpy like arrays
    Useful for preprocessing instead of training
    """
    def _next_data(self):
        logging.debug(f"{self.mode} BatchLoader._next_data[{self.data_counter}]")

        if self.data_counter < len(self):
            st = self.data_counter * self.batch_size
            ed = st + self.batch_size
            batch_indices = self.sample_indices[st:ed]
            x, y = self.dataset[batch_indices]
            x.load()
            y.load()
            return x, y
        else:
            logging.debug(f"{self.mode} BatchLoader.__next__: counter > len(self)")
            raise StopIteration

class ExpandedBatchLoader(BatchLoader):
    """Returns xarray DataArrays ready for original GraphCast, not Stacked form
    """
    def _next_data(self):
        logging.debug(f"{self.mode} BatchLoader._next_data[{self.data_counter}]")

        if self.data_counter < len(self):
            st = self.data_counter * self.batch_size
            ed = st + self.batch_size
            batch_indices = self.sample_indices[st:ed]
            data = self.dataset.get_batch_of_xarrays(batch_indices)
            return (d.load() for d in data)
        else:
            logging.debug(f"{self.mode} BatchLoader.__next__: counter > len(self)")
            raise StopIteration

class MPIBatchLoader(BatchLoader):
    """Make sure mpi4py and mpi4jax is installed
    """
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        mpi_topo,
        drop_last=True,
        num_workers=0,
        max_queue_size=1,
        rng_seed=None,
        sample_stride=None,
        start=0,
    ):
        assert _has_mpi, f"{self.name}.__init__: Unable to import mpi4py or mpi4jax, cannot use this class"

        self.topo = mpi_topo
        self.data_per_device = batch_size // self.topo.size
        self.local_batch_index = self.topo.rank*self.data_per_device
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            max_queue_size=max_queue_size,
            rng_seed=rng_seed,
            sample_stride=sample_stride,
            start=start,
        )
        logging.info(str(self))
        if shuffle:
            assert rng_seed is not None, f"{self.name}.__init__: need to set rng_seed in order for processes to be in sync without collectives"

        if self.data_per_device*self.topo.size != batch_size:
            logging.warning(f"{self.name}.__init__: batch_size = {batch_size} not divisible by MPI Size = {self.topo.size}")
            logging.warning(f"{self.name}.__init__: some data will be skipped in each batch")

        if batch_size > 1 and not drop_last:
            logging.warning(f"{self.name}.__init__: with batch_size>1 and drop_last=False, some MPI processes may grab incorrect indices in last batch. Expect an error at the end of the dataset")

    def __str__(self):
        myname = f"{__name__}.{self.name}"
        underline = "".join(["-" for _ in range(len(myname))])
        msg = f"\n{myname}\n{underline}\n" +\
            f"{'mode':<18s}: {self.mode}\n"

        for key in ["local_batch_index", "data_per_device", "batch_size"]:
            msg += f"{key:<18s}: {getattr(self, key):02d}\n"
        return msg


    def _next_data(self):
        if self.data_counter < len(self):
            st = (self.data_counter * self.batch_size) + self.local_batch_index
            ed = st + self.data_per_device
            batch_indices = self.sample_indices[st:ed]
            x, y = self.dataset[batch_indices]
            return x.values, y.values
        else:
            raise StopIteration

class MPIXBatchLoader(MPIBatchLoader):
    def _next_data(self):
        if self.data_counter < len(self):
            st = (self.data_counter * self.batch_size) + self.local_batch_index
            ed = st + self.data_per_device
            batch_indices = self.sample_indices[st:ed]

            if len(batch_indices) > 0:
                x, y = self.dataset[batch_indices]
                return x.load(), y.load()
            elif len(batch_indices) == 0 and not self.drop_last:
                return None, None
            else:
                raise IndexError(f"[Rank {self.topo.rank}] {self.name}._next_data: looking for indices [{st}:{ed}], but len(sample_indices) = {len(self.sample_indices)}")
        else:
            raise StopIteration

class MPIExpandedBatchLoader(MPIBatchLoader):
    def _next_data(self):
        if self.data_counter < len(self):
            st = (self.data_counter * self.batch_size) + self.local_batch_index
            ed = st + self.data_per_device
            batch_indices = self.sample_indices[st:ed]
            if len(batch_indices) > 0:
                data = self.dataset.get_batch_of_xarrays(batch_indices)
                return (d.load() for d in data)
            elif len(batch_indices) == 0 and not self.drop_last:
                return None, None, None
            else:
                raise IndexError(f"[Rank {self.topo.rank}] {self.name}._next_data: looking for indices [{st}:{ed}], but len(sample_indices) = {len(self.sample_indices)}")
        else:
            raise StopIteration
