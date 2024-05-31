import numpy as np

import threading
import concurrent
import queue

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
    stop_event = None
    lock = threading.Lock()

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        drop_last=True,
        num_workers=0,
        max_queue_size=1,
        rng_seed=None,
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.counter = 0
        self.sample_indices = list(int(idx) for idx in np.arange(len(self.dataset)))
        self.rstate = np.random.RandomState(rng_seed)

        self.num_workers = num_workers
        assert max_queue_size > 0
        max_queue_size = min(max_queue_size, len(self))
        self.max_queue_size = max_queue_size
        self.data_queue = queue.Queue(maxsize=max_queue_size)

        self.restart()

    def __len__(self) -> int:
        n_samples = len(self.dataset)
        if self.drop_last:
            n_batches = n_samples // self.batch_size
        else:
            n_batches = ceil(n_samples / self.batch_size)
        return n_batches

    def __iter__(self):
        self.counter = 0
        # if the threads have not been restarted, do it now
        if self.stop_event.is_set() or self.num_workers == 0:
            self.restart()
        return self

    def __next__(self):
        """Note that self.counter is the counter for looping with e.g. enumerate
        (i.e., how much has been removed from the queue)
        whereas self.data_counter is keeping track of how many data items have been put in the queue
        """
        if self.counter < len(self):
            x, y = self.get_data()
            self.counter += 1
            return x, y
        else:
            raise StopIteration

    def _next_data(self):

        if self.data_counter < len(self):
            st = self.data_counter * self.batch_size
            ed = st + self.batch_size
            batch_indices = self.sample_indices[st:ed]
            x, y = self.dataset[batch_indices]
            self.data_counter += 1
            return x.values, y.values
        else:
            raise StopIteration

    def generate(self):
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    data = self._next_data()
                self.data_queue.put(data)
            except StopIteration:
                self.stop()

    def get_data(self):
        """Pull a batch of data from the queue"""
        if self.num_workers > 0:
            return self.data_queue.get()
        else:
            return self._next_data()

    def restart(self):

        # reset the counter for how many items have been put in the queue
        self.data_counter = 0

        # first create a new stop_event and shuffle the indices
        self.stop_event = threading.Event()
        if self.shuffle:
            self.rstate.shuffle(self.sample_indices)
            self.sample_indices = list(int(idx) for idx in self.sample_indices)

        # start filling the queue
        if self.num_workers > 0:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers,
            )
            self.futures = [
                self.executor.submit(self.generate) for _ in range(self.num_workers)
            ]

    def stop(self):
        self.stop_event.set()
        self.data_queue.task_done()

    def shutdown(self):
        self.stop()
        if self.num_workers > 0:
            self.executor.shutdown()
