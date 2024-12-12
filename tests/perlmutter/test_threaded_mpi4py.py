from mpi4py import MPI
import threading

def worker(thread_id):
    print(f"Thread {thread_id} is working.")

# Request thread support (using THREAD_MULTIPLE for full support)
required_level = MPI.THREAD_MULTIPLE
provided_level = MPI.Query_thread()

if provided_level < required_level:
    print("Requested threading level not available!")
else:
    print(f"Thread level supported: {provided_level}")

# Now, create threads
threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# Perform MPI operations after threads have finished
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Process {rank} is done with MPI operations.")

# Now, create threads
threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Process {rank} is done with thread operations.")
# Finalize MPI once at the end
MPI.Finalize()

