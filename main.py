from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank() #Rank of this process
total = comm.Get_size() # Total number of processes

print rank,total

