import torch
import torch.distributed as dist
import os

def setup_distributed(backend="nccl", init_method="env://", world_size=1, rank=0):
    """
    Initialize the distributed environment.
    """
    if dist.is_initialized():
        print("Distributed is already initialized.")
        return

    # Print debug information
    print(f"Initializing distributed: backend={backend}, init_method={init_method}, world_size={world_size}, rank={rank}")
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

def main():
    # Set the environment variables for MASTER_ADDR and MASTER_PORT
    os.environ["MASTER_ADDR"] = "10.140.24.59"
    os.environ["MASTER_PORT"] = "29540"  # You can change this port if needed

    # Number of GPUs available
    world_size = torch.cuda.device_count()

    if world_size < 1:
        raise RuntimeError("No GPUs available. Please check your environment.")

    # Launch distributed workers
    print(f"Launching distributed test with {world_size} GPUs...")

    # Use torch.multiprocessing for multi-GPU testing
    torch.multiprocessing.spawn(
        worker, args=(world_size,), nprocs=world_size, join=True
    )

def worker(rank, world_size):
    """
    A single distributed worker process.
    """
    print(f"Worker {rank}/{world_size} initializing...")
    setup_distributed(backend="nccl", world_size=world_size, rank=rank)

    # Print worker details
    print(f"Worker {rank} is running on device {torch.cuda.current_device()}")

    # Synchronize all workers
    dist.barrier()
    print(f"Worker {rank} has completed synchronization.")

    # Clean up distributed environment
    dist.destroy_process_group()
    print(f"Worker {rank} has exited.")

if __name__ == "__main__":
    main()
