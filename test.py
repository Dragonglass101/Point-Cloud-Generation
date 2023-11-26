import psutil
import GPUtil
import torch
import os
import time

# Record the start time
start_time = time.time()

def get_cpu_info():
    # Get CPU information using psutil
    cpu_info = {
        "CPU Cores": psutil.cpu_count(logical=False),
        "Logical CPUs": psutil.cpu_count(logical=True),
        "CPU Usage (%)": psutil.cpu_percent(interval=1),
    }
    return cpu_info

def clear_console():
    for _ in range(12):
        print("\033[F\033[K", end="")  # Clears the current line and moves the cursor up

def get_gpu_info():
    # Get GPU information using GPUtil
    try:
        gpu_info = {}
        gpu_list = GPUtil.getGPUs()
        for i, gpu in enumerate(gpu_list):
            gpu_info[f"GPU {i + 1} Name"] = gpu.name
            gpu_info[f"GPU {i + 1} Driver"] = gpu.driver
            gpu_info[f"GPU {i + 1} Memory Total (MB)"] = gpu.memoryTotal
            gpu_info[f"GPU {i + 1} Memory Free (MB)"] = gpu.memoryFree
            gpu_info[f"GPU {i + 1} Memory Used (MB)"] = gpu.memoryUsed
            gpu_info[f"GPU {i + 1} GPU Utilization (%)"] = gpu.load * 100  # Convert load to percentage

        return gpu_info

    except Exception as e:
        return {"Error": str(e)}

if __name__ == "__main__":

    first = False

    while True:
        cpu_info = get_cpu_info()
        gpu_info = get_gpu_info()
        if first:
            clear_console()
        else:
            first = True
        print("CPU Information:")
        for key, value in cpu_info.items():
            print(f"{key}: {value}")

        # if "Error" in gpu_info:
        #     print(f"GPU Information: {gpu_info['Error']}")
        # else:
        print("GPU Information:")
        for key, value in gpu_info.items():
            print(f"{key}: {value}")

        curr_time = time.time()
        elapsed_time = curr_time - start_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")

        time.sleep(1)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Cuda version: ", torch.version.cuda)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time: {elapsed_time:.2f} seconds")
