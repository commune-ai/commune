# The MIT License (MIT)
# Copyright © 2023 Crazydevlegend

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# Step 1: Import necessary libraries and modules
import psutil
import igpu
import json
import time
import subprocess
from cryptography.fernet import Fernet

secret_key = b't4LXlXijLYmqvswYdo4MpA-fcTLN0fvYo-Qa0gq0wlc='#key
# Return the detailed information of cpu
def get_cpu_info():
    try:
        # Get the number of physical CPU cores
        physical_cores = psutil.cpu_count(logical=False)

        # Get CPU frequency
        cpu_frequency = psutil.cpu_freq()

        info = {}
        info["count"] = physical_cores
        info["frequency"] = cpu_frequency.current

        return info
    except Exception as e:
        return {}

# Return the detailed information of gpu
def get_gpu_info():
    try:
        # Count of existing gpus
        gpu_count = igpu.count_devices()

        # Get the detailed information for each gpu (name, capacity)
        gpu_details = []
        capacity = 0
        for i in range(gpu_count):
            gpu = igpu.get_device(i)
            gpu_details.append({"name" : gpu.name, "capacity" : gpu.memory.total, "utilization" : gpu.utilization.gpu})
            capacity += gpu.memory.total
        
        info = {"count":gpu_count, "capacity": capacity, "details": gpu_details}

        # Measure speed
        if gpu_count:
            # Run nvidia-smi command to get GPU information
            result = subprocess.run(['nvidia-smi', '--query-gpu=clocks.gr,clocks.mem', '--format=csv,noheader'], stdout=subprocess.PIPE)
            gpu_speed_info = result.stdout.decode().strip().split(',')

            graphics_speed = int(gpu_speed_info[0].split()[0])
            memory_speed = int(gpu_speed_info[1].split()[0])
            
            info['graphics_speed'] = graphics_speed
            info['memory_speed'] = memory_speed
        return info
            
    except Exception as e:
        return {}

# Return the detailed information of hard disk
def get_hard_disk_info():
    try:

        # Capacity-related information
        usage = psutil.disk_usage("/")
        info = {"total": usage.total, "free": usage.free, "used": usage.used}

        partition_info = []
        partitions = psutil.disk_partitions(all=True)
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.device)
                partition_info.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free
                })
            except Exception as e:
                continue

        # Measure write speed
        size_mb = 100
        file_size_bytes = size_mb * 1024 * 1024

        data = bytearray(file_size_bytes)
        file_path = 'test_speed_file.dat'

        # Write data to a file
        with open(file_path, 'wb') as file:
            start_time = time.time()
            file.write(data)
            end_time = time.time()

        write_speed = size_mb / (end_time - start_time)

        # Measure read speed
        with open(file_path, 'rb') as file:
            start_time = time.time()
            _ = file.read()
            end_time = time.time()

        read_speed = size_mb / (end_time - start_time)

        info['write_speed'] = write_speed
        info['read_speed'] = read_speed
        
        return info
    except Exception as e:
        return {}
    
# Return the detailed information of ram
def get_ram_info():
    try:
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()

        info = {
            "total": virtual_memory.total,
            "available": virtual_memory.available,
            "used": virtual_memory.used,
            "free": virtual_memory.free,
            "swap_total": swap_memory.total,
            "swap_used": swap_memory.used,
            "swap_free": swap_memory.free
        }

        # Measure read speed
        size_mb = 100
        data = bytearray(size_mb * 1024 * 1024)  # Create a byte array of the specified size (in MB)
        start_time = time.time()
        _ = data[0:size_mb * 1024 * 1024]  # Read the entire array from RAM
        end_time = time.time()
        read_time = end_time - start_time
        read_speed = size_mb / read_time

        # Measure write speed
        data = bytearray(size_mb * 1024 * 1024)  # Create a byte array of the specified size (in MB)
        start_time = time.time()
        data[0:size_mb * 1024 * 1024] = b'\x00'  # Write zeros to the entire array in RAM
        end_time = time.time()
        write_time = end_time - start_time
        write_speed = size_mb / write_time

        info['read_speed'] = read_speed # unit : MB/s
        info['write_speed'] = write_speed # unit : MB/s

        return info
    except Exception as e:
        return {}

def get_perf_info():
    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    hard_disk_info = get_hard_disk_info()
    ram_info = get_ram_info()

    perf_info = {'cpu' : cpu_info, 'gpu' : gpu_info, 'hard_disk' : hard_disk_info, 'ram' : ram_info}
    perf_str = json.dumps(perf_info)

    cipher_suite = Fernet(secret_key)

    encoded_str = cipher_suite.encrypt(perf_str.encode())

    return encoded_str

if __name__ == "__main__":
    print(f"{get_perf_info()}")
