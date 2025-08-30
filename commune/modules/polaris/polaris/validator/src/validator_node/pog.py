import paramiko
import json
import re
import logging

logger = logging.getLogger("remote_access")

def parse_ngrok_ssh(ssh_string):
    """Parses an ngrok SSH string into components."""
    pattern = r"ssh (.*?)@(.*?) -p (\d+)"
    match = re.match(pattern, ssh_string)
    if not match:
        raise ValueError("Invalid SSH string format.")
    return match.group(1), match.group(2), int(match.group(3))

# def fetch_compute_specs(ssh_string, password):
#     """Fetches system specifications from a remote machine via SSH."""
#     username, hostname, port = parse_ngrok_ssh(ssh_string)

#     client = paramiko.SSHClient()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

#     try:
#         client.connect(hostname=hostname, port=port, username=username, password=password)

#         # Detect OS
#         stdin, stdout, stderr = client.exec_command("systeminfo | findstr /B /C:\"OS Name\" /C:\"OS Version\"")
#         os_info = stdout.read().decode().strip().lower()

#         is_windows = "microsoft" in os_info or "windows" in os_info
#         is_linux = not is_windows  # Assuming it's Linux if not Windows

#         if is_linux:
#             # Existing Linux code remains unchanged
#             stdin, stdout, stderr = client.exec_command("lscpu")
#             lscpu_output = stdout.read().decode()

#             cpu_specs = {}
#             for line in lscpu_output.splitlines():
#                 key, _, value = line.partition(":")
#                 key, value = key.strip(), value.strip()

#                 if key == "CPU op-mode(s)":
#                     cpu_specs["op_modes"] = value
#                 elif key == "Address sizes":
#                     cpu_specs["address_sizes"] = value
#                 elif key == "Byte Order":
#                     cpu_specs["byte_order"] = value
#                 elif key == "CPU(s)":
#                     cpu_specs["total_cpus"] = int(value)
#                 elif key == "On-line CPU(s) list":
#                     cpu_specs["online_cpus"] = value
#                 elif key == "Vendor ID":
#                     cpu_specs["vendor_id"] = value
#                 elif key == "Model name":
#                     cpu_specs["cpu_name"] = value
#                 elif key == "CPU family":
#                     cpu_specs["cpu_family"] = int(value)
#                 elif key == "Model":
#                     cpu_specs["model"] = int(value)
#                 elif key == "Thread(s) per core":
#                     cpu_specs["threads_per_core"] = int(value)
#                 elif key == "Core(s) per socket":
#                     cpu_specs["cores_per_socket"] = int(value)
#                 elif key == "Socket(s)":
#                     cpu_specs["sockets"] = int(value)
#                 elif key == "CPU max MHz":
#                     cpu_specs["cpu_max_mhz"] = float(value)
#                 elif key == "CPU min MHz":
#                     cpu_specs["cpu_min_mhz"] = float(value)

#             # Check for GPU (NVIDIA)
#             gpu_info = []
#             stdin, stdout, stderr = client.exec_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
#             nvidia_output = stdout.read().decode().strip()
#             if nvidia_output:
#                 for line in nvidia_output.splitlines():
#                     name, memory = line.split(',')
#                     gpu_info.append({
#                         "gpu_name": name.strip(),
#                         "memory_total": memory.strip()
#                     })

#             # Fetch RAM information
#             stdin, stdout, stderr = client.exec_command("free -h | grep Mem")
#             ram_output = stdout.read().decode().strip()
#             ram = ram_output.split()[1]  # Total memory

#             # Fetch storage information
#             stdin, stdout, stderr = client.exec_command("lsblk -o NAME,TYPE,SIZE,MOUNTPOINT | grep disk")
#             storage_output = stdout.read().decode().strip()
#             storage_lines = storage_output.splitlines()
#             storage_info = []
#             for line in storage_lines:
#                 parts = line.split()
#                 if len(parts) >= 3:
#                     storage_info.append({
#                         "name": parts[0],
#                         "type": "SSD" if "sd" in parts[0] else "HDD",
#                         "capacity": parts[2]
#                     })

#         elif is_windows:
#             # Fetch Windows-specific specifications
#             cpu_specs = {}
#             stdin, stdout, stderr = client.exec_command("wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed /format:list")
#             wmic_cpu_output = stdout.read().decode().strip()
#             for line in wmic_cpu_output.splitlines():
#                 key, _, value = line.partition("=")
#                 key, value = key.strip(), value.strip()

#                 if key == "Name":
#                     cpu_specs["cpu_name"] = value
#                 elif key == "NumberOfCores":
#                     cpu_specs["cores_per_socket"] = int(value)
#                 elif key == "NumberOfLogicalProcessors":
#                     cpu_specs["total_cpus"] = int(value)
#                 elif key == "MaxClockSpeed":
#                     cpu_specs["cpu_max_mhz"] = float(value)

#             gpu_info = []
#             stdin, stdout, stderr = client.exec_command("wmic path win32_videocontroller get Name,AdapterRAM /format:list")
#             wmic_gpu_output = stdout.read().decode().strip()
#             current_gpu = {}
#             for line in wmic_gpu_output.splitlines():
#                 key, _, value = line.partition("=")
#                 if key == "Name":
#                     current_gpu["gpu_name"] = value
#                 elif key == "AdapterRAM":
#                     current_gpu["memory_total"] = f"{int(value) / (1024 ** 2):.2f} MB"
#                     gpu_info.append(current_gpu)
#                     current_gpu = {}

#             stdin, stdout, stderr = client.exec_command("systeminfo | findstr /C:\"Total Physical Memory\"")
#             ram_output = stdout.read().decode().strip()
#             ram = ram_output.split(":")[-1].strip()

#             stdin, stdout, stderr = client.exec_command("wmic logicaldisk get Size,DeviceID /format:list")
#             storage_output = stdout.read().decode().strip()
#             storage_info = []
#             for line in storage_output.splitlines():
#                 key, _, value = line.partition("=")
#                 if key == "DeviceID":
#                     device_id = value
#                 elif key == "Size":
#                     storage_info.append({
#                         "name": device_id,
#                         "type": "HDD",
#                         "capacity": f"{int(value) / (1024 ** 3):.2f} GB"
#                     })

#         else:
#             raise ValueError("Unsupported OS detected.")

#         # Construct the response
#         compute_resource = {
#             "resource_type": "GPU" if gpu_info else "CPU",
#             "ram": ram,
#             "storage": storage_info[0] if storage_info else {},
#             "is_active": True,
#             "cpu_specs": cpu_specs,
#             "gpu_specs": gpu_info if gpu_info else None
#         }

#         return compute_resource

#     except Exception as e:
#         # Return inactive status if connection fails
#         return {
#             "resource_type": "Unknown",
#             "ram": "Unknown",
#             "storage": {},
#             "is_active": False,
#             "cpu_specs": {},
#             "gpu_specs": None,
#             "error": "Miner currently inactive. Error: " + str(e)
#         }

#     finally:
#         client.close()


def execute_remote_command(client, command):
    """Executes a command on the remote server via SSH."""
    try:
        stdin, stdout, stderr = client.exec_command(command)
        return stdout.read().decode().strip(), stderr.read().decode().strip()
    except Exception as e:
        logger.error(f"Failed to execute remote command: {command}. Error: {e}")
        return None, str(e)

def get_remote_os(client):
    """Detects the remote OS."""
    try:
        stdout, _ = execute_remote_command(client, "uname")
        if "Linux" in stdout:
            return "Linux"
        stdout, _ = execute_remote_command(client, "systeminfo | findstr /B /C:\"OS Name\"")
        if "Windows" in stdout:
            return "Windows"
    except Exception as e:
        logger.error(f"Failed to detect remote OS. Error: {e}")
    return "Unknown"

def get_remote_cpu_info(client, os_type):
    """Fetches CPU info from the remote machine."""
    if os_type == "Linux":
        stdout, _ = execute_remote_command(client, "lscpu")
        cpu_info = {}
        for line in stdout.splitlines():
            key, _, value = line.partition(":")
            cpu_info[key.strip()] = value.strip()
        return {
            "op_modes": cpu_info.get("CPU op-mode(s)"),
            "address_sizes": cpu_info.get("Address sizes"),
            "byte_order": cpu_info.get("Byte Order"),
            "total_cpus": int(cpu_info.get("CPU(s)", 0)),
            "online_cpus": cpu_info.get("On-line CPU(s) list", ""),
            "vendor_id": cpu_info.get("Vendor ID"),
            "cpu_name": cpu_info.get("Model name"),
            "cpu_family": int(cpu_info.get("CPU family", 0)),
            "model": int(cpu_info.get("Model", 0)),
            "threads_per_core": int(cpu_info.get("Thread(s) per core", 1)),
            "cores_per_socket": int(cpu_info.get("Core(s) per socket", 1)),
            "sockets": int(cpu_info.get("Socket(s)", 1)),
            "cpu_max_mhz": float(cpu_info.get("CPU max MHz", 0)),
            "cpu_min_mhz": float(cpu_info.get("CPU min MHz", 0)),
        }
    elif os_type == "Windows":
        cmd = """powershell -Command "Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,MaxClockSpeed,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json" """
        stdout, _ = execute_remote_command(client, cmd)
        cpu_info = json.loads(stdout)
        return {
            "op_modes": "32-bit, 64-bit",
            "address_sizes": "64 bits",
            "byte_order": "Little Endian",
            "total_cpus": cpu_info.get("NumberOfLogicalProcessors", 0),
            "online_cpus": str(list(range(cpu_info.get("NumberOfLogicalProcessors", 0)))),
            "vendor_id": cpu_info.get("Manufacturer"),
            "cpu_name": cpu_info.get("Name"),
            "cpu_family": None,
            "model": None,
            "threads_per_core": cpu_info.get("NumberOfLogicalProcessors", 0) // cpu_info.get("NumberOfCores", 1),
            "cores_per_socket": cpu_info.get("NumberOfCores", 0),
            "sockets": 1,
            "cpu_max_mhz": cpu_info.get("MaxClockSpeed", 0),
            "cpu_min_mhz": None,
        }
    return {}

def get_remote_gpu_info(client, os_type):
    """Fetches GPU info from the remote machine."""
    if os_type == "Linux":
        cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
        stdout, _ = execute_remote_command(client, cmd)
        gpu_info = []
        for line in stdout.splitlines():
            name, memory = line.split(",")
            gpu_info.append({
                "gpu_name": name.strip(),
                "memory_total": f"{float(memory.strip()) / 1024:.2f} GB",
            })
        return gpu_info
    elif os_type == "Windows":
        cmd = """powershell -Command "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json" """
        stdout, _ = execute_remote_command(client, cmd)
        gpu_info = json.loads(stdout)
        if not isinstance(gpu_info, list):
            gpu_info = [gpu_info]
        return [
            {
                "gpu_name": gpu.get("Name"),
                "memory_total": f"{int(gpu.get('AdapterRAM', 0)) / (1024**3):.2f} GB" if gpu.get("AdapterRAM") else "Unknown",
            }
            for gpu in gpu_info
        ]
    return []

def get_remote_ram_info(client, os_type):
    """Fetches RAM info from the remote machine."""
    if os_type == "Linux":
        stdout, _ = execute_remote_command(client, "free -h | grep Mem")
        return stdout.split()[1] if stdout else "Unknown"
    elif os_type == "Windows":
        cmd = """powershell -Command "Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json" """
        stdout, _ = execute_remote_command(client, cmd)
        ram_info = json.loads(stdout)
        return f"{int(ram_info.get('TotalPhysicalMemory', 0)) / (1024**3):.2f} GB"
    return "Unknown"

def get_remote_storage_info(client, os_type):
    """Fetches storage info from the remote machine."""
    if os_type == "Linux":
        cmd = "lsblk -o NAME,TYPE,SIZE | grep disk"
        stdout, _ = execute_remote_command(client, cmd)
        storage_info = []
        for line in stdout.splitlines():
            name, _, size = line.split()
            storage_info.append({"name": name, "type": "Disk", "capacity": size})
        return storage_info[0] if storage_info else {"name": "Unknown", "type": "Unknown", "capacity": "Unknown"}
    elif os_type == "Windows":
        cmd = """powershell -Command "Get-PhysicalDisk | Select-Object MediaType,Size | ConvertTo-Json" """
        stdout, _ = execute_remote_command(client, cmd)
        storage_info = json.loads(stdout)
        if not isinstance(storage_info, list):
            storage_info = [storage_info]
        primary_storage = storage_info[0]
        capacity_gb = int(primary_storage.get("Size", 0)) / (1024**3)
        return {"name": "Disk", "type": primary_storage.get("MediaType", "Unknown"), "capacity": f"{capacity_gb:.2f} GB"}
    return {"name": "Unknown", "type": "Unknown", "capacity": "Unknown"}

def fetch_compute_specs(ssh_string, password):
    """Fetches system specifications from a remote machine via SSH."""
    username, hostname, port = parse_ngrok_ssh(ssh_string)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname=hostname, port=port, username=username, password=password)
        os_type = get_remote_os(client)

        cpu_specs = get_remote_cpu_info(client, os_type)
        gpu_specs = get_remote_gpu_info(client, os_type)
        ram = get_remote_ram_info(client, os_type)
        storage = get_remote_storage_info(client, os_type)

        return {
            "resource_type": "GPU" if gpu_specs else "CPU",
            "ram": ram,
            "storage": storage,
            "is_active": True,
            "cpu_specs": cpu_specs,
            "gpu_specs": gpu_specs if gpu_specs else None,
        }
    except Exception as e:
        logger.error(f"Failed to fetch compute specs: {e}")
        return {
            "resource_type": "Unknown",
            "ram": "Unknown",
            "storage": {},
            "is_active": False,
            "cpu_specs": {},
            "gpu_specs": None,
        }
    finally:
        client.close()




def compare_compute_resources(new_resource, existing_resource):
    """Compare new compute resource specs with existing ones and calculate a score."""
    score = 0
    total_checks = 0

    # Compare CPU specs
    for key in ["op_modes", "address_sizes", "byte_order", "total_cpus", "online_cpus", "vendor_id", "cpu_name", "cpu_family", "model", "threads_per_core", "cores_per_socket", "sockets", "cpu_max_mhz", "cpu_min_mhz"]:
        total_checks += 1
        new_value = new_resource["cpu_specs"].get(key)
        existing_value = existing_resource["cpu_specs"].get(key)
        
        # Handle numeric comparisons safely
        try:
            if isinstance(new_value, str) and isinstance(existing_value, str):
                if new_value == existing_value:
                    score += 1
            elif new_value is not None and existing_value is not None:
                if float(new_value) == float(existing_value):
                    score += 1
        except ValueError:
            # Skip comparison if conversion to float/int fails
            pass


    # Compare RAM
    total_checks += 1
    if new_resource.get("ram") == existing_resource.get("ram"):
        score += 1

    # Compare storage
    total_checks += 1
    if new_resource.get("storage", {}).get("capacity") == existing_resource.get("storage", {}).get("capacity"):
        score += 1

    # Compare resource type (CPU/GPU)
    total_checks += 1
    if new_resource.get("resource_type") == existing_resource.get("resource_type"):
        score += 1

    # Final score
    comparison_result = {
        "score": score,
        "total_checks": total_checks,
        "percentage": (score / total_checks) * 100
    }

    return comparison_result


def compute_resource_score(resource):
    """
    Calculate a score for a compute resource (CPU or GPU) based on its specifications.

    Parameters:
    resource (dict or list): A dictionary containing compute resource details, or a list of such dictionaries.

    Returns:
    float or list: A score representing the performance of the resource, or a list of scores if a list is provided.
    """
    if isinstance(resource, list):
        # If the input is a list, calculate the score for each resource
        return [compute_resource_score(item) for item in resource]

    if not isinstance(resource, dict):
        raise TypeError("Expected 'resource' to be a dictionary or a list of dictionaries, but got type: {}".format(type(resource)))

    if "resource_type" not in resource:
        raise KeyError("The key 'resource_type' is missing from the resource dictionary.")

    score = 0
    weights = {
        "cpu": {
            "cores": 0.4,
            "threads_per_core": 0.1,
            "max_clock_speed": 0.3,
            "ram": 0.15,
            "storage_speed": 0.05
        },
        "gpu": {
            "vram": 0.5,
            "compute_cores": 0.3,
            "bandwidth": 0.2
        }
    }

    if resource["resource_type"] == "CPU":
        cpu_specs = resource.get("cpu_specs", {})
        ram = resource.get("ram", "0GB")
        storage = resource.get("storage", {})

        # Convert RAM and storage speed to numeric values
        try:
            ram = float(ram.replace("GB", ""))
        except ValueError:
            ram = 0  # Default to 0 if RAM is invalid

        storage_speed = storage.get("read_speed", "0MB/s")
        try:
            storage_speed = float(storage_speed.replace("MB/s", ""))
        except ValueError:
            storage_speed = 0  # Default to 0 if storage speed is invalid

        # Normalize CPU values for scoring
        cores_score = cpu_specs.get("total_cpus", 0) / 64  # Assuming max 64 cores
        threads_score = cpu_specs.get("threads_per_core", 0) / 2  # Assuming max 2 threads/core
        clock_speed_score = cpu_specs.get("cpu_max_mhz", 0) / 5000  # Assuming max 5 GHz
        ram_score = ram / 128  # Assuming max 128GB RAM
        storage_score = storage_speed / 1000  # Assuming max 1000MB/s

        # Weighted score for CPU
        score += (
            cores_score * weights["cpu"]["cores"] +
            threads_score * weights["cpu"]["threads_per_core"] +
            clock_speed_score * weights["cpu"]["max_clock_speed"] +
            ram_score * weights["cpu"]["ram"] +
            storage_score * weights["cpu"]["storage_speed"]
        )

    elif resource["resource_type"] == "GPU":
        gpu_specs = resource.get("gpu_specs", {})
        if isinstance(gpu_specs, list) and len(gpu_specs) > 0:
            gpu_specs = gpu_specs[0]  # Take the first GPU if there are multiple

        vram = gpu_specs.get("memory_total", "0GB")
        compute_cores = gpu_specs.get("compute_cores", 0)
        bandwidth = gpu_specs.get("bandwidth", "0GB/s")

        # Convert VRAM and bandwidth to numeric values
        try:
            vram = float(vram.replace("GB", ""))
        except ValueError:
            vram = 0  # Default to 0 if VRAM is invalid

        try:
            bandwidth = float(bandwidth.replace("GB/s", ""))
        except ValueError:
            bandwidth = 0  # Default to 0 if bandwidth is invalid

        # Normalize GPU values for scoring
        vram_score = vram / 48  # Assuming max 48GB VRAM
        compute_cores_score = compute_cores / 10000  # Assuming max 10k cores
        bandwidth_score = bandwidth / 1000  # Assuming max 1 TB/s

        # Weighted score for GPU
        score += (
            vram_score * weights["gpu"]["vram"] +
            compute_cores_score * weights["gpu"]["compute_cores"] +
            bandwidth_score * weights["gpu"]["bandwidth"]
        )

    else:
        raise ValueError(f"Unknown resource type: {resource['resource_type']}")

    return round(score, 3)


