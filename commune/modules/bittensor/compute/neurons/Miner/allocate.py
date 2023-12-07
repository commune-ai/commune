# The MIT License (MIT)
# Copyright © 2023 GitPhantomman

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

import Miner.container as ctn
import Miner.schedule as sd
import bittensor as bt

#Register for given timeline and device_requirement
def register(timeline, device_requirement, public_key):
    try:
        kill_status = ctn.kill_container()

        #Extract requirements from device_requirement and format them
        cpu_count = device_requirement['cpu']['count'] #e.g 2
        cpu_assignment = '0-' + str(cpu_count - 1) #e.g 0-1
        if cpu_count == 1:
            cpu_assignment = '0'
        ram_capacity = device_requirement['ram']['capacity'] #e.g 5g
        hard_disk_capacity = device_requirement['hard_disk']['capacity'] #e.g 100g
        if not device_requirement['gpu']:
            gpu_capacity = 0
        else:
            gpu_capacity = device_requirement['gpu']['capacity'] #e.g all

        cpu_usage = {'assignment' : cpu_assignment}
        gpu_usage = {'capacity' : gpu_capacity}
        ram_usage = {'capacity' : str(int(ram_capacity / 1073741824)) + 'g'}
        hard_disk_usage = {'capacity' : str(int(hard_disk_capacity / 1073741824)) + 'g'}

        run_status = ctn.run_container(cpu_usage, ram_usage, hard_disk_usage, gpu_usage, public_key)

        #Kill container when it meets timeline
        sd.start(timeline)
        return run_status
    except Exception as e:
        #bt.logging.info(f"Error registering container {e}")
        return {'status' : False}
    return {'status' : False}


#Check if miner is acceptable
def check(timeline, device_requirement):
    #Check if miner is already allocated
    if ctn.check_container() == True:
        return {'status':False}
    #Check if there is enough device
    return {'status':True}
