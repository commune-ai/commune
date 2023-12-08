
<div align="center">

# **Compute Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

### The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)

</div>

---

This repo contains all the necessary files and functions to define Bittensor's Compute Subnet. You can try running miners on netuid 15 in Bittensor's test network.

# Introduction
This repository is a compute-composable subnet. This subnet has integrated various cloud platforms (e.g., Runpod, Lambda, AWS) into a cohesive unit, enabling higher-level cloud platforms to offer seamless compute composability across different underlying platforms. With the proliferation of cloud platforms, there's a need for a subnet that can seamlessly integrate these platforms, allowing for efficient resource sharing and allocation. This compute-composable subnet will enable nodes to contribute computational power, with validators ensuring the integrity and efficiency of the shared resources.

- `compute/protocol.py`: The file where the wire-protocol used by miners and validators is defined.
- `neurons/miner.py`: This script which defines the miner's behavior, i.e., how the miner responds to requests from validators.
- `neurons/validator.py`: This script which defines the validator's behavior, i.e., how the validator requests information from miners and determines scores.

---

# Installation
This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.

## Install Bittensor
```bash
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/opentensor/bittensor/master/scripts/install.sh)"
```
## Install Dependencies
```bash
git clone https://github.com/neuralinternet/Compute-Subnet.git
cd Compute-Subnet
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```
## Setup Docker for Miner
To run a miner, you must [install](https://docs.docker.com/engine/install/ubuntu) and start the docker service by running `sudo systemctl start docker` and `sudo apt install at`.

</div>

---
# Running a Miner / Validator
Prior to running a miner or validator, you must [create a wallet](https://github.com/opentensor/docs/blob/main/reference/btcli.md) and [register the wallet to a netuid](https://github.com/opentensor/docs/blob/main/subnetworks/registration.md). Once you have done so, you can run the miner and validator with the following commands.

## Running Miner

Miners contribute processing resources, notably GPU (Graphics Processing Unit) and CPU (Central Processing Unit) instances, to facilitate optimal performance in essential GPU and CPU-based computing tasks. The system operates on a performance-based reward mechanism, where miners are incentivized through a tiered reward structure correlated to the processing capability of their hardware. High-performance devices are eligible for increased compensation, reflecting their greater contribution to the network's computational throughput. Emphasizing the integration of GPU instances is critical due to their superior computational power, particularly in tasks demanding parallel processing capabilities. Consequently, miners utilizing GPU instances are positioned to receive substantially higher rewards compared to their CPU counterparts, in alignment with the greater processing power and efficiency GPUs bring to the network.

A key aspect of the miners' contribution is the management of resource reservations. Miners have the autonomy to set specific timelines for each reservation of their computational resources. This timeline dictates the duration for which the resources are allocated to a particular task or user. Once the set timeline reaches its conclusion, the reservation automatically expires, thereby freeing up the resources for subsequent allocations. This mechanism ensures a dynamic and efficient distribution of computational power, catering to varying demands within the network.

```bash
# To run the miner
cd neurons
python -m miner.py 
    --netuid <your netuid>  # The subnet id you want to connect to
    --subtensor.network <your chain url>  # blockchain endpoint you want to connect
    --wallet.name <your miner wallet> # name of your wallet
    --wallet.hotkey <your miner hotkey> # hotkey name of your wallet
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
```

## Running Validator

Validators hold the critical responsibility of rigorously assessing and verifying the computational capabilities of miners. This multifaceted evaluation process commences with validators requesting miners to provide comprehensive performance data, which includes not only processing speeds and efficiencies but also critical metrics like Random Access Memory (RAM) capacity and disk space availability.

The inclusion of RAM and disk space measurements is vital, as these components significantly impact the overall performance and reliability of the miners' hardware. RAM capacity influences the ability to handle large or multiple tasks simultaneously, while adequate disk space ensures sufficient storage.

Following the receipt of this detailed hardware and performance information, validators proceed to test the miners' computational integrity. This is achieved by presenting them with complex hashing challenges, designed to evaluate the processing power and reliability of the miners' systems. Validators adjust the difficulty of these problems based on the comprehensive performance profile of each miner, including their RAM and disk space metrics.

In addition to measuring the time taken by miners to resolve these problems, validators meticulously verify the accuracy of the responses. This thorough examination of both speed and precision, complemented by the assessment of RAM and disk space utilization, forms the crux of the evaluation process.

Based on this extensive analysis, validators update the miners' scores, reflecting a holistic view of their computational capacity, efficiency, and hardware quality. This score then determines the miner's weight within the network, directly influencing their potential rewards and standing.
This scoring process, implemented through a Python script, considers various factors including CPU, GPU, hard disk, and RAM performance. The script's structure and logic are outlined below:
1. **Score Calculation Function:**
   - The `score` function aggregates performance data from different hardware components.
   - It calculates individual scores for CPU, GPU, hard disk, and RAM.
   - These scores are then weighted and summed to produce a total score.
   - A registration bonus is applied if the miner is registered, enhancing the total score.

2. **Component-Specific Scoring Functions:**
   - `get_cpu_score`, `get_gpu_score`, `get_hard_disk_score`, and `get_ram_score` are functions dedicated to calculating scores for each respective hardware component.
   - These functions consider the count, frequency, capacity, and speed of each component, applying a specific level value for normalization.
   - The scores are derived based on the efficiency and capacity of the hardware.

3. **Weight Assignment:**
   - The script defines weights for each hardware component's score, signifying their importance in the overall performance.
   - GPU has the highest weight (0.55), reflecting its significance in mining operations.
   - CPU, hard disk, and RAM have lower weights (0.2, 0.1, and 0.15, respectively).

4. **Registration Check:**
   - The `check_if_registered` function verifies if a miner is registered using an external API (`wandb.Api()`).
   - Registered miners receive a bonus to their total score, incentivizing official registration in the network.

5. **Score Aggregation:**
   - The individual scores are combined into a numpy array, `score_list`.
   - The weights are also arranged in an array, `weight_list`.
   - The final score is calculated using a dot product of these arrays, multiplied by 10, and then adjusted with the registration bonus.
  
6. **Handling Multiple CPUs/GPUs:**
   - The scoring functions for CPUs (`get_cpu_score`) and GPUs (`get_gpu_score`) are designed to process data that can represent multiple units.
   - For CPUs, the `count` variable in `cpu_info` represents the number of CPU cores or units available. The score is calculated based on the cumulative capability, taking into account the total count and frequency of all CPU cores.
   - For GPUs, similar logic applies. The script can handle data representing multiple GPUs, calculating a cumulative score based on their collective capacity and speed.

7. **CPU Scoring (Multiple CPUs):**
   - The CPU score is computed by multiplying the total number of CPU cores (`count`) by their average frequency (`frequency`), normalized against a predefined level.
   - This approach ensures that configurations with multiple CPUs are appropriately rewarded for their increased processing power.

8. **GPU Scoring (Multiple GPUs):**
   - The GPU score is calculated by considering the total capacity (`capacity`) and the average speed (average of `graphics_speed` and `memory_speed`) of all GPUs in the system.
   - The score reflects the aggregate performance capabilities of all GPUs, normalized against a set level.
   - This method effectively captures the enhanced computational power provided by multiple GPU setups.

9. **Aggregated Performance Assessment:**
   - The final score calculation in the `score` function integrates the individual scores from CPU, GPU, hard disk, and RAM.
   - This integration allows the scoring system to holistically assess the collective performance of all hardware components, including scenarios with multiple CPUs and GPUs.

10. **Implications for Miners:**
   - Miners with multiple GPUs and/or CPUs stand to gain a higher score due to the cumulative calculation of their hardware's capabilities.
   - This approach incentivizes miners to enhance their hardware setup with additional CPUs and GPUs, thereby contributing more processing power to the network.

The weight assignments are as follows:
- **GPU Weight:** 0.55
- **CPU Weight:** 0.2
- **Hard Disk Weight:** 0.1
- **RAM Weight:** 0.15

### Example 1: Miner A's Hardware Scores and Weighted Total

1. **CPU Score:** Calculated as `(2 cores * 3.0 GHz) / 1024 / 50`.
2. **GPU Score:** Calculated as `(8 GB * (1 GHz + 1 GHz) / 2) / 200000`.
3. **Hard Disk Score:** Calculated as `(500 GB * (100 MB/s + 100 MB/s) / 2) / 1000000`.
4. **RAM Score:** Calculated as `(16 GB * 2 GB/s) / 200000`.

Now, applying the weights:

- Total Score = (CPU Score × 0.2) + (GPU Score × 0.55) + (Hard Disk Score × 0.1) + (RAM Score × 0.15)
- If registered, add a registration bonus.

### Example 2: Miner B's Hardware Scores and Weighted Total

1. **CPU Score:** Calculated as `(4 cores * 2.5 GHz) / 1024 / 50`.
2. **GPU Score:** Calculated as `((6 GB + 6 GB) * (1.5 GHz + 1.2 GHz) / 2) / 200000`.
3. **Hard Disk Score:** Calculated as `(1 TB * (200 MB/s + 150 MB/s) / 2) / 1000000`.
4. **RAM Score:** Calculated as `(32 GB * 3 GB/s) / 200000`.

Applying the weights:

- Total Score = (CPU Score × 0.2) + (GPU Score × 0.55) + (Hard Disk Score × 0.1) + (RAM Score × 0.15)
- Since Miner B is not registered, no registration bonus is added.

### Impact of Weights on Total Score

- The GPU has the highest weight (0.55), signifying its paramount importance in mining tasks, which are often GPU-intensive. Miners with powerful GPUs will thus receive a significantly higher portion of their total score from the GPU component.
- The CPU, while important, has a lower weight (0.2), reflecting its lesser but still vital role in the mining process.
- The hard disk and RAM have the lowest weights (0.1 and 0.15, respectively), indicating their supportive but less critical roles compared to GPU and CPU.

It is important to note that the role of validators, in contrast to miners, does not require the integration of GPU instances. Their function revolves around data integrity and accuracy verification, involving relatively modest network traffic and lower computational demands. As a result, their hardware requirements are less intensive, focusing more on stability and reliability rather than high-performance computation.

```bash
# To run the validator
cd neurons
python -m validator.py 
    --netuid <your netuid> # The subnet id you want to connect to
    --subtensor.network <your chain url> # blockchain endpoint you want to connect
    --wallet.name <your validator wallet>  # name of your wallet
    --wallet.hotkey <your validator hotkey> # hotkey name of your wallet
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
```

## Resource Allocation Mechanism 

The allocation mechanism within subnet 27 is designed to optimize the utilization of computational resources effectively. Key aspects of this mechanism include:

1. **Resource Requirement Analysis:** The mechanism begins by analyzing the specific resource requirements of each task, including CPU, GPU, memory, and storage needs.

2. **Miner Selection:** Based on the analysis, the mechanism selects suitable miners that meet the resource requirements. This selection process considers the current availability, performance history, and network weights of the miners.

3. **Dynamic Allocation:** The allocation of tasks to miners is dynamic, allowing for real-time adjustments based on changing network conditions and miner performance.

4. **Efficiency Optimization:** The mechanism aims to maximize network efficiency by matching the most suitable miners to each task, ensuring optimal use of the network's computational power.

5. **Load Balancing:** It also incorporates load balancing strategies to prevent overburdening individual miners, thereby maintaining a healthy and sustainable network ecosystem.

Through these functionalities, the allocation mechanism ensures that computational resources are utilized efficiently and effectively, contributing to the overall robustness and performance of the network.

Validators can send requests to reserve access to resources from miners by specifying the specs manually in the in `register.py` and running this script: https://github.com/neuralinternet/Compute-Subnet/blob/main/neurons/register.py for example:
```{'cpu':{'count':1}, 'gpu':{'count':1}, 'hard_disk':{'capacity':10737418240}, 'ram':{'capacity':1073741824}}```

</div>

---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Neural Internet

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
```
