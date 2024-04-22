
import subprocess
import asyncio
from asyncio import sleep


def score_module(module1, module2):
    while True:
        subprocess.run(["c", "call", module1, "score_module", module2])
        subprocess.run(["c", "call", module1, "vote"])
        subprocess.run(["c", "set_weights"])
        asyncio.run(sleep(1))
        subprocess.run(["c", "call", module1, "votes"])

module_list = [
    "YOUR MODULES HERE"
]

for i in range(len(module_list)): 
    score_module(module_list[i], module_list[len(module_list)-1-i])
    if i == len(module_list) + 1:
        i=1

