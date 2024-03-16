#!/bin/bash

for port in $(seq 50000 50100); do
    pid=$(lsof -ti :$port)
    if [[ ! -z "$pid" ]]; then
        echo "Killing process $pid on port $port"
        kill -9 $pid
    fi
done