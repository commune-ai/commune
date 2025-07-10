#!/bin/bash

PORT=$1

if [ -z "$PORT" ]; then
  echo "Usage: $0 <port>"
  exit 1
fi

# Get PID for the port
PID=$(lsof -i :$PORT | awk 'NR==2 {print $2}')

if [ -z "$PID" ]; then
  echo "No process found on port $PORT"
  exit 1
fi

echo "Monitoring memory for PID $PID on port $PORT..."
echo -e "Timestamp\t\tPID\tMEM\tRSIZE\tVSIZE"

# Live monitor loop
while true; do
  ts=$(date "+%Y-%m-%d %H:%M:%S")
  top -l 1 -pid $PID -stats pid,mem,rsize,vsize | grep "^ *$PID" | \
    awk -v ts="$ts" '{printf "%s\t%s\t%s\t%s\t%s\n", ts, $1, $2, $3, $4}'
  sleep 1
done
