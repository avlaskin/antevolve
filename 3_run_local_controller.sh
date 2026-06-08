#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export EVOWORKERS=$(cat workers.txt)
export CONTROLLER_API_KEY="experiment"

echo "Running controller with workers: $EVOWORKERS"
echo "Storage path: /tmp/lmc/data"
echo "Ensuring storage directory exists: /tmp/lmc/data"
mkdir -p /tmp/lmc/data
nohup python3 -m antevolve.controller.service \
    --storage-mode local \
    --local-storage-path /tmp/lmc/data > controller.log 2>&1 &
echo "localhost:8989" > controller.txt
