#!/bin/bash

if [ -f "workers.txt" ]; then
    docker logs -f "worker1"
else
    echo "No workers.txt file found."
fi