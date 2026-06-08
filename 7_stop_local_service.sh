#!/bin/bash
if [ -f "controller.txt" ]; then
    pkill -f controller.service
    rm -f controller.txt

    for i in {1..10}
    do
        echo "Stopping and removing worker$i..."
        docker kill "worker$i"
        docker rm "worker$i"
    done

    # Cleanup the workers.txt file created by the start script
    rm -f workers.txt

    echo "All local workers stopped and removed."
fi