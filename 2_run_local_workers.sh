#!/bin/bash

# Here we also need access to the data
if [ -z "$( ls -A './data' )" ]; then
   echo "Data folder is empty. Can not continue."
   echo "Copy data as .zip file to the folder." 
   exit 1;
else
   echo "Found the data folder."
fi

# Remove existing workers.txt if it exists to start fresh
rm -f workers.txt

WORKER_URLS=""

for i in {1..10}
do
   PORT=$((9000 + i))
   # Map external port to the same internal port, assuming worker listens on the specified port
   docker run -d --name "worker$i" --cpus 2 --add-host=host.docker.internal:host-gateway -v ./data:/app/data -p "$PORT:$PORT" antworker --port "$PORT"
   
   if [ -z "$WORKER_URLS" ]; then
       WORKER_URLS="http://0.0.0.0:$PORT"
   else
       WORKER_URLS="$WORKER_URLS,http://0.0.0.0:$PORT"
   fi
done

echo "$WORKER_URLS" > workers.txt
