#!/bin/bash

cd oriosearch
docker compose stop
cd ../semantic_search
docker compose stop

cd ../deployer/testbed
./down.sh

echo "Stopping Deployer Server..."
pkill -f deploy_server.py
echo "All tools stopped."