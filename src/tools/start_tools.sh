#!/bin/bash

cd oriosearch
docker compose up -d
cd ../semantic_search
docker compose up -d

echo "Starting the ran components"
cd ../deployer/testbed
./start.sh

echo "Starting Deployer Server..."
cd ..
nohup python3 deploy_server.py > deploy_server.log 2>&1 &
echo "All tools started."