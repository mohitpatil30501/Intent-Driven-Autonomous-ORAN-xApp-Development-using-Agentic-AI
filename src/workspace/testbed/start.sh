#!/bin/bash

# Start the main application
echo "Starting the monitoring components"
cd monitoring
docker compose up -d

sleep 10

echo "Starting the core components"
cd ..
cd core
docker compose up -d

sleep 10

echo "Starting the nearrtric components"
cd ..
cd nearrtric
docker compose up -d --build

sleep 10

echo "Starting the ran components"
cd ..
cd ran
docker compose up -d

