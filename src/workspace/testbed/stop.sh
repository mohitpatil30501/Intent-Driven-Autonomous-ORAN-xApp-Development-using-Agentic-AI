#!/bin/bash

# Start the main application
echo "Stopping the core components"
cd core
docker compose stop

sleep 5

echo "Starting the nearrtric components"
cd ..
cd nearrtric
docker compose stop

sleep 5

echo "Starting the ran components"
cd ..
cd ran
docker compose stop

