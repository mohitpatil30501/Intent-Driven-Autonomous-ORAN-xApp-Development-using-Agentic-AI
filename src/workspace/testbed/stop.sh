#!/bin/bash

# Start the main application
echo "Starting the monitoring components"
cd monitoring
docker compose stop

sleep 10

echo "Starting the core components"
cd ..
cd core
docker compose stop

sleep 10

echo "Starting the nearrtric components"
cd ..
cd nearrtric
docker compose stop

sleep 10

echo "Starting the ran components"
cd ..
cd ran
docker compose stop

