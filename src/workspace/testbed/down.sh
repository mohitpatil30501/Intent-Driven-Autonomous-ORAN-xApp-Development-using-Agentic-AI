#!/bin/bash

# Start the main application
echo "Starting the core components"
cd core
docker compose down -v

sleep 10

echo "Starting the nearrtric components"
cd ..
cd nearrtric
docker compose down -v

sleep 10

echo "Starting the ran components"
cd ..
cd ran
docker compose down -v

