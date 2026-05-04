#!/bin/bash

# Start the main application
echo "Starting the core components"
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

