#!/bin/bash

echo "Deleting the ran components"
cd ran
docker compose down -v

sleep 5

echo "Starting the nearrtric components"
cd ..
cd nearrtric
docker compose down -v

sleep 5

echo "Starting the core components"
cd ..
cd core
docker compose down -v
