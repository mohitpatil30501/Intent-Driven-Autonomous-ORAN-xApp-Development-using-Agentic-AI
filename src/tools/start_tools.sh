#!/bin/bash

cd oriosearch
docker compose up -d
cd ../semantic_search
docker compose up -d --build