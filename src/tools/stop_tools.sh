#!/bin/bash

cd oriosearch
docker compose stop
cd ../semantic_search
docker compose stop
cd ../structural_rag
docker compose stop