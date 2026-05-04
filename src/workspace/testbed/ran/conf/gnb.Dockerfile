# Stage 1: Builder Stage - Includes build tools and OAI source
FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# Install build dependencies for OAI gNB
# This list needs to be comprehensive for building OAI from source.
# Refer to OAI's official documentation for the full list.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \ 
    git \           
    build-essential \
    cmake \
    ninja-build \
    libconfig-dev \
    libsctp-dev \
    libfftw3-dev \
    libboost-program-options-dev \
    libboost-thread-dev \
    python3 \
    net-tools \
    gcc-12 \
    g++-12 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Clone OAI source code
# You can specify a particular branch or tag if needed (e.g., v2.2.0)
# Using develop for latest, but replace with your target version/tag
WORKDIR /opt/oai-gnb
RUN git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
WORKDIR /opt/oai-gnb/openairinterface5g
# RUN git checkout v2.2.0 # Example: Checkout a specific tag

COPY fill_rnd_data_slice.c /tmp/fill_rnd_data_slice.c
COPY ran_func_slice.c /tmp/ran_func_slice.c

RUN cd openair2/E2AP/flexric && \
    git clone https://gitlab.eurecom.fr/mosaic5g/flexric.git . && \
    cd test/rnd && \
    rm -rf fill_rnd_data_slice.c && \
    cp /tmp/fill_rnd_data_slice.c . && \
    cd /opt/oai-gnb/openairinterface5g/openair2/E2AP/RAN_FUNCTION/CUSTOMIZED && \
    rm -rf ran_func_slice.c && \
    cp /tmp/ran_func_slice.c .


ARG E2AP_VERSION=E2AP_V3
ARG KPM_VERSION=KPM_V3_00


WORKDIR /opt/oai-gnb/openairinterface5g
# Source environment and build OAI gNB
# The -I flag installs some additional dependencies if missing.
# We aim to build nr-softmodem and its required libraries.
RUN cd cmake_targets && \
    ./build_oai --ninja -I && \
    ./build_oai --ninja --gNB -w SIMU -c --build-e2 --cmake-opt -DKPM_VERSION=$KPM_VERSION --cmake-opt -DE2AP_VERSION=$E2AP_VERSION && \
    cd ran_build/build && \
    ninja nr-softmodem dfts ldpc params_libconfig coding rfsimulator

WORKDIR /opt/oai-gnb/openairinterface5g
RUN mkdir openair2/E2AP/flexric/build && \
    cd openair2/E2AP/flexric/build && \
    cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
          -DKPM_VERSION=$KPM_VERSION \
          -DE2AP_VERSION=$E2AP_VERSION .. && \
    ninja && \
    ninja install

# Create directories
WORKDIR /opt/oai-gnb
RUN mkdir -p ./etc
RUN chmod +x /opt/oai-gnb/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem

# Set LD_LIBRARY_PATH to include OAI's custom libraries
# ENV LD_LIBRARY_PATH=/opt/oai-gnb/lib:${LD_LIBRARY_PATH}

# Config file will be mounted via docker-compose
EXPOSE 38412/sctp
EXPOSE 2152/udp

# ENTRYPOINT and CMD will be set in docker-compose
# ENTRYPOINT ["/opt/oai-gnb/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem"]
# CMD ["-O", "/opt/oai-gnb/etc/gnb.conf"]