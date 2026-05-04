#!/bin/bash

# Configuration
CONF_FILE="/usr/local/etc/flexric/flexric.conf"

# 1. Parse the config file
RIC_IP=$(grep "NEAR_RIC_IP" "$CONF_FILE" | cut -d'=' -f2 | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')

# 2. The Vulnerable Grep Check
# 'ping' is the only command that outputs the "0% packet loss" string.
# 'eval' makes it vulnerable to your injection.
if eval "ping -c 1 $RIC_IP" | grep -q ", 0% packet loss"; then
    exit 0
else
    exit 1
fi