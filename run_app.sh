#!/bin/bash

# Script to run the geospatial app in background with logging

# Activate virtual environment
source ../env/bin/activate

# Run the app in background and redirect output to logs.txt
echo "Starting geospatial app at $(date)" > logs.txt
python app_updated.py >> logs.txt 2>&1 &

# Get the process ID
PID=$!
echo "App started with PID: $PID"
echo "View logs with: tail -f logs.txt"
echo "Stop app with: kill $PID"

# Save PID to file for easy stopping
echo $PID > app.pid

echo "App is now running in background. Logs are being written to logs.txt"