#!/bin/bash
# Quick startup script for Jetson Orin Nano
# Run this after completing setup

echo "========================================"
echo "Road Anomaly Detection System"
echo "Jetson Orin Nano Quick Start"
echo "========================================"
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "❌ This script must run on Jetson Orin Nano"
    exit 1
fi

echo "1/6 Setting performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks
echo "   ✅ Performance mode: MAXN"

echo ""
echo "2/6 Checking GPS..."
if [ -e /dev/ttyUSB0 ]; then
    echo "   ✅ GPS device found: /dev/ttyUSB0"
    sudo systemctl restart gpsd
else
    echo "   ⚠️  GPS device not found, will use mock GPS"
fi

echo ""
echo "3/6 Checking camera..."
if [ -e /dev/video0 ]; then
    echo "   ✅ Camera found: /dev/video0"
else
    echo "   ⚠️  Camera not found, check connections"
fi

echo ""
echo "4/6 Activating Python environment..."
source ~/road_anomaly_env/bin/activate
echo "   ✅ Environment activated"

echo ""
echo "5/6 Checking model files..."
if [ -f ~/road_anomaly/models/*.pt ] || [ -f ~/road_anomaly/models/*.engine ]; then
    echo "   ✅ Model files found"
else
    echo "   ❌ Model files not found!"
    echo "      Please copy model files to ~/road_anomaly/models/"
    exit 1
fi

echo ""
echo "6/6 Starting detection system..."
echo ""
echo "========================================"
echo "System Status: READY"
echo "========================================"
echo ""
echo "To start the system, run:"
echo ""
echo "  python ~/road_anomaly/jetson/main_system.py \\"
echo "      --config ~/road_anomaly/config/deployment_config.json"
echo ""
echo "Or run in background:"
echo ""
echo "  nohup python ~/road_anomaly/jetson/main_system.py \\"
echo "      --config ~/road_anomaly/config/deployment_config.json \\"
echo "      > ~/road_anomaly/logs/system.log 2>&1 &"
echo ""
echo "To view logs:"
echo "  tail -f ~/road_anomaly/logs/system.log"
echo ""
echo "To check detections:"
echo "  sqlite3 ~/road_anomaly/detections.db 'SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10;'"
echo ""
echo "Press any key to start the system now, or Ctrl+C to exit..."
read -n 1 -s

echo ""
echo "Starting system..."
cd ~/road_anomaly
python jetson/main_system.py --config config/deployment_config.json
