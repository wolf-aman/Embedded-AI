#!/bin/bash
# Benchmark TensorRT Engines on Jetson Orin Nano

echo "========================================="
echo "📊 TensorRT Engine Benchmark"
echo "========================================="
echo ""

# Set max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

echo "🔥 Performance mode enabled"
echo ""

# Benchmark FP16
if [ -f "yolov8s_fp16.engine" ]; then
    echo "Testing FP16 engine..."
    /usr/src/tensorrt/bin/trtexec \
        --loadEngine=yolov8s_fp16.engine \
        --iterations=100 \
        --avgRuns=10
    echo ""
fi

# Benchmark INT8
if [ -f "yolov8s_int8.engine" ]; then
    echo "Testing INT8 engine..."
    /usr/src/tensorrt/bin/trtexec \
        --loadEngine=yolov8s_int8.engine \
        --iterations=100 \
        --avgRuns=10
    echo ""
fi

echo "✅ Benchmark complete!"
