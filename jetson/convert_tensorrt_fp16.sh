#!/bin/bash
# TensorRT FP16 Conversion Script for Jetson Orin Nano
# Recommended: Best balance between speed and accuracy

echo "========================================="
echo "🚀 TensorRT FP16 Conversion"
echo "========================================="

ONNX_FILE="./yolov8s_fp32.onnx"
ENGINE_FP16="./yolov8s_fp16.engine"

# Check if trtexec is available
if ! command -v /usr/src/tensorrt/bin/trtexec &> /dev/null; then
    echo "❌ trtexec not found. Install TensorRT on Jetson first."
    exit 1
fi

echo ""
echo "📦 Input:  $ONNX_FILE"
echo "📦 Output: $ENGINE_FP16"
echo "⚙️  Precision: FP16 (Half Precision)"
echo "🎯 Target: Jetson Orin Nano (40 TOPS)"
echo ""
echo "Converting... (this may take 5-10 minutes)"
echo ""

# Convert with FP16 precision
/usr/src/tensorrt/bin/trtexec \
    --onnx=$ONNX_FILE \
    --saveEngine=$ENGINE_FP16 \
    --fp16 \
    --workspace=4096 \
    --minShapes=images:1x3x800x800 \
    --optShapes=images:1x3x800x800 \
    --maxShapes=images:1x3x800x800 \
    --verbose \
    --noTF32 \
    --useSpinWait \
    --useCudaGraph \
    --separateProfileRun \
    --skipInference \
    --streams=1 \
    --avgRuns=10 \
    --duration=0

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ FP16 TensorRT Engine Created!"
    echo "========================================="
    ls -lh $ENGINE_FP16
    echo ""
    echo "📊 Expected Performance:"
    echo "   • Inference Speed: 30-40 FPS"
    echo "   • Latency: 25-33 ms"
    echo "   • Accuracy: ~99% of FP32"
    echo ""
    echo "🎉 Ready for deployment!"
else
    echo "❌ FP16 conversion failed!"
    exit 1
fi
