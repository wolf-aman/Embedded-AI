"""
TensorRT Inference Engine for Jetson Orin Nano
Real-time road anomaly detection with optimized performance
Optimized for low memory usage on embedded devices
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import time
import json
import gc
from memory_monitor import MemoryMonitor


class TensorRTInferenceEngine:
    """
    High-performance TensorRT inference for Jetson Orin Nano
    """
    
    def __init__(self, engine_path, config_path=None, enable_memory_monitor=True):
        """
        Args:
            engine_path: Path to TensorRT engine file (.engine)
            config_path: Path to deployment config JSON
            enable_memory_monitor: Enable memory monitoring for Jetson
        """
        self.engine_path = Path(engine_path)
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(enable_logging=enable_memory_monitor) if enable_memory_monitor else None
        if self.memory_monitor:
            self.memory_monitor.measure("TensorRT initialization start")
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()
        
        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
        
        # Memory buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        
        # Model info
        self.input_shape = None
        self.output_shapes = []
        
        # Performance tracking
        self.inference_times = []
        self.frame_count = 0
        
        # Load engine
        self._load_engine()
        self._allocate_buffers()
        
        if self.memory_monitor:
            self.memory_monitor.measure("TensorRT engine loaded")
        
        print(f"✅ TensorRT engine loaded: {self.engine_path}")
    
    def _default_config(self):
        """Default configuration"""
        return {
            'model': {
                'confidence_threshold': 0.35,
                'iou_threshold': 0.45,
                'num_classes': 4,
                'class_names': ['Pothole', 'Alligator Crack', 'Longitudinal Crack', 'Other Damage']
            },
            'max_memory_mb': 3000  # Memory threshold for Jetson Nano 4GB
        }
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        
        print(f"Loading TensorRT engine from {self.engine_path}...")
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        print(f"Engine loaded successfully")
        print(f"Number of bindings: {self.engine.num_bindings}")
    
    def _allocate_buffers(self):
        """Allocate memory buffers for inputs and outputs"""
        
        self.stream = cuda.Stream()
        
        for binding in range(self.engine.num_bindings):
            binding_shape = self.engine.get_binding_shape(binding)
            binding_size = trt.volume(binding_shape)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(binding_size, binding_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            # Store input/output info
            if self.engine.binding_is_input(binding):
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': binding_shape,
                    'dtype': binding_dtype
                })
                self.input_shape = binding_shape
                print(f"Input shape: {binding_shape}")
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': binding_shape,
                    'dtype': binding_dtype
                })
                self.output_shapes.append(binding_shape)
                print(f"Output shape: {binding_shape}")
    
    def preprocess(self, image):
        """
        Preprocess image for inference
        OPTIMIZED: Resize early, normalize once (Best Practices #2 and #3)
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image tensor
        """
        # Get input dimensions
        _, c, h, w = self.input_shape
        
        # Resize early to save memory (Best Practice #2)
        img_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize once (Best Practice #3) to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        # Ensure contiguous array
        img_contiguous = np.ascontiguousarray(img_batch)
        
        return img_contiguous
    
    def infer(self, image):
        """
        Run inference on an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections [class_id, confidence, x1, y1, x2, y2]
        """
        # Measure memory periodically
        if self.memory_monitor and self.frame_count % 100 == 0:
            self.memory_monitor.measure(f"TensorRT Frame {self.frame_count}")
            self.memory_monitor.check_memory_threshold(
                threshold_mb=self.config.get('max_memory_mb', 3000)
            )
        
        # Start timing
        start_time = time.time()
        
        # Preprocess (resize early to save memory)
        input_tensor = self.preprocess(image)
        
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Copy output from device
        for output in self.outputs:
            cuda.memcpy_dtoh_async(
                output['host'],
                output['device'],
                self.stream
            )
        
        # Synchronize
        self.stream.synchronize()
        
        # Get outputs
        output_data = [output['host'].reshape(output['shape']) for output in self.outputs]
        
        # Post-process
        detections = self.postprocess(output_data, image.shape)
        
        # Track performance
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        # Periodic memory cleanup
        if self.frame_count % 50 == 0:
            gc.collect()
        
        return detections
    
    def postprocess(self, outputs, original_shape):
        """
        Post-process model outputs to get detections
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (H, W, C)
            
        Returns:
            List of detections [class_id, confidence, x1, y1, x2, y2]
        """
        # YOLOv8 output format: [batch, num_predictions, 4+num_classes]
        # where first 4 are [x_center, y_center, width, height]
        
        predictions = outputs[0][0]  # Remove batch dimension
        
        detections = []
        conf_threshold = self.config['model']['confidence_threshold']
        
        # Get original dimensions
        orig_h, orig_w = original_shape[:2]
        _, _, input_h, input_w = self.input_shape
        
        # Scale factors
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        
        for pred in predictions:
            # Extract bbox and class scores
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]
            
            # Get best class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence < conf_threshold:
                continue
            
            # Convert to x1, y1, x2, y2
            x1 = int((x_center - width / 2) * scale_x)
            y1 = int((y_center - height / 2) * scale_y)
            x2 = int((x_center + width / 2) * scale_x)
            y2 = int((y_center + height / 2) * scale_y)
            
            # Clip to image boundaries
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            detections.append({
                'class_id': int(class_id),
                'class_name': self.config['model']['class_names'][int(class_id)],
                'confidence': float(confidence),
                'bbox': [x1, y1, x2, y2]
            })
        
        # Apply NMS
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections):
        """Apply Non-Maximum Suppression"""
        
        if len(detections) == 0:
            return []
        
        iou_threshold = self.config['model']['iou_threshold']
        
        # Convert to format for NMS
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # Apply NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.config['model']['confidence_threshold'],
            iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # Filter detections
        filtered_detections = [detections[i] for i in indices.flatten()]
        
        return filtered_detections
    
    def get_avg_inference_time(self):
        """Get average inference time"""
        if not self.inference_times:
            return 0
        return np.mean(self.inference_times[-100:])  # Last 100 frames
    
    def get_fps(self):
        """Get current FPS"""
        avg_time = self.get_avg_inference_time()
        if avg_time == 0:
            return 0
        return 1000.0 / avg_time


def visualize_detections(image, detections, show_fps=True, fps=0):
    """
    Draw detections on image
    
    Args:
        image: Input image
        detections: List of detections
        show_fps: Whether to show FPS
        fps: FPS value
        
    Returns:
        Image with visualizations
    """
    img_vis = image.copy()
    
    # Define colors for each class
    colors = [
        (0, 0, 255),    # Pothole - Red
        (255, 0, 0),    # Alligator Crack - Blue
        (0, 255, 255),  # Longitudinal Crack - Yellow
        (255, 0, 255)   # Other Damage - Magenta
    ]
    
    for det in detections:
        class_id = det['class_id']
        class_name = det['class_name']
        confidence = det['confidence']
        x1, y1, x2, y2 = det['bbox']
        
        # Get color
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Background for label
        cv2.rectangle(
            img_vis,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        
        # Label text
        cv2.putText(
            img_vis,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    # Show FPS
    if show_fps:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            img_vis,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    
    return img_vis


def main():
    """Test inference engine"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='TensorRT Inference Test')
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT engine')
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--image', type=str, help='Test image path')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--video', type=str, help='Video file path')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = TensorRTInferenceEngine(args.engine, args.config)
    
    # Test mode
    if args.image:
        # Single image inference
        image = cv2.imread(args.image)
        detections = engine.infer(image)
        
        print(f"Detections: {len(detections)}")
        for det in detections:
            print(f"  {det['class_name']}: {det['confidence']:.3f}")
        
        # Visualize
        img_vis = visualize_detections(image, detections)
        cv2.imwrite('detection_result.jpg', img_vis)
        print("Result saved to detection_result.jpg")
        
    else:
        # Video/camera inference
        if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(args.camera)
        
        print("Starting inference... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference
            detections = engine.infer(frame)
            
            # Visualize
            fps = engine.get_fps()
            img_vis = visualize_detections(frame, detections, True, fps)
            
            # Display
            cv2.imshow('Road Anomaly Detection', img_vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
