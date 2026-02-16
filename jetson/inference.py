"""
Main Inference Pipeline for Jetson Orin Nano
Supports both ONNX and TensorRT engines with Ultralytics
Optimized for low memory usage on embedded devices
"""

import cv2
import numpy as np
from pathlib import Path
import time
import json
import gc
from ultralytics import YOLO
from datetime import datetime
from memory_monitor import MemoryMonitor


class JetsonInferenceEngine:
    """
    Simplified inference engine using Ultralytics YOLO
    Works with .pt, .onnx, or .engine files
    """
    
    def __init__(self, model_path, config_path=None, enable_memory_monitor=True):
        """
        Args:
            model_path: Path to model (.pt, .onnx, or .engine)
            config_path: Path to deployment config JSON
            enable_memory_monitor: Enable memory monitoring for Jetson
        """
        self.model_path = Path(model_path)
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(enable_logging=enable_memory_monitor) if enable_memory_monitor else None
        if self.memory_monitor:
            self.memory_monitor.measure("Initialization start")
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()
        
        # Image preprocessing config (resize early to save memory)
        self.input_size = self.config.get('input_size', 640)  # Default YOLOv8 input size
        
        # Load model
        print(f"📦 Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Configure model
        self.conf_threshold = self.config['model']['confidence_threshold']
        self.iou_threshold = self.config['model']['iou_threshold']
        
        # Performance tracking
        self.inference_times = []
        self.frame_count = 0
        
        if self.memory_monitor:
            self.memory_monitor.measure("Model loaded")
        
        print(f"✅ Model loaded successfully")
        print(f"   Confidence threshold: {self.conf_threshold}")
        print(f"   IoU threshold: {self.iou_threshold}")
        print(f"   Input size: {self.input_size}")
    
    def _default_config(self):
        """Default configuration with automatic device detection"""
        import torch
        
        # Auto-detect CUDA availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print(f"🎮 GPU Detected: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  No GPU detected, using CPU (inference will be slower)")
        
        return {
            'model': {
                'confidence_threshold': 0.35,
                'iou_threshold': 0.45,
                'num_classes': 4,
                'class_names': ['Pothole', 'Alligator Crack', 'Longitudinal Crack', 'Other Damage']
            },
            'inference': {
                'device': device,  # Auto-detected: 'cuda' or 'cpu'
                'half': True if device == 'cuda' else False,  # FP16 only on GPU
            },
            'input_size': 640,  # Resize images early to save memory
            'max_memory_mb': 3000  # Memory threshold for Jetson Nano 4GB
        }
    
    def preprocess_image(self, image):
        """
        Preprocess image - RESIZE EARLY to save memory
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image
        """
        # Get original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Resize early to save memory (Best Practice #2)
        if max(orig_h, orig_w) > self.input_size:
            # Calculate aspect-preserving dimensions
            if orig_w > orig_h:
                new_w = self.input_size
                new_h = int(orig_h * (self.input_size / orig_w))
            else:
                new_h = self.input_size
                new_w = int(orig_w * (self.input_size / orig_h))
            
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return image
    
    def infer(self, image, return_annotated=False):
        """
        Run inference on an image
        
        Args:
            image: Input image (BGR format from OpenCV)
            return_annotated: Whether to return annotated image
            
        Returns:
            detections: List of detection dictionaries
            annotated_image: Annotated image (if return_annotated=True)
        """
        # Measure memory before inference
        if self.memory_monitor and self.frame_count % 100 == 0:
            self.memory_monitor.measure(f"Frame {self.frame_count}")
            self.memory_monitor.check_memory_threshold(
                threshold_mb=self.config.get('max_memory_mb', 3000)
            )
        
        start_time = time.time()
        
        # Preprocess: Resize early to save memory (Best Practice #2)
        image_preprocessed = self.preprocess_image(image)
        
        # Run inference
        results = self.model.predict(
            image_preprocessed,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.config['inference']['device'],
            half=self.config['inference'].get('half', True),
            verbose=False
        )
        
        # Track performance
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        # Parse results
        detections = self._parse_results(results[0])
        
        # Periodic memory cleanup
        if self.frame_count % 50 == 0:
            gc.collect()
        
        if return_annotated:
            annotated_image = self._annotate_image(image_preprocessed, detections)
            return detections, annotated_image
        
        return detections
    
    def _parse_results(self, result):
        """Parse YOLO results into detection dictionaries"""
        
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for i in range(len(boxes)):
            detection = {
                'class_id': int(class_ids[i]),
                'class_name': self.config['model']['class_names'][class_ids[i]],
                'confidence': float(confidences[i]),
                'bbox': [int(x) for x in boxes[i]],  # [x1, y1, x2, y2]
                'timestamp': datetime.now().isoformat()
            }
            detections.append(detection)
        
        return detections
    
    def _annotate_image(self, image, detections):
        """Draw detections on image"""
        
        img_annotated = image.copy()
        
        # Define colors for each class
        colors = {
            0: (0, 0, 255),      # Pothole - Red
            1: (255, 0, 0),      # Alligator Crack - Blue
            2: (0, 255, 255),    # Longitudinal Crack - Yellow
            3: (255, 0, 255)     # Other Damage - Magenta
        }
        
        for det in detections:
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            color = colors.get(class_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                img_annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            cv2.putText(
                img_annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return img_annotated
    
    def get_performance_stats(self):
        """Get performance statistics"""
        
        if not self.inference_times:
            return {
                'avg_inference_time_ms': 0,
                'fps': 0,
                'frame_count': 0
            }
        
        recent_times = self.inference_times[-100:]  # Last 100 frames
        avg_time = np.mean(recent_times)
        
        return {
            'avg_inference_time_ms': avg_time,
            'fps': 1000.0 / avg_time if avg_time > 0 else 0,
            'frame_count': self.frame_count,
            'min_time_ms': np.min(recent_times),
            'max_time_ms': np.max(recent_times)
        }


class CameraStream:
    """
    Camera stream handler for Jetson
    Supports CSI camera, USB camera, and video files
    """
    
    def __init__(self, source=0, width=1280, height=720, fps=30):
        """
        Args:
            source: Camera index (0 for CSI), video file path, or RTSP stream
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize capture
        self._init_capture()
    
    def _init_capture(self):
        """Initialize video capture"""
        
        if isinstance(self.source, int) and self.source == 0:
            # Try CSI camera first (Jetson)
            gst_pipeline = self._get_gstreamer_pipeline()
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                # Fall back to regular camera
                print("⚠️ GStreamer CSI camera failed, trying USB camera...")
                self.cap = cv2.VideoCapture(0)
        else:
            # Video file or USB camera
            self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera/video source: {self.source}")
        
        # Set properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        print(f"✅ Camera initialized: {self.width}x{self.height} @ {self.fps}fps")
    
    def _get_gstreamer_pipeline(self):
        """Get GStreamer pipeline for CSI camera on Jetson"""
        
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), "
            f"width=(int){self.width}, height=(int){self.height}, "
            f"format=(string)NV12, framerate=(fraction){self.fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width=(int){self.width}, height=(int){self.height}, "
            f"format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! "
            f"appsink"
        )
    
    def read(self):
        """Read a frame from the camera"""
        return self.cap.read()
    
    def release(self):
        """Release the camera"""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def main():
    """Main inference loop"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Road Anomaly Detection - Jetson Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--source', type=str, default='0', help='Camera index or video file')
    parser.add_argument('--width', type=int, default=1280, help='Frame width')
    parser.add_argument('--height', type=int, default=720, help='Frame height')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
    parser.add_argument('--display', action='store_true', help='Display output')
    parser.add_argument('--save', type=str, help='Save output video to file')
    
    args = parser.parse_args()
    
    # Parse source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    print("="*60)
    print("🚗 Road Anomaly Detection System - Jetson Orin Nano")
    print("="*60)
    
    # Initialize inference engine
    engine = JetsonInferenceEngine(args.model, args.config)
    
    # Initialize camera
    camera = CameraStream(source, args.width, args.height, args.fps)
    
    # Video writer (optional)
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (args.width, args.height))
    
    print("\n🎥 Starting inference... Press 'q' to quit\n")
    
    try:
        while True:
            # Read frame
            ret, frame = camera.read()
            if not ret:
                print("⚠️ Failed to read frame")
                break
            
            # Run inference
            detections, annotated_frame = engine.infer(frame, return_annotated=True)
            
            # Get performance stats
            stats = engine.get_performance_stats()
            
            # Draw FPS on frame
            fps_text = f"FPS: {stats['fps']:.1f} | Detections: {len(detections)}"
            cv2.putText(
                annotated_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Print detections
            if len(detections) > 0 and engine.frame_count % 30 == 0:  # Print every 30 frames
                print(f"\nFrame {engine.frame_count}:")
                for det in detections:
                    print(f"  {det['class_name']}: {det['confidence']:.3f}")
                print(f"  Performance: {stats['fps']:.1f} FPS")
            
            # Display
            if args.display:
                cv2.imshow('Road Anomaly Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Save
            if writer:
                writer.write(annotated_frame)
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        # Cleanup
        camera.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        final_stats = engine.get_performance_stats()
        print("\n" + "="*60)
        print("📊 Final Performance Statistics")
        print("="*60)
        print(f"Total frames: {final_stats['frame_count']}")
        print(f"Average FPS: {final_stats['fps']:.2f}")
        print(f"Average inference time: {final_stats['avg_inference_time_ms']:.2f} ms")
        print(f"Min inference time: {final_stats['min_time_ms']:.2f} ms")
        print(f"Max inference time: {final_stats['max_time_ms']:.2f} ms")
        print("="*60)


if __name__ == "__main__":
    main()
