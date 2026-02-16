"""
Simple Standalone Object Detection Demo
Test your trained YOLOv8 model on images, videos, or webcam
No dependencies on other modules - works independently
Optimized for Jetson Nano/Orin Nano deployment
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time
import gc
from ultralytics import YOLO

try:
    from jetson.memory_monitor import MemoryMonitor
except ImportError:
    class MemoryMonitor:
        def __init__(self, *args, **kwargs):
            pass
        def measure(self, *args, **kwargs):
            pass
        def check_memory_threshold(self, *args, **kwargs):
            return True


class RoadAnomalyDetector:
    """Standalone detector for road anomalies - Optimized for Jetson"""
    
    def __init__(self, model_path, conf_threshold=0.35, iou_threshold=0.45, enable_memory_monitor=True):
        """
        Args:
            model_path: Path to model (.pt, .onnx, or .engine)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            enable_memory_monitor: Enable memory monitoring
        """
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(enable_logging=enable_memory_monitor)
        self.memory_monitor.measure("Initialization start")
        
        print(f"📦 Loading model: {model_path}")
        
        # Auto-detect device
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  Using CPU (no GPU detected) - inference will be slower")
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Input size for early resizing
        self.input_size = 640
        
        # Class names and colors
        self.class_names = {
            0: "Pothole",
            1: "Alligator Crack",
            2: "Longitudinal Crack",
            3: "Other Damage"
        }
        
        self.colors = {
            0: (0, 0, 255),      # Red for Pothole
            1: (0, 165, 255),    # Orange for Alligator Crack
            2: (0, 255, 255),    # Yellow for Longitudinal Crack
            3: (255, 0, 255)     # Magenta for Other Damage
        }
        
        self.memory_monitor.measure("Model loaded")
        
        print(f"✅ Model loaded successfully")
        print(f"   Classes: {list(self.class_names.values())}")
        print(f"   Confidence: {conf_threshold}, IoU: {iou_threshold}")
    
    def preprocess_image(self, image):
        """
        Preprocess image - RESIZE EARLY to save memory (Best Practice #2)
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        orig_h, orig_w = image.shape[:2]
        
        # Resize early if image is large
        if max(orig_h, orig_w) > self.input_size:
            if orig_w > orig_h:
                new_w = self.input_size
                new_h = int(orig_h * (self.input_size / orig_w))
            else:
                new_h = self.input_size
                new_w = int(orig_w * (self.input_size / orig_h))
            
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return image
    
    def detect_image(self, image_path, output_path=None, show=True):
        """
        Detect road anomalies in a single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)
            show: Display the result
        
        Returns:
            detections: List of detection dictionaries
        """
        print(f"\n🖼️  Processing image: {image_path}")
        
        # Read image (Best Practice #1: Don't load all images at once)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return []
        
        # Preprocess: Resize early to save memory (Best Practice #2)
        image = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        inference_time = (time.time() - start_time) * 1000
        
        # Process results
        detections = []
        result = results[0]
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            detection = {
                'class_id': cls_id,
                'class_name': self.class_names[cls_id],
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            }
            detections.append(detection)
            
            # Draw on image
            color = self.colors[cls_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{self.class_names[cls_id]}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add info overlay
        info_text = f"Detections: {len(detections)} | Time: {inference_time:.1f}ms"
        cv2.putText(image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Print results
        print(f"✅ Found {len(detections)} anomalies ({inference_time:.1f}ms)")
        for det in detections:
            print(f"   • {det['class_name']}: {det['confidence']:.3f}")
        
        # Save if requested
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"💾 Saved to: {output_path}")
        
        # Display if requested
        if show:
            cv2.imshow("Road Anomaly Detection", image)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections
    
    def detect_video(self, video_path, output_path=None, show=True):
        """
        Detect road anomalies in a video file
        
        Args:
            video_path: Path to input video (or 0 for webcam)
            output_path: Path to save annotated video (optional)
            show: Display the result
        
        Returns:
            stats: Dictionary with detection statistics
        """
        # Open video
        if video_path == 0 or str(video_path).isdigit():
            cap = cv2.VideoCapture(int(video_path))
            print(f"\n📹 Opening webcam: {video_path}")
        else:
            cap = cv2.VideoCapture(str(video_path))
            print(f"\n📹 Processing video: {video_path}")
        
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Resolution: {width}x{height} @ {fps} FPS")
        if total_frames > 0:
            print(f"   Total frames: {total_frames}")
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"💾 Saving output to: {output_path}")
        
        # Statistics
        stats = {
            'total_frames': 0,
            'total_detections': 0,
            'class_counts': {name: 0 for name in self.class_names.values()},
            'avg_fps': 0,
            'avg_inference_time': 0
        }
        
        inference_times = []
        frame_count = 0
        
        print("\n▶️  Processing... (Press 'q' to quit)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference
                start_time = time.time()
                results = self.model(frame, conf=self.conf_threshold, 
                                   iou=self.iou_threshold, verbose=False)
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
                
                # Process results
                result = results[0]
                num_detections = len(result.boxes)
                stats['total_detections'] += num_detections
                
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Update statistics
                    stats['class_counts'][self.class_names[cls_id]] += 1
                    
                    # Draw on frame
                    color = self.colors[cls_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{self.class_names[cls_id]}: {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                                 (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add overlay
                current_fps = 1000 / inference_time if inference_time > 0 else 0
                info_text = f"Frame: {frame_count} | Detections: {num_detections} | {current_fps:.1f} FPS"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write frame
                if writer:
                    writer.write(frame)
                
                # Display
                if show:
                    cv2.imshow("Road Anomaly Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⏹️  Stopped by user")
                        break
                
                # Progress update
                if frame_count % 30 == 0:
                    avg_fps = 1000 / np.mean(inference_times[-30:])
                    print(f"   Frame {frame_count}: {num_detections} detections, {avg_fps:.1f} FPS")
        
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Final statistics
        stats['total_frames'] = frame_count
        stats['avg_inference_time'] = np.mean(inference_times)
        stats['avg_fps'] = 1000 / stats['avg_inference_time'] if stats['avg_inference_time'] > 0 else 0
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Total frames: {stats['total_frames']}")
        print(f"   Total detections: {stats['total_detections']}")
        print(f"   Average FPS: {stats['avg_fps']:.1f}")
        print(f"   Average inference: {stats['avg_inference_time']:.1f}ms")
        print(f"\n   Detections by class:")
        for class_name, count in stats['class_counts'].items():
            if count > 0:
                print(f"      {class_name}: {count}")
        
        return stats
    
    def detect_webcam(self, camera_id=0, output_path=None):
        """
        Real-time detection from webcam
        
        Args:
            camera_id: Camera device ID (default: 0)
            output_path: Path to save video (optional)
        """
        return self.detect_video(camera_id, output_path, show=True)


def main():
    parser = argparse.ArgumentParser(description='Road Anomaly Detection Demo')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file (.pt, .onnx, or .engine)')
    parser.add_argument('--source', type=str, required=True,
                       help='Input source (image, video, or webcam ID)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results')
    parser.add_argument('--conf', type=float, default=0.35,
                       help='Confidence threshold (default: 0.35)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--no-show', action='store_true',
                       help='Don\'t display results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RoadAnomalyDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Determine source type
    source_path = Path(args.source)
    
    # Check if webcam
    if args.source.isdigit():
        print(f"\n🎥 Starting webcam detection...")
        detector.detect_webcam(int(args.source), args.output)
    
    # Check if image
    elif source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        detector.detect_image(args.source, args.output, show=not args.no_show)
    
    # Check if video
    elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        detector.detect_video(args.source, args.output, show=not args.no_show)
    
    else:
        print(f"❌ Unsupported source type: {args.source}")
        print("   Supported: images (.jpg, .png), videos (.mp4, .avi), webcam (0, 1, etc.)")
        return 1
    
    print("\n✅ Detection complete!")
    return 0


if __name__ == '__main__':
    main()
