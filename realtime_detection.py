"""
Real-Time Road Anomaly Detection Application
Advanced GUI with statistics, alerts, and recording capabilities
Optimized for Jetson Nano/Orin Nano deployment
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time
import json
import gc
from datetime import datetime
from collections import deque
from ultralytics import YOLO

try:
    from jetson.memory_monitor import MemoryMonitor
except ImportError:
    # Create a dummy MemoryMonitor if not available
    class MemoryMonitor:
        def __init__(self, *args, **kwargs):
            pass
        def measure(self, *args, **kwargs):
            pass
        def check_memory_threshold(self, *args, **kwargs):
            return True


class RealTimeDetector:
    """Real-time road anomaly detector with advanced features - Optimized for Jetson"""
    
    def __init__(self, model_path, config_path=None, enable_memory_monitor=True):
        """
        Args:
            model_path: Path to model (.pt, .onnx, or .engine)
            config_path: Path to deployment config JSON (optional)
            enable_memory_monitor: Enable memory monitoring
        """
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(enable_logging=enable_memory_monitor)
        self.memory_monitor.measure("Initialization start")
        
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.conf_threshold = config['model']['confidence_threshold']
                self.iou_threshold = config['model']['iou_threshold']
                self.class_names = {i: name for i, name in enumerate(config['model']['class_names'])}
                self.input_size = config.get('input_size', 640)
        else:
            self.conf_threshold = 0.35
            self.iou_threshold = 0.45
            self.input_size = 640  # Default YOLOv8 input size
            self.class_names = {
                0: "Pothole",
                1: "Alligator Crack",
                2: "Longitudinal Crack",
                3: "Other Damage"
            }
        
        # Load model
        print(f"📦 Loading model: {model_path}")
        
        # Auto-detect device
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  Using CPU (no GPU detected)")
        
        self.model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        
        self.memory_monitor.measure("Model loaded")
        
        # Colors for each class
        self.colors = {
            0: (0, 0, 255),      # Red
            1: (0, 165, 255),    # Orange
            2: (0, 255, 255),    # Yellow
            3: (255, 0, 255)     # Magenta
        }
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'class_counts': {name: 0 for name in self.class_names.values()},
            'high_confidence_detections': 0
        }
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.inference_history = deque(maxlen=30)
        
        # Recording
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        # Alert system
        self.alert_active = False
        self.alert_start_time = None
        self.alert_duration = 2.0  # seconds
        
        # UI state
        self.show_stats = True
        self.show_detections = True
        self.paused = False
        
        # Frame counter for periodic operations
        self.frame_count = 0
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame - RESIZE EARLY to save memory (Best Practice #2)
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        orig_h, orig_w = frame.shape[:2]
        
        # Resize early if frame is large
        if max(orig_h, orig_w) > self.input_size:
            if orig_w > orig_h:
                new_w = self.input_size
                new_h = int(orig_h * (self.input_size / orig_w))
            else:
                new_h = self.input_size
                new_w = int(orig_w * (self.input_size / orig_h))
            
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    def _calculate_fps(self, inference_time):
        """Calculate FPS from inference time"""
        if inference_time > 0:
            fps = 1000 / inference_time
            self.fps_history.append(fps)
            return np.mean(self.fps_history)
        return 0
    
    def _draw_info_panel(self, frame, detections, fps, inference_time):
        """Draw information panel on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent panel background
        panel_height = 200 if self.show_stats else 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        title = "Road Anomaly Detection System"
        cv2.putText(frame, title, (10, 30), 
                   cv2.FONT_HERSHEY_BOLD, 0.8, (0, 255, 0), 2)
        
        # Status indicators
        status_x = 10
        status_y = 60
        
        # Recording indicator
        if self.recording:
            recording_text = "● REC"
            cv2.putText(frame, recording_text, (status_x, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            status_x += 100
        
        # Paused indicator
        if self.paused:
            cv2.putText(frame, "PAUSED", (status_x, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            status_x += 100
        
        # Current detections
        det_text = f"Detections: {len(detections)}"
        cv2.putText(frame, det_text, (status_x, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Performance metrics
        perf_text = f"FPS: {fps:.1f} | Latency: {inference_time:.1f}ms"
        cv2.putText(frame, perf_text, (w - 300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detailed statistics
        if self.show_stats:
            stats_y = 100
            cv2.putText(frame, "Statistics:", (10, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            stats_y += 25
            
            cv2.putText(frame, f"Total Frames: {self.stats['total_frames']}", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            stats_y += 20
            
            cv2.putText(frame, f"Total Detections: {self.stats['total_detections']}", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            stats_y += 20
            
            cv2.putText(frame, "By Class:", (10, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            stats_y += 20
            
            for class_id, class_name in self.class_names.items():
                count = self.stats['class_counts'][class_name]
                color = self.colors[class_id]
                cv2.putText(frame, f"  {class_name}: {count}", 
                           (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                stats_y += 18
        
        # Alert overlay
        if self.alert_active:
            elapsed = time.time() - self.alert_start_time
            if elapsed < self.alert_duration:
                alert_text = "⚠ HIGH SEVERITY ANOMALY DETECTED ⚠"
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_BOLD, 1.0, 2)[0]
                alert_x = (w - text_size[0]) // 2
                alert_y = h - 50
                
                # Flashing effect
                if int(elapsed * 4) % 2 == 0:
                    cv2.putText(frame, alert_text, (alert_x, alert_y), 
                               cv2.FONT_HERSHEY_BOLD, 1.0, (0, 0, 255), 3)
            else:
                self.alert_active = False
        
        return frame
    
    def _draw_detections(self, frame, detections):
        """Draw detection boxes and labels"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            conf = det['confidence']
            color = self.colors[cls_id]
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class_name']}: {conf:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0] + 4, y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # High confidence indicator
            if conf > 0.7:
                cv2.circle(frame, (x2 - 10, y1 + 10), 5, (0, 255, 0), -1)
        
        return frame
    
    def _draw_controls_help(self, frame):
        """Draw control instructions"""
        h, w = frame.shape[:2]
        help_y = h - 120
        
        controls = [
            "Controls:",
            "Q: Quit",
            "R: Start/Stop Recording",
            "S: Toggle Stats",
            "SPACE: Pause/Resume"
        ]
        
        for i, text in enumerate(controls):
            cv2.putText(frame, text, (w - 220, help_y + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def _start_recording(self, width, height, fps):
        """Start video recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"detection_recording_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.recording = True
        self.recording_start_time = time.time()
        
        print(f"🔴 Recording started: {output_path}")
    
    def _stop_recording(self):
        """Stop video recording"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            
            duration = time.time() - self.recording_start_time
            print(f"⏹️  Recording stopped (Duration: {duration:.1f}s)")
    
    def run(self, source=0, save_detections=True):
        """
        Run real-time detection
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            save_detections: Save detection log to JSON
        """
        # Open video source
        if isinstance(source, int) or source.isdigit():
            cap = cv2.VideoCapture(int(source))
            print(f"📹 Opening camera: {source}")
        else:
            cap = cv2.VideoCapture(str(source))
            print(f"📹 Opening video: {source}")
        
        if not cap.isOpened():
            print(f"❌ Failed to open video source: {source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Video opened: {width}x{height} @ {fps} FPS")
        print(f"\n▶️  Starting detection... (Press 'Q' to quit)\n")
        
        # Detection log
        detection_log = []
        
        try:
            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("⚠️  End of video stream")
                        break
                    
                    self.stats['total_frames'] += 1
                    
                    # Measure memory periodically (Best Practice #5)
                    if self.stats['total_frames'] % 100 == 0:
                        self.memory_monitor.measure(f"Frame {self.stats['total_frames']}")
                        self.memory_monitor.check_memory_threshold(threshold_mb=3000)
                    
                    # Preprocess: Resize early to save memory (Best Practice #2)
                    frame_processed = self.preprocess_frame(frame)
                    
                    # Run inference
                    start_time = time.time()
                    results = self.model(frame_processed, conf=self.conf_threshold, 
                                       iou=self.iou_threshold, verbose=False)
                    inference_time = (time.time() - start_time) * 1000
                    self.inference_history.append(inference_time)
                    
                    # Process detections
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
                            'bbox': [x1, y1, x2, y2],
                            'timestamp': datetime.now().isoformat()
                        }
                        detections.append(detection)
                        
                        # Update statistics
                        self.stats['total_detections'] += 1
                        self.stats['class_counts'][self.class_names[cls_id]] += 1
                        
                        if conf > 0.7:
                            self.stats['high_confidence_detections'] += 1
                        
                        # Trigger alert for potholes with high confidence
                        if cls_id == 0 and conf > 0.7:
                            self.alert_active = True
                            self.alert_start_time = time.time()
                    
                    # Save to log
                    if save_detections and detections:
                        detection_log.append({
                            'frame': self.stats['total_frames'],
                            'timestamp': datetime.now().isoformat(),
                            'detections': detections
                        })
                    
                    # Draw detections
                    if self.show_detections:
                        frame_processed = self._draw_detections(frame_processed, detections)
                    
                    # Calculate FPS
                    fps_current = self._calculate_fps(inference_time)
                    
                    # Draw UI elements
                    frame_processed = self._draw_info_panel(frame_processed, detections, fps_current, inference_time)
                    frame_processed = self._draw_controls_help(frame_processed)
                    
                    # Record if enabled
                    if self.recording and self.video_writer:
                        self.video_writer.write(frame_processed)
                    
                    # Periodic memory cleanup
                    if self.stats['total_frames'] % 50 == 0:
                        gc.collect()
                    
                    # Display
                    cv2.imshow("Real-Time Road Anomaly Detection", frame_processed)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n⏹️  Quit requested")
                    break
                elif key == ord('r'):
                    if not self.recording:
                        self._start_recording(width, height, fps)
                    else:
                        self._stop_recording()
                elif key == ord('s'):
                    self.show_stats = not self.show_stats
                elif key == ord(' '):
                    self.paused = not self.paused
                    status = "paused" if self.paused else "resumed"
                    print(f"⏸️  {status.capitalize()}")
                elif key == ord('d'):
                    self.show_detections = not self.show_detections
        
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        
        finally:
            # Cleanup
            if self.recording:
                self._stop_recording()
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Save detection log
            if save_detections and detection_log:
                log_path = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(log_path, 'w') as f:
                    json.dump({
                        'statistics': self.stats,
                        'detections': detection_log
                    }, f, indent=2)
                print(f"💾 Detection log saved: {log_path}")
            
            # Print final statistics
            self._print_summary()
    
    def _print_summary(self):
        """Print detection summary"""
        print(f"\n{'='*60}")
        print("📊 Detection Summary")
        print(f"{'='*60}")
        print(f"Total Frames Processed: {self.stats['total_frames']}")
        print(f"Total Detections: {self.stats['total_detections']}")
        print(f"High Confidence Detections: {self.stats['high_confidence_detections']}")
        
        if self.inference_history:
            avg_inference = np.mean(self.inference_history)
            avg_fps = 1000 / avg_inference if avg_inference > 0 else 0
            print(f"Average Inference Time: {avg_inference:.1f}ms")
            print(f"Average FPS: {avg_fps:.1f}")
        
        print(f"\nDetections by Class:")
        for class_name, count in self.stats['class_counts'].items():
            percentage = (count / self.stats['total_detections'] * 100) if self.stats['total_detections'] > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Real-Time Road Anomaly Detection Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use webcam with trained model
  python realtime_detection.py --model optimized_models/yolov8s_best.pt --source 0
  
  # Process video file
  python realtime_detection.py --model optimized_models/yolov8s_best.pt --source video.mp4
  
  # Use with config file
  python realtime_detection.py --model optimized_models/yolov8s_best.pt --config config/deployment_config.json --source 0
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file (.pt, .onnx, or .engine)')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (camera ID or video file path)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to deployment config JSON (optional)')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save detection log')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RealTimeDetector(
        model_path=args.model,
        config_path=args.config
    )
    
    # Parse source
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Run detection
    detector.run(source=source, save_detections=not args.no_save)


if __name__ == '__main__':
    main()
