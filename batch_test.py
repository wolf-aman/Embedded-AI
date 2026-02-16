"""
Batch Testing and Evaluation Script
Process multiple images/videos and generate comprehensive reports
Optimized for Jetson Nano/Orin Nano with memory-efficient generators
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time
import json
import gc
from datetime import datetime
from ultralytics import YOLO
import os
from tqdm import tqdm

try:
    from jetson.memory_monitor import MemoryMonitor
except ImportError:
    # Create a dummy MemoryMonitor if not available
    class MemoryMonitor:
        def __init__(self, *args, **kwargs):
            pass
        def measure(self, *args, **kwargs):
            pass
        def print_summary(self):
            pass
        def check_memory_threshold(self, *args, **kwargs):
            return True


class BatchTester:
    """Batch testing for road anomaly detection - Memory optimized for Jetson"""
    
    def __init__(self, model_path, output_dir='test_results', enable_memory_monitor=True):
        """
        Args:
            model_path: Path to model (.pt, .onnx, or .engine)
            output_dir: Directory to save results
            enable_memory_monitor: Enable memory monitoring
        """
        print(f"📦 Loading model: {model_path}")
        
        # Auto-detect device
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  Using CPU (no GPU detected)")
        
        self.model = YOLO(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(enable_logging=enable_memory_monitor)
        self.memory_monitor.measure("Model loaded")
        
        # Class names and colors
        self.class_names = {
            0: "Pothole",
            1: "Alligator Crack",
            2: "Longitudinal Crack",
            3: "Other Damage"
        }
        
        self.colors = {
            0: (0, 0, 255),      # Red
            1: (0, 165, 255),    # Orange
            2: (0, 255, 255),    # Yellow
            3: (255, 0, 255)     # Magenta
        }
        
        # Input size for early resizing
        self.input_size = 640
        
        print(f"✅ Model loaded successfully")
        print(f"📁 Output directory: {self.output_dir}")
    
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
    
    def process_image(self, image_path, conf=0.35, iou=0.45, save_viz=True):
        """
        Process a single image
        
        Returns:
            Dictionary with results and metrics
        """
        # Read image (Best Practice #1: Don't load all at once)
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Preprocess: Resize early to save memory (Best Practice #2)
        image = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        results = self.model(image, conf=conf, iou=iou, verbose=False)
        inference_time = (time.time() - start_time) * 1000
        
        # Process results
        detections = []
        result = results[0]
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            detections.append({
                'class_id': cls_id,
                'class_name': self.class_names[cls_id],
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
            
            # Draw on image
            if save_viz:
                color = self.colors[cls_id]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                label = f"{self.class_names[cls_id]}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1, y1 - label_size[1] - 5), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add info overlay
        info_text = f"Detections: {len(detections)} | Time: {inference_time:.1f}ms"
        cv2.putText(image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save visualization
        if save_viz:
            output_path = self.output_dir / f"viz_{Path(image_path).name}"
            cv2.imwrite(str(output_path), image)
        
        return {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'detections': detections,
            'num_detections': len(detections),
            'inference_time_ms': inference_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def image_generator(self, dir_path, extensions=None):
        """
        Generator function to yield image files one at a time
        (Best Practice #1: Never load full dataset)
        (Best Practice #4: Use generators)
        
        Args:
            dir_path: Path to directory with images
            extensions: List of image extensions to process
            
        Yields:
            Path to image file
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        dir_path = Path(dir_path)
        
        for ext in extensions:
            # Use generator pattern - yield one at a time
            for image_file in dir_path.glob(f"*{ext}"):
                yield image_file
            for image_file in dir_path.glob(f"*{ext.upper()}"):
                yield image_file
    
    def process_directory(self, dir_path, conf=0.35, iou=0.45, extensions=None):
        """
        Process all images in a directory using generator (memory-efficient)
        
        Args:
            dir_path: Path to directory with images
            conf: Confidence threshold
            iou: IoU threshold
            extensions: List of image extensions to process
        
        Returns:
            List of results and summary statistics
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        dir_path = Path(dir_path)
        
        # Use generator to get image files (Best Practice #1: Never load full dataset)
        image_files = list(self.image_generator(dir_path, extensions))
        image_files = sorted(set(image_files))
        
        if not image_files:
            print(f"❌ No images found in {dir_path}")
            return None
        
        print(f"\n📸 Processing {len(image_files)} images from {dir_path}")
        print(f"   Confidence: {conf}, IoU: {iou}")
        
        self.memory_monitor.measure("Before batch processing")
        
        results = []
        stats = {
            'total_images': len(image_files),
            'images_with_detections': 0,
            'total_detections': 0,
            'class_counts': {name: 0 for name in self.class_names.values()},
            'avg_inference_time': 0,
            'avg_detections_per_image': 0
        }
        
        inference_times = []
        
        # Process images one at a time using generator pattern (Best Practice #4)
        for idx, image_file in enumerate(tqdm(image_files, desc="Processing")):
            result = self.process_image(image_file, conf, iou)
            
            if result:
                results.append(result)
                inference_times.append(result['inference_time_ms'])
                
                if result['num_detections'] > 0:
                    stats['images_with_detections'] += 1
                    stats['total_detections'] += result['num_detections']
                    
                    for det in result['detections']:
                        stats['class_counts'][det['class_name']] += 1
            
            # Periodic memory cleanup and monitoring (Best Practice #5)
            if (idx + 1) % 10 == 0:
                gc.collect()
                self.memory_monitor.measure(f"After {idx + 1} images")
                self.memory_monitor.check_memory_threshold(threshold_mb=3000)
        
        # Calculate statistics
        if inference_times:
            stats['avg_inference_time'] = np.mean(inference_times)
            stats['min_inference_time'] = np.min(inference_times)
            stats['max_inference_time'] = np.max(inference_times)
        
        if stats['total_images'] > 0:
            stats['avg_detections_per_image'] = stats['total_detections'] / stats['total_images']
            stats['detection_rate'] = stats['images_with_detections'] / stats['total_images']
        
        # Print memory summary (Best Practice #5)
        self.memory_monitor.measure("After batch processing")
        self.memory_monitor.print_summary()
        
        return results, stats
    
    def process_video_batch(self, video_path, conf=0.35, iou=0.45, sample_rate=1, max_frames=None):
        """
        Process video file with frame sampling
        
        Args:
            video_path: Path to video file
            conf: Confidence threshold
            iou: IoU threshold
            sample_rate: Process every Nth frame
            max_frames: Maximum frames to process (None for all)
        
        Returns:
            Results and statistics
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"\n🎥 Processing video: {Path(video_path).name}")
        print(f"   Resolution: {width}x{height} @ {fps} FPS")
        print(f"   Total frames: {total_frames}, Sample rate: 1/{sample_rate}")
        
        # Create output video
        output_video_path = self.output_dir / f"annotated_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        results = []
        stats = {
            'video_path': str(video_path),
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'frames_with_detections': 0,
            'class_counts': {name: 0 for name in self.class_names.values()},
            'avg_inference_time': 0
        }
        
        inference_times = []
        frame_count = 0
        
        pbar = tqdm(total=total_frames, desc="Processing")
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            frame_count += 1
            stats['total_frames'] = frame_count
            
            # Sample frames
            if frame_count % sample_rate != 0:
                pbar.update(1)
                continue
            
            stats['processed_frames'] += 1
            
            # Run inference
            start_time = time.time()
            model_results = self.model(frame, conf=conf, iou=iou, verbose=False)
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            # Process results
            detections = []
            result = model_results[0]
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    'class_id': cls_id,
                    'class_name': self.class_names[cls_id],
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
                
                # Draw on frame
                color = self.colors[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{self.class_names[cls_id]}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Update stats
                stats['total_detections'] += 1
                stats['class_counts'][self.class_names[cls_id]] += 1
            
            if detections:
                stats['frames_with_detections'] += 1
            
            # Add frame info
            info_text = f"Frame: {frame_count} | Detections: {len(detections)} | {inference_time:.1f}ms"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save to results
            results.append({
                'frame': frame_count,
                'detections': detections,
                'num_detections': len(detections),
                'inference_time_ms': inference_time
            })
            
            # Write frame
            out.write(frame)
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        # Calculate statistics
        if inference_times:
            stats['avg_inference_time'] = np.mean(inference_times)
            stats['avg_fps'] = 1000 / stats['avg_inference_time']
        
        if stats['processed_frames'] > 0:
            stats['detection_rate'] = stats['frames_with_detections'] / stats['processed_frames']
        
        print(f"✅ Video processed and saved to: {output_video_path}")
        
        return results, stats
    
    def generate_report(self, results, stats, report_type='images'):
        """Generate HTML report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"report_{report_type}_{timestamp}.html"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Road Anomaly Detection Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        .stat-value {{ font-size: 32px; font-weight: bold; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .class-badge {{ display: inline-block; padding: 5px 10px; border-radius: 5px; color: white; font-size: 12px; margin-right: 5px; }}
        .pothole {{ background: #e74c3c; }}
        .alligator {{ background: #e67e22; }}
        .longitudinal {{ background: #f39c12; }}
        .other {{ background: #9b59b6; }}
        .timestamp {{ color: #7f8c8d; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛣️ Road Anomaly Detection Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>📊 Summary Statistics</h2>
        <div class="stats">
"""
        
        if report_type == 'images':
            html += f"""
            <div class="stat-card">
                <div class="stat-label">Total Images</div>
                <div class="stat-value">{stats['total_images']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">With Detections</div>
                <div class="stat-value">{stats['images_with_detections']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Detections</div>
                <div class="stat-value">{stats['total_detections']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Inference Time</div>
                <div class="stat-value">{stats['avg_inference_time']:.1f}ms</div>
            </div>
"""
        else:  # video
            html += f"""
            <div class="stat-card">
                <div class="stat-label">Total Frames</div>
                <div class="stat-value">{stats['total_frames']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Processed Frames</div>
                <div class="stat-value">{stats['processed_frames']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Detections</div>
                <div class="stat-value">{stats['total_detections']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg FPS</div>
                <div class="stat-value">{stats.get('avg_fps', 0):.1f}</div>
            </div>
"""
        
        html += """
        </div>
        
        <h2>🏷️ Detections by Class</h2>
        <table>
            <tr>
                <th>Class</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
"""
        
        for class_name, count in stats['class_counts'].items():
            percentage = (count / stats['total_detections'] * 100) if stats['total_detections'] > 0 else 0
            badge_class = class_name.lower().replace(' ', '')
            html += f"""
            <tr>
                <td><span class="class-badge {badge_class}">{class_name}</span></td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"📄 Report saved: {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Batch Testing for Road Anomaly Detection')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file (.pt, .onnx, or .engine)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image directory or video file')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.35,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--sample-rate', type=int, default=1,
                       help='For videos: process every Nth frame')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='For videos: maximum frames to process')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = BatchTester(args.model, args.output)
    
    source_path = Path(args.source)
    
    # Process based on source type
    if source_path.is_dir():
        # Process directory of images
        results, stats = tester.process_directory(
            source_path, 
            conf=args.conf, 
            iou=args.iou
        )
        
        if results:
            # Save results
            results_path = tester.output_dir / 'results_images.json'
            with open(results_path, 'w') as f:
                json.dump({'results': results, 'statistics': stats}, f, indent=2)
            print(f"\n💾 Results saved: {results_path}")
            
            # Generate report
            tester.generate_report(results, stats, 'images')
            
            # Print summary
            print(f"\n{'='*60}")
            print("📊 Summary:")
            print(f"   Total images: {stats['total_images']}")
            print(f"   Images with detections: {stats['images_with_detections']}")
            print(f"   Total detections: {stats['total_detections']}")
            print(f"   Average inference time: {stats['avg_inference_time']:.1f}ms")
            print(f"{'='*60}\n")
    
    elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Process video
        results, stats = tester.process_video_batch(
            source_path,
            conf=args.conf,
            iou=args.iou,
            sample_rate=args.sample_rate,
            max_frames=args.max_frames
        )
        
        if results:
            # Save results
            results_path = tester.output_dir / 'results_video.json'
            with open(results_path, 'w') as f:
                json.dump({'results': results, 'statistics': stats}, f, indent=2)
            print(f"\n💾 Results saved: {results_path}")
            
            # Generate report
            tester.generate_report(results, stats, 'video')
            
            # Print summary
            print(f"\n{'='*60}")
            print("📊 Summary:")
            print(f"   Total frames: {stats['total_frames']}")
            print(f"   Processed frames: {stats['processed_frames']}")
            print(f"   Total detections: {stats['total_detections']}")
            print(f"   Average FPS: {stats.get('avg_fps', 0):.1f}")
            print(f"{'='*60}\n")
    
    else:
        print(f"❌ Invalid source: {source_path}")
        print("   Provide either a directory of images or a video file")
        return 1
    
    return 0


if __name__ == '__main__':
    main()
