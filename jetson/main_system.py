"""
Complete Multi-Agent Road Anomaly Detection System
Integrates detection, GPS tagging, and reporting agents
"""

import cv2
import argparse
import json
import signal
import sys
from pathlib import Path
from datetime import datetime
import time

# Import our agents
from inference import JetsonInferenceEngine, CameraStream
from gps_agent import GPSAgent, generate_maps_link
from reporting_agent import ReportingAgent


class RoadAnomalySystem:
    """
    Complete multi-agent system for road anomaly detection
    """
    
    def __init__(self, config_path):
        """
        Args:
            config_path: Path to system configuration JSON
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print("="*70)
        print("🚗 ROAD ANOMALY DETECTION SYSTEM - JETSON ORIN NANO")
        print("="*70)
        print(f"Configuration: {config_path}")
        print()
        
        # Initialize components
        self.inference_engine = None
        self.camera = None
        self.gps_agent = None
        self.reporting_agent = None
        
        # State management
        self.running = False
        self.frame_count = 0
        self.detection_count = 0
        self.last_detection_time = None
        
        # Output directories
        self.output_dir = Path(self.config.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.images_dir = self.output_dir / 'detections'
        self.images_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.start_time = None
        
        # Initialize all components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        print("🔧 Initializing system components...\n")
        
        # 1. Inference Engine
        print("1/4 Loading detection model...")
        model_path = self.config['model']['path']
        
        self.inference_engine = JetsonInferenceEngine(
            model_path=model_path,
            config_path=None  # Use embedded config
        )
        self.inference_engine.config = self.config
        print("   ✅ Detection engine ready\n")
        
        # 2. Camera
        print("2/4 Initializing camera...")
        camera_config = self.config['camera']
        
        # Parse camera source
        source = camera_config['source']
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass  # Keep as string for video files
        
        self.camera = CameraStream(
            source=source,
            width=camera_config['width'],
            height=camera_config['height'],
            fps=camera_config['fps']
        )
        print("   ✅ Camera ready\n")
        
        # 3. GPS Agent
        print("3/4 Initializing GPS agent...")
        gps_config = self.config['gps']
        
        self.gps_agent = GPSAgent(
            port=gps_config.get('port', '/dev/ttyUSB0'),
            baudrate=gps_config.get('baudrate', 9600),
            use_mock=not gps_config.get('enabled', True)
        )
        self.gps_agent.start()
        print("   ✅ GPS agent ready\n")
        
        # 4. Reporting Agent
        print("4/4 Initializing reporting agent...")
        reporting_config = self.config['reporting']
        
        self.reporting_agent = ReportingAgent(
            db_path=reporting_config.get('database', 'detections.db'),
            config=reporting_config
        )
        self.reporting_agent.start()
        print("   ✅ Reporting agent ready\n")
        
        print("="*70)
        print("✅ ALL SYSTEMS READY")
        print("="*70)
        print()
    
    def run(self):
        """Main system loop"""
        
        self.running = True
        self.start_time = time.time()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🎥 Starting detection loop...")
        print("Press Ctrl+C to stop")
        print()
        
        # Main loop
        try:
            while self.running:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    print("⚠️ Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Run inference
                detections = self.inference_engine.infer(frame)
                
                # Process detections
                if len(detections) > 0:
                    self._process_detections(frame, detections)
                
                # Display (if enabled)
                if self.config.get('display', False):
                    self._display_frame(frame, detections)
                
                # Print status periodically
                if self.frame_count % 100 == 0:
                    self._print_status()
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def _process_detections(self, frame, detections):
        """Process detected anomalies"""
        
        for detection in detections:
            self.detection_count += 1
            self.last_detection_time = datetime.now()
            
            # Add GPS coordinates
            detection = self.gps_agent.tag_detection(detection)
            
            # Save detection image (if confidence is high enough)
            min_confidence = self.config['reporting'].get('min_confidence', 0.5)
            
            if detection['confidence'] >= min_confidence:
                # Save annotated image
                image_filename = self._save_detection_image(frame, detection)
                detection['image_path'] = str(image_filename)
                
                # Queue for reporting
                self.reporting_agent.queue_detection(detection)
                
                # Print detection info
                self._print_detection(detection)
    
    def _save_detection_image(self, frame, detection):
        """Save image with detection"""
        
        # Create annotated copy
        img_annotated = frame.copy()
        
        x1, y1, x2, y2 = detection['bbox']
        color = (0, 0, 255)  # Red
        
        # Draw bounding box
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"
        cv2.putText(
            img_annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        # Add GPS info
        gps_text = f"GPS: {detection['gps']['latitude']:.6f}, {detection['gps']['longitude']:.6f}"
        cv2.putText(
            img_annotated,
            gps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filename = self.images_dir / f"{detection['class_name']}_{timestamp}.jpg"
        cv2.imwrite(str(filename), img_annotated)
        
        return filename
    
    def _display_frame(self, frame, detections):
        """Display frame with detections"""
        
        img_display = frame.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 0, 255)
            
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(
                img_display,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Add status info
        stats = self.inference_engine.get_performance_stats()
        fps_text = f"FPS: {stats['fps']:.1f} | Detections: {self.detection_count}"
        cv2.putText(
            img_display,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Show GPS
        gps_text = self.gps_agent.get_location_string()
        cv2.putText(
            img_display,
            gps_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )
        
        cv2.imshow('Road Anomaly Detection System', img_display)
        cv2.waitKey(1)
    
    def _print_detection(self, detection):
        """Print detection information"""
        
        print(f"\n🚨 DETECTION #{self.detection_count}")
        print(f"   Type: {detection['class_name']}")
        print(f"   Confidence: {detection['confidence']:.2%}")
        print(f"   Location: {detection['gps']['latitude']:.6f}, {detection['gps']['longitude']:.6f}")
        print(f"   Maps: {generate_maps_link(detection['gps']['latitude'], detection['gps']['longitude'])}")
        print(f"   Time: {detection['timestamp']}")
    
    def _print_status(self):
        """Print system status"""
        
        uptime = time.time() - self.start_time
        stats = self.inference_engine.get_performance_stats()
        
        print(f"\n📊 STATUS @ Frame {self.frame_count}")
        print(f"   Uptime: {uptime/60:.1f} minutes")
        print(f"   FPS: {stats['fps']:.1f}")
        print(f"   Total detections: {self.detection_count}")
        print(f"   Location: {self.gps_agent.get_location_string()}")
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print("\n⚠️ Shutdown signal received")
        self.running = False
    
    def shutdown(self):
        """Gracefully shutdown all components"""
        
        print("\n" + "="*70)
        print("🛑 SHUTTING DOWN SYSTEM")
        print("="*70)
        
        # Stop components
        if self.gps_agent:
            self.gps_agent.stop()
        
        if self.reporting_agent:
            self.reporting_agent.stop()
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        self._print_final_stats()
        
        print("\n✅ System shutdown complete")
    
    def _print_final_stats(self):
        """Print final statistics"""
        
        uptime = time.time() - self.start_time if self.start_time else 0
        
        print("\n📈 FINAL STATISTICS")
        print(f"   Total runtime: {uptime/60:.1f} minutes")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Total detections: {self.detection_count}")
        print(f"   Detection rate: {self.detection_count / (uptime/60):.2f} per minute")
        
        # Get reporting statistics
        if self.reporting_agent:
            print("\n📊 Detection breakdown:")
            stats = self.reporting_agent.get_statistics()
            for stat in stats:
                print(f"   {stat['class_name']}: {stat['count']}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Complete Road Anomaly Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with custom config
  python main_system.py --config deployment_config.json
  
  # Run with display enabled
  python main_system.py --config deployment_config.json --display
  
  # Run with custom output directory
  python main_system.py --config deployment_config.json --output /data/detections
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/deployment_config.json',
        help='Path to system configuration file'
    )
    
    parser.add_argument(
        '--display',
        action='store_true',
        help='Enable video display'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Override output directory'
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        print("\nPlease create a configuration file or specify the correct path.")
        sys.exit(1)
    
    # Load and modify config if needed
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.display:
        config['display'] = True
    
    if args.output:
        config['output_dir'] = args.output
    
    # Save modified config
    temp_config = Path('temp_config.json')
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Initialize and run system
        system = RoadAnomalySystem(str(temp_config))
        system.run()
    finally:
        # Cleanup temp config
        if temp_config.exists():
            temp_config.unlink()


if __name__ == "__main__":
    main()
