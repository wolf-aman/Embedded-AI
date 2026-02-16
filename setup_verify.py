"""
Setup and Verification Script
Checks dependencies and prepares the environment for detection
"""

import sys
import subprocess
import platform
from pathlib import Path
import json


class SetupManager:
    """Manages setup and verification"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        
    def print_header(self, text):
        """Print formatted header"""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}\n")
    
    def check_python_version(self):
        """Check Python version"""
        print("🐍 Checking Python version...")
        version = sys.version_info
        
        if version.major == 3 and version.minor >= 8:
            print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Required: 3.8+)")
            self.issues.append("Python version must be 3.8 or higher")
            return False
    
    def check_package(self, package_name, import_name=None):
        """Check if a package is installed"""
        if import_name is None:
            import_name = package_name
        
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
            return True
        except ImportError:
            print(f"   ❌ {package_name} not found")
            return False
    
    def check_dependencies(self):
        """Check all required dependencies"""
        print("📦 Checking dependencies...")
        
        required_packages = {
            'opencv-python': 'cv2',
            'numpy': 'numpy',
            'ultralytics': 'ultralytics',
            'torch': 'torch',
            'pillow': 'PIL',
            'tqdm': 'tqdm'
        }
        
        missing = []
        for package, import_name in required_packages.items():
            if not self.check_package(package, import_name):
                missing.append(package)
        
        if missing:
            self.issues.append(f"Missing packages: {', '.join(missing)}")
            print(f"\n   To install missing packages:")
            print(f"   pip install {' '.join(missing)}")
            return False
        
        return True
    
    def check_cuda(self):
        """Check CUDA availability"""
        print("\n🎮 Checking CUDA...")
        
        try:
            import torch
            if torch.cuda.is_available():
                print(f"   ✅ CUDA available")
                print(f"      Device: {torch.cuda.get_device_name(0)}")
                print(f"      Version: {torch.version.cuda}")
                return True
            else:
                print(f"   ⚠️  CUDA not available (CPU mode will be used)")
                self.warnings.append("CUDA not available - inference will be slower on CPU")
                return False
        except ImportError:
            print(f"   ❌ PyTorch not installed")
            return False
    
    def check_model_files(self):
        """Check if model files exist"""
        print("\n🤖 Checking model files...")
        
        model_dir = Path("optimized_models")
        if not model_dir.exists():
            print(f"   ❌ Model directory not found: {model_dir}")
            self.issues.append("optimized_models directory not found")
            return False
        
        # Check for model files
        model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.onnx")) + list(model_dir.glob("*.engine"))
        
        if not model_files:
            print(f"   ❌ No model files found in {model_dir}")
            self.issues.append("No model files (.pt, .onnx, or .engine) found")
            return False
        
        print(f"   ✅ Found {len(model_files)} model file(s):")
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"      • {model_file.name} ({size_mb:.1f} MB)")
        
        return True
    
    def check_config_files(self):
        """Check configuration files"""
        print("\n⚙️  Checking configuration...")
        
        config_file = Path("config/deployment_config.json")
        alt_config = Path("optimized_models/deployment_config.json")
        
        if config_file.exists():
            print(f"   ✅ Config found: {config_file}")
            try:
                with open(config_file) as f:
                    config = json.load(f)
                print(f"      Classes: {config['model']['num_classes']}")
                print(f"      Confidence: {config['model']['confidence_threshold']}")
                return True
            except Exception as e:
                print(f"   ⚠️  Config file exists but has errors: {e}")
                self.warnings.append("Config file has errors")
        elif alt_config.exists():
            print(f"   ✅ Config found: {alt_config}")
            return True
        else:
            print(f"   ⚠️  No config file found (will use defaults)")
            self.warnings.append("No configuration file found")
        
        return True
    
    def check_application_scripts(self):
        """Check if application scripts exist"""
        print("\n📜 Checking application scripts...")
        
        scripts = [
            'detect_demo.py',
            'realtime_detection.py',
            'batch_test.py'
        ]
        
        all_found = True
        for script in scripts:
            if Path(script).exists():
                print(f"   ✅ {script}")
            else:
                print(f"   ❌ {script} not found")
                all_found = False
        
        if not all_found:
            self.issues.append("Some application scripts are missing")
        
        return all_found
    
    def test_model_loading(self):
        """Test loading a model"""
        print("\n🧪 Testing model loading...")
        
        try:
            from ultralytics import YOLO
            
            # Find a model file
            model_dir = Path("optimized_models")
            model_files = list(model_dir.glob("*.pt"))
            
            if not model_files:
                print("   ⚠️  No .pt model files to test")
                return True
            
            model_path = model_files[0]
            print(f"   Testing with: {model_path.name}")
            
            model = YOLO(str(model_path))
            print(f"   ✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Model loading failed: {e}")
            self.issues.append(f"Model loading test failed: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directories...")
        
        directories = [
            'test_results',
            'recordings',
            'logs'
        ]
        
        for directory in directories:
            path = Path(directory)
            if not path.exists():
                path.mkdir(exist_ok=True)
                print(f"   ✅ Created: {directory}")
            else:
                print(f"   ✅ Exists: {directory}")
    
    def generate_quick_start(self):
        """Generate quick start guide"""
        print("\n📝 Generating quick start guide...")
        
        guide = """
╔══════════════════════════════════════════════════════════════════╗
║           ROAD ANOMALY DETECTION - QUICK START GUIDE             ║
╚══════════════════════════════════════════════════════════════════╝

🎯 GETTING STARTED

1. Simple Image Detection:
   python detect_demo.py --model optimized_models/yolov8s_best.pt --source test_image.jpg

2. Webcam Real-Time Detection:
   python realtime_detection.py --model optimized_models/yolov8s_best.pt --source 0

3. Video File Detection:
   python realtime_detection.py --model optimized_models/yolov8s_best.pt --source video.mp4

4. Batch Testing (Directory):
   python batch_test.py --model optimized_models/yolov8s_best.pt --source test_images/

5. Full System (Jetson with GPS):
   python jetson/main_system.py --config config/deployment_config.json

═══════════════════════════════════════════════════════════════════

📋 AVAILABLE APPLICATIONS

• detect_demo.py - Simple standalone detection for images/videos/webcam
• realtime_detection.py - Advanced real-time detection with GUI and recording
• batch_test.py - Batch processing with reports and visualizations
• jetson/main_system.py - Complete multi-agent system for Jetson deployment

═══════════════════════════════════════════════════════════════════

⚙️  COMMON OPTIONS

--model PATH       Path to model file (.pt, .onnx, or .engine)
--source SOURCE    Input source (image, video, or webcam ID)
--conf FLOAT       Confidence threshold (default: 0.35)
--iou FLOAT        IoU threshold for NMS (default: 0.45)
--output PATH      Output path for results
--config PATH      Configuration file path

═══════════════════════════════════════════════════════════════════

🎮 REAL-TIME DETECTION CONTROLS

Q         - Quit application
R         - Start/Stop recording
S         - Toggle statistics panel
SPACE     - Pause/Resume
D         - Toggle detection visualization

═══════════════════════════════════════════════════════════════════

💡 TIPS

• Use FP16/INT8 TensorRT engines on Jetson for better performance
• Adjust confidence threshold based on your accuracy requirements
• For batch testing, use --sample-rate to skip frames in videos
• Check test_results/ directory for generated reports and visualizations

═══════════════════════════════════════════════════════════════════

For more information, see:
• README.md - Complete documentation
• QUICK_START.md - Detailed quick start guide
• JETSON_SETUP.md - Jetson Orin Nano setup instructions

"""
        
        with open("APPLICATION_GUIDE.txt", "w") as f:
            f.write(guide)
        
        print(guide)
        print(f"   💾 Guide saved to: APPLICATION_GUIDE.txt")
    
    def run(self):
        """Run all checks"""
        self.print_header("Road Anomaly Detection - Setup & Verification")
        
        # Run all checks
        self.check_python_version()
        self.check_dependencies()
        self.check_cuda()
        self.check_model_files()
        self.check_config_files()
        self.check_application_scripts()
        self.test_model_loading()
        self.create_directories()
        
        # Print summary
        self.print_header("Summary")
        
        if not self.issues:
            print("✅ All checks passed! System is ready.\n")
            self.generate_quick_start()
            return 0
        else:
            print("❌ Issues found:\n")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
            print()
        
        if self.warnings:
            print("⚠️  Warnings:\n")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
            print()
        
        if self.issues:
            print("Please resolve the issues above and run this script again.\n")
            return 1
        
        return 0


def main():
    """Main entry point"""
    setup = SetupManager()
    return setup.run()


if __name__ == '__main__':
    sys.exit(main())
