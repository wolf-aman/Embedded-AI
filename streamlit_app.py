"""
Streamlit Web GUI for Road Anomaly Detection
Professional web interface for real-time detection and analysis
Optimized for Jetson Nano/Orin Nano deployment
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
import gc
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from jetson.memory_monitor import MemoryMonitor
except ImportError:
    # Create a dummy MemoryMonitor if not available
    class MemoryMonitor:
        def __init__(self, *args, **kwargs):
            self.enabled = False
        def measure(self, *args, **kwargs):
            pass
        def check_memory_threshold(self, *args, **kwargs):
            return True
        def print_summary(self):
            pass

# Page config
st.set_page_config(
    page_title="Road Anomaly Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .detection-box {
        border: 2px solid #1f77b4;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)


class RoadAnomalyDetectorGUI:
    """Road anomaly detector with Streamlit GUI - Optimized for Jetson"""
    
    def __init__(self):
        self.class_names = {
            0: "Pothole",
            1: "Alligator Crack",
            2: "Longitudinal Crack",
            3: "Other Damage"
        }
        
        self.colors = {
            0: (255, 0, 0),      # Red for Pothole
            1: (255, 165, 0),    # Orange for Alligator Crack
            2: (255, 255, 0),    # Yellow for Longitudinal Crack
            3: (255, 0, 255)     # Magenta for Other Damage
        }
        
        self.color_names = {
            0: "Red",
            1: "Orange",
            2: "Yellow",
            3: "Magenta"
        }
        
        # Input size for early resizing
        self.input_size = 640
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(enable_logging=False)
    
    def preprocess_image(self, image):
        """
        Preprocess image - RESIZE EARLY to save memory (Best Practice #2)
        
        Args:
            image: Input image (numpy array)
            
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
    
    @st.cache_resource
    def load_model(_self, model_path):
        """Load YOLO model (cached)"""
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            conf = det['confidence']
            
            color = self.colors[cls_id]
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{det['class_name']}: {conf:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Label background
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 4, y1), color, -1)
            
            # Label text
            cv2.putText(img, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img
    
    def detect_image(self, model, image, conf_threshold, iou_threshold):
        """Detect anomalies in image"""
        # Measure memory
        self.memory_monitor.measure("Before detection")
        
        # Preprocess: Resize early to save memory (Best Practice #2)
        image = self.preprocess_image(image)
        
        start_time = time.time()
        results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        inference_time = (time.time() - start_time) * 1000
        
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
                'bbox': [x1, y1, x2, y2],
                'color': self.color_names[cls_id]
            })
        
        return detections, inference_time
    
    def process_video(self, model, video_path, conf_threshold, iou_threshold, progress_bar, status_text):
        """Process video file - Memory optimized with generators (Best Practice #4)"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Measure memory
        self.memory_monitor.measure("Video processing start")
        
        # Create temporary output file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        # Process frames one at a time (Best Practice #1: Never load full dataset)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Detect (preprocessing done inside detect_image)
            detections, _ = self.detect_image(model, frame, conf_threshold, iou_threshold)
            all_detections.extend(detections)
            
            # Draw
            if detections:
                frame = self.draw_detections(frame, detections)
            
            # Write frame
            out.write(frame)
            
            # Periodic memory cleanup (Best Practice #5)
            if frame_count % 50 == 0:
                gc.collect()
                self.memory_monitor.measure(f"Frame {frame_count}")
                self.memory_monitor.check_memory_threshold(threshold_mb=3000)
        
        cap.release()
        out.release()
        
        # Final memory measurement
        self.memory_monitor.measure("Video processing complete")
        
        return output_path, all_detections, frame_count
    
    def create_statistics_chart(self, detections):
        """Create detection statistics chart"""
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if not class_counts:
            return None
        
        df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
        
        fig = px.bar(df, x='Class', y='Count',
                     title='Detections by Class',
                     color='Class',
                     color_discrete_map={
                         'Pothole': '#FF0000',
                         'Alligator Crack': '#FFA500',
                         'Longitudinal Crack': '#FFFF00',
                         'Other Damage': '#FF00FF'
                     })
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14),
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_confidence_distribution(self, detections):
        """Create confidence distribution chart"""
        if not detections:
            return None
        
        confidences = [det['confidence'] for det in detections]
        
        fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20,
                                           marker_color='#667eea')])
        fig.update_layout(
            title='Confidence Distribution',
            xaxis_title='Confidence',
            yaxis_title='Count',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        
        return fig


def main():
    """Main application"""
    
    detector = RoadAnomalyDetectorGUI()
    
    # Header
    st.markdown('<h1 class="main-header">🛣️ Road Anomaly Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/pothole.png", width=100)
        st.title("⚙️ Configuration")
        
        # Model selection
        model_dir = Path("optimized_models")
        if not model_dir.exists():
            st.error("❌ Model directory not found!")
            st.stop()
        
        model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.onnx"))
        
        if not model_files:
            st.error("❌ No model files found!")
            st.stop()
        
        model_options = {f.name: str(f) for f in model_files}
        selected_model = st.selectbox("Select Model", list(model_options.keys()), index=0)
        model_path = model_options[selected_model]
        
        st.divider()
        
        # Detection parameters (Optimized defaults for best results)
        st.subheader("Detection Parameters")
        with st.expander("⚙️ Advanced Settings (Optional)", expanded=False):
            conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05,
                                       help="Minimum confidence for detections (Default: 0.35 - Optimized)")
            iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05,
                                      help="IoU threshold for NMS (Default: 0.45 - Optimized)")
            st.info("💡 Default values are optimized for best detection results. Only adjust if needed.")
        
        st.divider()
        
        # About
        st.subheader("📋 Detected Classes")
        st.markdown("""
        - 🔴 **Pothole**
        - 🟠 **Alligator Crack**
        - 🟡 **Longitudinal Crack**
        - 🟣 **Other Damage**
        """)
        
        st.divider()
        st.caption("Built with ❤️ using YOLOv8 & Streamlit")
    
    # Load model
    with st.spinner("🔄 Loading model..."):
        model = detector.load_model(model_path)
    
    if model is None:
        st.error("Failed to load model!")
        st.stop()
    
    st.success(f"✅ Model loaded: **{selected_model}**")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Image Detection", "🎥 Video Detection", 
                                       "📹 Webcam Detection", "📊 Batch Analysis"])
    
    # Tab 1: Image Detection
    with tab1:
        st.header("📸 Image Detection")
        st.markdown("Upload an image to detect road anomalies")
        
        uploaded_file = st.file_uploader("Choose an image...", 
                                         type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                                         key="image_upload")
        
        col1, col2 = st.columns([1, 1])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Detect button
            if st.button("🔍 Detect Anomalies", type="primary", use_container_width=True):
                with st.spinner("🔄 Detecting..."):
                    detections, inference_time = detector.detect_image(
                        model, image_bgr, conf_threshold, iou_threshold
                    )
                    
                    # Draw detections
                    if detections:
                        result_image = detector.draw_detections(image_bgr, detections)
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    else:
                        result_image_rgb = image_np
                
                with col2:
                    st.subheader("Detection Results")
                    st.image(result_image_rgb, use_container_width=True)
                
                # Statistics
                st.divider()
                
                # Metrics row
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Total Detections", len(detections))
                
                with metric_cols[1]:
                    st.metric("Inference Time", f"{inference_time:.1f} ms")
                
                with metric_cols[2]:
                    high_conf = sum(1 for d in detections if d['confidence'] > 0.7)
                    st.metric("High Confidence", high_conf)
                
                with metric_cols[3]:
                    if detections:
                        avg_conf = np.mean([d['confidence'] for d in detections])
                        st.metric("Avg Confidence", f"{avg_conf:.3f}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                
                # Detailed results
                if detections:
                    st.divider()
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        fig = detector.create_statistics_chart(detections)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with chart_col2:
                        fig = detector.create_confidence_distribution(detections)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("🔍 Detection Details")
                    
                    # Create table
                    det_df = pd.DataFrame([{
                        'Class': d['class_name'],
                        'Confidence': f"{d['confidence']:.3f}",
                        'BBox': f"[{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]",
                        'Color': d['color']
                    } for d in detections])
                    
                    st.dataframe(det_df, use_container_width=True)
                    
                    # Download results
                    st.divider()
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        # Download annotated image
                        result_pil = Image.fromarray(result_image_rgb)
                        buf = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                        result_pil.save(buf.name, 'JPEG')
                        
                        with open(buf.name, 'rb') as f:
                            st.download_button(
                                label="📥 Download Annotated Image",
                                data=f,
                                file_name=f"detected_{uploaded_file.name}",
                                mime="image/jpeg",
                                use_container_width=True
                            )
                    
                    with col_dl2:
                        # Download JSON
                        results_json = json.dumps({
                            'image': uploaded_file.name,
                            'timestamp': datetime.now().isoformat(),
                            'detections': detections,
                            'inference_time_ms': inference_time
                        }, indent=2)
                        
                        st.download_button(
                            label="📥 Download JSON Results",
                            data=results_json,
                            file_name=f"results_{uploaded_file.name}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                else:
                    st.info("✨ No anomalies detected in this image!")
    
    # Tab 2: Video Detection
    with tab2:
        st.header("🎥 Video Detection")
        st.markdown("Upload a video to detect road anomalies")
        
        uploaded_video = st.file_uploader("Choose a video...", 
                                          type=['mp4', 'avi', 'mov', 'mkv'],
                                          key="video_upload")
        
        if uploaded_video is not None:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            st.video(video_path)
            
            if st.button("🔍 Process Video", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("🔄 Processing video..."):
                    output_path, detections, total_frames = detector.process_video(
                        model, video_path, conf_threshold, iou_threshold,
                        progress_bar, status_text
                    )
                
                status_text.text(f"✅ Processed {total_frames} frames")
                
                st.success("✅ Video processing complete!")
                
                # Show result video
                st.subheader("Annotated Video")
                st.video(output_path)
                
                # Statistics
                st.divider()
                
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Total Frames", total_frames)
                
                with metric_cols[1]:
                    st.metric("Total Detections", len(detections))
                
                with metric_cols[2]:
                    frames_with_det = len(set(i for i, _ in enumerate(detections)))
                    st.metric("Frames w/ Detections", frames_with_det)
                
                with metric_cols[3]:
                    if detections:
                        avg_conf = np.mean([d['confidence'] for d in detections])
                        st.metric("Avg Confidence", f"{avg_conf:.3f}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                
                if detections:
                    st.divider()
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        fig = detector.create_statistics_chart(detections)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with chart_col2:
                        fig = detector.create_confidence_distribution(detections)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button
                    st.divider()
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=f,
                            file_name=f"detected_{uploaded_video.name}",
                            mime="video/mp4",
                            use_container_width=True
                        )
    
    # Tab 3: Webcam Detection
    with tab3:
        st.header("📹 Webcam Detection")
        st.markdown("Real-time detection from your webcam")
        
        st.info("👉 Configure settings and click 'Capture & Detect' to start")
        
        # Webcam controls (simplified with default values)
        with st.expander("⚙️ Camera Settings (Optional)", expanded=False):
            camera_id = st.number_input("Camera ID", min_value=0, max_value=10, value=0, step=1,
                                       help="Usually 0 for default camera, try 1 or 2 if not working")
            st.info("💡 Default Camera ID is 0 (built-in camera). Change only if needed.")
        
        st.divider()
        
        # Capture and process button
        if st.button("📸 Capture & Detect", type="primary", use_container_width=True):
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                st.error(f"❌ Failed to open camera {camera_id}")
                st.info("💡 Troubleshooting tips:")
                st.markdown("""
                - Try different Camera IDs (0, 1, 2...)
                - Check if another app is using the camera
                - Ensure camera permissions are granted
                - Connect an external webcam if built-in doesn't work
                """)
            else:
                st.success(f"✅ Camera {camera_id} opened successfully")
                
                # Create columns for live view
                col_frame, col_stats = st.columns([2, 1])
                
                with col_frame:
                    st.subheader("Live Detection")
                    frame_placeholder = st.empty()
                
                with col_stats:
                    st.subheader("Statistics")
                    metric_placeholder = st.empty()
                    chart_placeholder = st.empty()
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Statistics
                all_detections = []
                frame_times = []
                frame_idx = 0
                
                # Add stop button placeholder
                st.info("🛑 Close the browser tab or refresh the page to stop detection")
                
                try:
                    # Continuous detection until camera fails or user stops
                    while True:
                        frame_idx += 1
                        progress_bar.progress(min(frame_idx / 100, 1.0))  # Continuous progress
                        status_text.text(f"🔴 Live Detection - Frame {frame_idx}")
                        
                        # Read frame
                        ret, frame = cap.read()
                        if not ret:
                            st.warning(f"⚠️ Could not read frame {frame_idx}")
                            break
                        
                        # Detect
                        start_time = time.time()
                        detections, inference_time = detector.detect_image(
                            model, frame, conf_threshold, iou_threshold
                        )
                        frame_times.append(inference_time)
                        
                        # Store detections
                        all_detections.extend(detections)
                        
                        # Draw detections
                        if detections:
                            frame = detector.draw_detections(frame, detections)
                        
                        # Add overlay info
                        info_text = f"Frame: {frame_idx} | Detections: {len(detections)}"
                        cv2.putText(frame, info_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        inference_text = f"Inference: {inference_time:.1f}ms | FPS: {1000/inference_time:.1f}"
                        cv2.putText(frame, inference_text, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Add LIVE indicator
                        cv2.putText(frame, "LIVE", (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Convert and display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        # Update statistics (every 10 frames for performance)
                        if frame_idx % 10 == 0:
                            with metric_placeholder.container():
                                m1, m2 = st.columns(2)
                                with m1:
                                    st.metric("Total Detections", len(all_detections))
                                with m2:
                                    avg_fps = 1000 / np.mean(frame_times[-30:]) if frame_times else 0
                                    st.metric("Current FPS", f"{avg_fps:.1f}")
                        
                        # Small delay for display (adjust based on performance)
                        time.sleep(0.01)  # Minimal delay for smooth streaming
                
                except Exception as e:
                    st.error(f"❌ Error during capture: {e}")
                
                finally:
                    cap.release()
                
                status_text.text("✅ Capture complete!")
                
                # Final results
                st.divider()
                st.subheader("📊 Session Summary")
                
                summary_cols = st.columns(4)
                
                with summary_cols[0]:
                    st.metric("Frames Processed", frame_idx)
                
                with summary_cols[1]:
                    st.metric("Total Detections", len(all_detections))
                
                with summary_cols[2]:
                    if frame_times:
                        avg_inference = np.mean(frame_times)
                        st.metric("Avg Inference", f"{avg_inference:.1f}ms")
                    else:
                        st.metric("Avg Inference", "N/A")
                
                with summary_cols[3]:
                    if frame_times:
                        avg_fps = 1000 / np.mean(frame_times)
                        st.metric("Avg FPS", f"{avg_fps:.1f}")
                    else:
                        st.metric("Avg FPS", "N/A")
                
                # Charts
                if all_detections:
                    st.divider()
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        fig = detector.create_statistics_chart(all_detections)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with chart_col2:
                        fig = detector.create_confidence_distribution(all_detections)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Detection details
                    st.subheader("🔍 Detection Details")
                    det_df = pd.DataFrame([{
                        'Class': d['class_name'],
                        'Confidence': f"{d['confidence']:.3f}",
                        'Color': d['color']
                    } for d in all_detections])
                    st.dataframe(det_df, use_container_width=True)
                else:
                    st.info("✨ No anomalies detected during this session")
        
        else:
            # Instructions when not running
            st.info("👆 Configure settings and click 'Capture & Detect' to start")
            
            with st.expander("ℹ️ Webcam Usage Guide"):
                st.markdown("""
                ### How to Use:
                1. **Select Camera ID**: Usually 0 for default camera
                2. **Set Max Frames**: How many frames to process (e.g., 100 = ~3-4 seconds)
                3. **Click 'Capture & Detect'**: Start real-time detection
                4. **View Results**: See live detections and statistics
                
                ### Tips for Best Results:
                - ✅ Ensure adequate lighting
                - ✅ Position camera to capture road surface clearly  
                - ✅ Close other applications using the camera
                - ✅ Start with max frames = 50 to test, then increase
                - ✅ Use lower confidence threshold to catch more anomalies
                
                ### Troubleshooting:
                - **Camera not opening?** Try Camera ID 1, 2, or 3
                - **Permission denied?** Check system/browser camera permissions
                - **Slow performance?** Reduce max frames or increase confidence threshold
                - **No detections?** Lower confidence threshold or improve lighting
                
                ### Alternative Options:
                For continuous real-time detection with better performance, use the CLI script:
                ```bash
                python realtime_detection.py --model optimized_models/yolov8s_best.pt --source 0
                ```
                
                This provides:
                - ✨ Unlimited continuous detection
                - ✨ Recording capabilities (press R)
                - ✨ Better performance
                - ✨ Keyboard controls
                """)
            
            st.markdown("---")
            st.info("💡 **Pro Tip:** The CLI real-time script offers better performance for extended monitoring sessions")
    
    # Tab 4: Batch Analysis
    with tab4:
        st.header("📊 Batch Analysis")
        st.markdown("Upload multiple images for batch processing")
        
        uploaded_files = st.file_uploader("Choose images...", 
                                          type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                                          accept_multiple_files=True,
                                          key="batch_upload")
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} images uploaded")
            
            if st.button("🔍 Process Batch", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_results = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    # Read image
                    image = Image.open(uploaded_file)
                    image_np = np.array(image)
                    
                    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    else:
                        image_bgr = image_np
                    
                    # Detect
                    detections, inference_time = detector.detect_image(
                        model, image_bgr, conf_threshold, iou_threshold
                    )
                    
                    all_results.append({
                        'filename': uploaded_file.name,
                        'num_detections': len(detections),
                        'detections': detections,
                        'inference_time': inference_time
                    })
                
                status_text.text("✅ Batch processing complete!")
                
                st.success(f"✅ Processed {len(uploaded_files)} images")
                
                # Summary statistics
                st.divider()
                st.subheader("📊 Batch Summary")
                
                total_detections = sum(r['num_detections'] for r in all_results)
                images_with_det = sum(1 for r in all_results if r['num_detections'] > 0)
                avg_inference = np.mean([r['inference_time'] for r in all_results])
                
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Total Images", len(uploaded_files))
                
                with metric_cols[1]:
                    st.metric("Total Detections", total_detections)
                
                with metric_cols[2]:
                    st.metric("Images w/ Detections", images_with_det)
                
                with metric_cols[3]:
                    st.metric("Avg Inference", f"{avg_inference:.1f} ms")
                
                # Results table
                st.divider()
                st.subheader("📋 Detailed Results")
                
                results_df = pd.DataFrame([{
                    'Filename': r['filename'],
                    'Detections': r['num_detections'],
                    'Inference (ms)': f"{r['inference_time']:.1f}"
                } for r in all_results])
                
                st.dataframe(results_df, use_container_width=True)
                
                # Overall statistics
                all_detections = []
                for r in all_results:
                    all_detections.extend(r['detections'])
                
                if all_detections:
                    st.divider()
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        fig = detector.create_statistics_chart(all_detections)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with chart_col2:
                        fig = detector.create_confidence_distribution(all_detections)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
