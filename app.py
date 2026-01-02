"""
Streamlit-based Interactive Deepfake Detection Application
Provides frame-by-frame analysis and explainable AI visualization.
"""

import warnings
import os

# Suppress warnings that don't affect functionality
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress Streamlit file watcher errors with PyTorch (known issue)
os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import pandas as pd
import io
from typing import Optional, List, Dict
import time

# Import utilities
from utils.face_utils import FaceDetector
from utils.video_utils import extract_frames, get_video_info, resize_frame
from utils.inference import (
    load_model, preprocess_face, predict_single_face,
    predict_batch, aggregate_video_predictions
)
import config

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .verdict-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .verdict-fake {
        background-color: #ffebee;
        color: #c62828;
        border: 3px solid #c62828;
    }
    .verdict-real {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }
    .confidence-badge {
        font-size: 1.5rem;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .frame-analysis {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detection_model(model_path: str):
    """
    Load and cache the deepfake detection model.
    
    Args:
        model_path: Path to model checkpoint
    
    Returns:
        Loaded model and device
    """
    if not Path(model_path).exists():
        st.error(f"Model not found at {model_path}. Please train a model first or update the path.")
        st.stop()
    
    with st.spinner("Loading deepfake detection model..."):
        model, device = load_model(model_path)
        use_amp = config.MIXED_PRECISION and device.type == 'cuda'
    
    return model, device, use_amp


def process_image(uploaded_file, model, device, use_amp, face_detector):
    """
    Process uploaded image and return predictions.
    
    Args:
        uploaded_file: Uploaded image file
        model: Trained model
        device: Device for inference
        use_amp: Use mixed precision
        face_detector: Face detector instance
    
    Returns:
        Dictionary with results
    """
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Detect and crop face
    with st.spinner("Detecting face..."):
        cropped_face, bbox = face_detector.detect_and_crop(image)
    
    if cropped_face is None:
        st.warning("⚠️ No face detected in the image. Please upload an image with a visible face.")
        return None
    
    # Preprocess and predict
    with st.spinner("Running inference..."):
        face_tensor = preprocess_face(cropped_face)
        prediction = predict_single_face(model, face_tensor, device, use_amp)
    
    return {
        'original_image': image,
        'cropped_face': cropped_face,
        'prediction': prediction
    }


def process_video(uploaded_file, model, device, use_amp, face_detector, 
                 fps: float = None, show_only_fake: bool = False, 
                 min_confidence: float = 0.0):
    """
    Process uploaded video and return frame-by-frame predictions.
    
    Args:
        uploaded_file: Uploaded video file
        model: Trained model
        device: Device for inference
        use_amp: Use mixed precision
        face_detector: Face detector instance
        fps: Frames per second to extract
        show_only_fake: Show only frames predicted as fake
        min_confidence: Minimum confidence to display frame
    
    Returns:
        Dictionary with results
    """
    # Save uploaded file temporarily
    temp_path = Path("temp_video.mp4")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    try:
        # Get video info
        video_info = get_video_info(str(temp_path))
        if video_info is None:
            st.error("Could not read video file.")
            return None
        
        st.info(f"📹 Video Info: {video_info['width']}x{video_info['height']}, "
                f"{video_info['duration']:.2f}s, {video_info['fps']:.2f} FPS")
        
        # Extract frames
        with st.spinner("Extracting frames..."):
            frames = extract_frames(str(temp_path), fps=fps)
        
        if len(frames) == 0:
            st.error("No frames extracted from video.")
            return None
        
        # Process frames
        frame_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (frame_num, frame) in enumerate(frames):
            status_text.text(f"Processing frame {idx + 1}/{len(frames)}...")
            progress_bar.progress((idx + 1) / len(frames))
            
            # Detect face
            cropped_face, bbox = face_detector.detect_face_in_frame(frame)
            
            if cropped_face is not None:
                # Predict
                face_tensor = preprocess_face(cropped_face)
                prediction = predict_single_face(model, face_tensor, device, use_amp)
                
                frame_results.append({
                    'frame_num': frame_num,
                    'frame': frame,
                    'cropped_face': cropped_face,
                    'prediction': prediction
                })
        
        progress_bar.empty()
        status_text.empty()
        
        if len(frame_results) == 0:
            st.warning("⚠️ No faces detected in any frames. Please upload a video with visible faces.")
            return None
        
        # Aggregate predictions
        video_prediction = aggregate_video_predictions(
            frame_results,
            method=config.AGGREGATION_METHOD
        )
        
        # Filter frames if requested
        if show_only_fake:
            frame_results = [
                r for r in frame_results 
                if r['prediction']['prediction'] == 1 and 
                   r['prediction']['confidence'] >= min_confidence
            ]
        
        return {
            'video_info': video_info,
            'frame_results': frame_results,
            'video_prediction': video_prediction
        }
    
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


def display_image_results(results: Dict):
    """Display results for image input."""
    st.subheader("📸 Image Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image**")
        st.image(results['original_image'], use_container_width=True)
    
    with col2:
        st.markdown("**Detected Face (224×224)**")
        st.image(results['cropped_face'], use_container_width=True)
    
    # Prediction
    pred = results['prediction']
    verdict = "FAKE" if pred['prediction'] == 1 else "REAL"
    confidence = pred['confidence'] * 100
    
    # Display verdict
    color = "🔴" if verdict == "FAKE" else "🟢"
    verdict_class = "verdict-fake" if verdict == "FAKE" else "verdict-real"
    
    st.markdown(f"""
    <div class="verdict-box {verdict_class}">
        {color} FINAL VERDICT: {verdict}<br>
        <span style="font-size: 1.5rem;">Confidence: {confidence:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed probabilities
    with st.expander("📊 Detailed Probabilities", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Real Probability", f"{pred['real_prob']*100:.2f}%")
        with col2:
            st.metric("Fake Probability", f"{pred['fake_prob']*100:.2f}%")
        
        # Progress bars
        st.progress(pred['real_prob'], text="Real")
        st.progress(pred['fake_prob'], text="Fake")


def display_video_results(results: Dict):
    """Display results for video input."""
    st.subheader("🎬 Video Analysis Results")
    
    # Video-level verdict
    vp = results['video_prediction']
    verdict = vp['verdict']
    confidence = vp['confidence'] * 100
    
    color = "🔴" if verdict == "FAKE" else "🟢"
    verdict_class = "verdict-fake" if verdict == "FAKE" else "verdict-real"
    
    st.markdown(f"""
    <div class="verdict-box {verdict_class}">
        {color} FINAL VERDICT: {verdict}<br>
        <span style="font-size: 1.5rem;">Confidence: {confidence:.2f}%</span><br>
        <span style="font-size: 1rem;">Frames Analyzed: {vp['num_frames']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary statistics
    with st.expander("📊 Video Summary Statistics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames", vp['num_frames'])
        with col2:
            st.metric("Avg Real Prob", f"{vp['real_prob']*100:.2f}%")
        with col3:
            st.metric("Avg Fake Prob", f"{vp['fake_prob']*100:.2f}%")
        with col4:
            fake_frames = sum(1 for r in results['frame_results'] if r['prediction']['prediction'] == 1)
            st.metric("Fake Frames", f"{fake_frames}/{vp['num_frames']}")
    
    # Frame-by-frame analysis
    st.subheader("🎞️ Frame-by-Frame Analysis")
    
    frame_results = results['frame_results']
    
    if len(frame_results) == 0:
        st.info("No frames to display based on current filters.")
        return
    
    # Create dataframe for frame analysis
    df_data = []
    for r in frame_results:
        df_data.append({
            'Frame #': r['frame_num'],
            'Real Prob': f"{r['prediction']['real_prob']*100:.2f}%",
            'Fake Prob': f"{r['prediction']['fake_prob']*100:.2f}%",
            'Prediction': 'FAKE' if r['prediction']['prediction'] == 1 else 'REAL',
            'Confidence': f"{r['prediction']['confidence']*100:.2f}%"
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Display frames in grid
    st.markdown("### Frame Visualizations")
    
    num_cols = 3
    for i in range(0, len(frame_results), num_cols):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            if i + j < len(frame_results):
                r = frame_results[i + j]
                with col:
                    st.markdown(f"**Frame #{r['frame_num']}**")
                    
                    # Resize frame for display
                    display_frame = resize_frame(r['frame'], max_size=300)
                    st.image(display_frame, use_container_width=True)
                    
                    # Show cropped face
                    st.image(r['cropped_face'], use_container_width=True, caption="Detected Face")
                    
                    # Prediction info
                    pred = r['prediction']
                    verdict_frame = "FAKE" if pred['prediction'] == 1 else "REAL"
                    conf_frame = pred['confidence'] * 100
                    
                    st.markdown(f"""
                    **Prediction:** {verdict_frame}  
                    **Confidence:** {conf_frame:.2f}%  
                    **Real:** {pred['real_prob']*100:.2f}% | **Fake:** {pred['fake_prob']*100:.2f}%
                    """)


def generate_report(results: Dict, input_type: str) -> str:
    """Generate text report of analysis."""
    report = []
    report.append("=" * 60)
    report.append("DEEPFAKE DETECTION REPORT")
    report.append("=" * 60)
    report.append("")
    
    if input_type == "image":
        report.append("Input Type: Image")
        report.append("")
        pred = results['prediction']
        verdict = "FAKE" if pred['prediction'] == 1 else "REAL"
        report.append(f"Verdict: {verdict}")
        report.append(f"Confidence: {pred['confidence']*100:.2f}%")
        report.append(f"Real Probability: {pred['real_prob']*100:.2f}%")
        report.append(f"Fake Probability: {pred['fake_prob']*100:.2f}%")
    
    else:  # video
        report.append("Input Type: Video")
        vp = results['video_prediction']
        report.append(f"Total Frames Analyzed: {vp['num_frames']}")
        report.append("")
        report.append(f"Final Verdict: {vp['verdict']}")
        report.append(f"Confidence: {vp['confidence']*100:.2f}%")
        report.append(f"Average Real Probability: {vp['real_prob']*100:.2f}%")
        report.append(f"Average Fake Probability: {vp['fake_prob']*100:.2f}%")
        report.append("")
        report.append("Frame-by-Frame Results:")
        report.append("-" * 60)
        for r in results['frame_results']:
            pred = r['prediction']
            verdict = "FAKE" if pred['prediction'] == 1 else "REAL"
            report.append(f"Frame #{r['frame_num']}: {verdict} "
                         f"(Real: {pred['real_prob']*100:.2f}%, "
                         f"Fake: {pred['fake_prob']*100:.2f}%)")
    
    report.append("")
    report.append("=" * 60)
    return "\n".join(report)


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<div class="main-header">🎭 Deepfake Detection System</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model path
        default_model_path = str(config.MODEL_DIR / "best_model.pth")
        model_path = st.text_input(
            "Model Path",
            value=default_model_path,
            help="Path to trained model checkpoint (.pth file)"
        )
        
        # Frame sampling rate (for videos)
        fps_slider = st.slider(
            "Frame Sampling Rate (FPS)",
            min_value=1,
            max_value=10,
            value=config.FPS,
            help="Frames per second to extract from videos"
        )
        
        # Filter options
        st.subheader("🔍 Filter Options")
        show_only_fake = st.checkbox(
            "Show only FAKE frames",
            value=False,
            help="Display only frames predicted as fake"
        )
        
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Minimum confidence to display frame (for fake frames filter)"
        )
    
    # Load model
    try:
        model, device, use_amp = load_detection_model(model_path)
        st.sidebar.success(f"✅ Model loaded on {device}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Initialize face detector
    face_detector = FaceDetector(device=device)
    
    # Main content area
    st.header("📤 Upload Media")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
        help="Upload an image (.jpg, .png) or video (.mp4, .avi, .mov) with visible faces"
    )
    
    if uploaded_file is not None:
        file_type = uploaded_file.type
        
        # Determine if image or video
        is_image = file_type.startswith('image/')
        is_video = file_type.startswith('video/') or uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov'))
        
        if is_image:
            st.info("📸 Processing image...")
            results = process_image(uploaded_file, model, device, use_amp, face_detector)
            
            if results is not None:
                display_image_results(results)
                
                # Download report
                report = generate_report(results, "image")
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name="deepfake_detection_report.txt",
                    mime="text/plain"
                )
        
        elif is_video:
            st.info("🎬 Processing video...")
            results = process_video(
                uploaded_file, model, device, use_amp, face_detector,
                fps=fps_slider, show_only_fake=show_only_fake,
                min_confidence=min_confidence
            )
            
            if results is not None:
                display_video_results(results)
                
                # Download report
                report = generate_report(results, "video")
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name="deepfake_detection_report.txt",
                    mime="text/plain"
                )
        
        else:
            st.error("Unsupported file type. Please upload an image (.jpg, .png) or video (.mp4, .avi, .mov).")
    
    else:
        # Instructions
        st.info("""
        👆 **Upload an image or video to get started!**
        
        ### How to use:
        1. **Upload Media**: Click "Browse files" and select an image or video
        2. **Wait for Processing**: The system will extract faces and run inference
        3. **View Results**: See frame-by-frame analysis and final verdict
        
        ### Supported Formats:
        - **Images**: .jpg, .jpeg, .png
        - **Videos**: .mp4, .avi, .mov
        
        ### Features:
        - 🎯 Frame-by-frame analysis
        - 📊 Confidence scores for each frame
        - 🎬 Video-level aggregation
        - 📥 Downloadable reports
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Deepfake Detection System | Built with Streamlit & PyTorch</p>
        <p>For research and educational purposes</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

