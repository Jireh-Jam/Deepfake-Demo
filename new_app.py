import streamlit as st
import torch
from PIL import Image, ImageDraw
import tempfile
import os
import pandas as pd
import numpy as np
from deepfake_detector import DeepfakeDetectionSystem
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import onnx
import onnxruntime
import numpy as np
from PIL import Image
import face_alignment
from torchvision import transforms
import streamlit as st
from face_alignment.api import FaceAlignment, LandmarksType

import streamlit as st
import onnxruntime
import numpy as np
from PIL import Image
import tempfile
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
import face_alignment
from face_alignment.api import FaceAlignment, LandmarksType
from io import BytesIO
import base64

# Configure page
st.set_page_config(
    page_title="Demo: Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Create tick and cross images programmatically
def create_overlay_images():
    # Create green tick
    tick_img = Image.new('RGBA', (40, 40), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tick_img)
    draw.line([(8, 20), (16, 28), (32, 12)], fill=(0, 255, 0), width=4)

    # Create red cross
    cross_img = Image.new('RGBA', (40, 40), (0, 0, 0, 0))
    draw = ImageDraw.Draw(cross_img)
    draw.line([(10, 10), (30, 30)], fill=(255, 0, 0), width=4)
    draw.line([(10, 30), (30, 10)], fill=(255, 0, 0), width=4)

    return tick_img, cross_img


# Complete Custom CSS styles
st.markdown("""
    <style>
    /* Main background with AI-inspired gradient */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0f3460 0%, #162447 100%);
        color: #ffffff;
    }

    /* Warning message styling */
    .warning-message {
        background-color: rgba(255, 87, 51, 0.1);
        border-left: 4px solid #ff5733;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
        color: #ffffff;
    }

    /* Result message styling */
    .result-message {
        background-color: rgba(66, 103, 178, 0.1);
        border-left: 4px solid #4267B2;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
        color: #ffffff;
    }

    /* Card styling */
    .analysis-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Status card styling */
    .status-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }

    .status-real {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(76, 175, 80, 0.2) 100%);
        border: 1px solid rgba(76, 175, 80, 0.3);
    }

    .status-fake {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(244, 67, 54, 0.2) 100%);
        border: 1px solid rgba(244, 67, 54, 0.3);
    }

    /* Metric styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #4CAF50;
        text-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
    }

    .metric-label {
        color: #b0b0b0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }

    /* Title styling */
    h1, h2, h3 {
        color: #b211f7;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
    }

    /* Image container styling */
    .image-container {
        position: relative;
        display: inline-block;
    }

    .image-overlay {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 30px;
        height: 30px;
    }

    /* Educational content styling */
    .edu-content {
        background: rgba(255, 255, 255, 0.03);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        color: #e0e0e0;
    }

    .edu-content h2 {
        color: #b211f7;
        margin-bottom: 1.5rem;
        font-size: 1.8rem;
    }

    .edu-content h3 {
        color: #b211f7;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }

    .edu-content p {
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    .edu-content ul {
        margin-left: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .edu-content li {
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }

    /* Warning section in educational content */
    .edu-warning {
        background: rgba(255, 87, 51, 0.1);
        border-left: 4px solid #ff5733;
        padding: 1rem;
        margin: 1.5rem 0;
        border-radius: 0 10px 10px 0;
    }

    /* Responsive adjustments */
    @media (max-width: 1024px) {
        .main-container {
            grid-template-columns: 1fr;
        }
    }

    /* Enhanced image analysis section */
    .image-analysis {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 2rem;
        margin-bottom: 2rem;
    }

    .image-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
    }

    .image-title {
        color: #fff;
        text-align: center;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Model path
MODEL_PATH = r"/deepfake_detector.onnx"

# Create overlay images
TICK_IMG, CROSS_IMG = create_overlay_images()


class ONNXDeepfakeDetectionSystem:
    def __init__(self, model_path):
        try:
            print("Initializing ONNXDeepfakeDetectionSystem...")

            # Configure session options
            session_options = onnxruntime.SessionOptions()
            session_options.enable_cpu_mem_arena = False
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1

            # Initialize ONNX Runtime session
            self.session = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=['CPUExecutionProvider']
            )
            print("ONNX model loaded successfully")

            # Initialize face alignment
            print("Initializing face alignment...")
            self.fa = FaceAlignment(
                landmarks_type=LandmarksType.TWO_D,
                device='cpu',
                flip_input=False
            )
            print("Face alignment initialized")

            # Define transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def align_face(self, image):
        """Detect and align face in image"""
        try:
            if isinstance(image, str):
                image = Image.open(image)
            if isinstance(image, Image.Image):
                image = np.array(image)

            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if isinstance(image, np.ndarray) else image
            else:
                image_rgb = image

            landmarks = self.fa.get_landmarks(image_rgb)
            if not landmarks or len(landmarks) == 0:
                print("No face detected")
                return None, None

            landmark = landmarks[0]
            margin = 30
            left = max(0, int(np.min(landmark[:, 0])) - margin)
            right = min(image.shape[1], int(np.max(landmark[:, 0])) + margin)
            top = max(0, int(np.min(landmark[:, 1])) - margin)
            bottom = min(image.shape[0], int(np.max(landmark[:, 1])) + margin)

            face = image[top:bottom, left:right]
            face = Image.fromarray(face)

            return face, (left, top, right, bottom)

        except Exception as e:
            print(f"Error in face alignment: {str(e)}")
            return None, None

    def predict(self, image):
        """Predict if an image is fake or real"""
        try:
            face, bbox = self.align_face(image)
            if face is None:
                return None, None, None

            x = self.transform(face).numpy()
            x = np.expand_dims(x, 0)

            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: x})
            logits = output[0][0]

            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            FAKE_THRESHOLD = 0.298
            pred = 1 if probs[1] >= FAKE_THRESHOLD else 0
            conf = probs[1] if pred == 1 else probs[0]

            return pred, conf, bbox

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None, None, None


def display_educational_content():
    st.markdown(
        """
        <div class="edu-content">
            <h2>Understanding Deepfakes</h2>
            <p>Deepfakes are synthetic media created using artificial intelligence to replace or manipulate faces, voices, and other elements in images and videos.</p>
            <h3><span>🔍</span> How to Spot Deepfakes</h3>
            <p>No AI detection tool is perfect. This system should be used as just one part of your verification process:</p>
            <ul>
                <li>Unnatural blinking patterns or facial expressions</li>
                <li>Inconsistent lighting and shadows</li>
                <li>Blurry or misaligned facial features</li>
                <li>Unnatural hair textures or skin boundaries</li>
                <li>Artifacts around eyes, teeth, or jewelry</li>
            </ul>
            <h3><span>⚠️</span> Important Warning</h3>
            <p>No AI detection tool is perfect. This system should be used as just one part of your verification process:</p>
            <ul>
                <li>Always verify content through multiple sources</li>
                <li>Consult professional fact-checkers when possible</li>
                <li>If in doubt, don't share the content</li>
                <li>Report suspicious content to platform moderators</li>
            </ul>
            <h3><span>🛡️</span> Best Practices</h3>
            <p>No AI detection tool is perfect. This system should be used as just one part of your verification process:</p>
            <ul>
                <li>Stay updated on deepfake technology developments</li>
                <li>Use multiple verification tools</li>
                <li>Consider context and source credibility</li>
                <li>When uncertain, err on the side of caution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def display_model_metrics():
    """Display model performance metrics"""
    st.markdown("""
        <div class="analysis-section">
            <h2>Model Performance Metrics</h2>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
                <div class="metric-card">
                    <div class="metric-value">94.2%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">92.8%</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">93.5%</div>
                    <div class="metric-label">Sensitivity</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">93.1%</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def calculate_reliability(conf):
    """Calculate reliability and create visualization"""
    if conf >= 0.5:  # REAL prediction
        prediction = "REAL"
        if conf >= 0.7:
            reliability = "High"
            color = "#4CAF50"
            message = "Strongly indicates REAL"
        else:
            reliability = "Medium"
            color = "#FFA726"
            message = "Moderately indicates REAL"
    else:  # FAKE prediction
        prediction = "FAKE"
        if conf <= 0.39:
            reliability = "High"
            color = "#4CAF50"
            message = "Strongly indicates FAKE"
        else:
            reliability = "Medium"
            color = "#FFA726"
            message = "Moderately indicates FAKE"

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 2))

    regions = [
        ("Strong Fake", 0.0, 0.39, "red"),
        ("Moderate Fake", 0.39, 0.5, "lightcoral"),
        ("Moderate Real", 0.5, 0.7, "lightgreen"),
        ("Strong Real", 0.7, 1.0, "green")
    ]

    for name, start, end, color in regions:
        ax.axvspan(start, end, alpha=0.3, color=color, label=name)

    ax.axvline(x=conf, color='blue', linewidth=2, label='Current Score')
    ax.set_xlim(0, 1)
    ax.set_title('Confidence Score Gauge')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    return reliability, color, message, fig


def load_model():
    """Load the ONNX model"""
    try:
        detector = ONNXDeepfakeDetectionSystem(MODEL_PATH)
        return detector
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return None


def image_analysis(detector):
    """Handle image analysis mode"""
    st.session_state.mode = 'image'
    st.title("🔍 Demo: Deepfake Detection System")

    # Sidebar controls
    st.sidebar.subheader("Analysis Controls")
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        # Create main content columns
        left_col, right_col = st.columns([0.7, 0.3])

        with left_col:
            # Create two sub-columns for original and analyzed images
            img_col1, img_col2 = st.columns(2)

            with img_col1:
                st.markdown("<h3 style='text-align: center;'>Original Image</h3>", unsafe_allow_html=True)
                image = Image.open(uploaded_file)
                # Resize image while maintaining aspect ratio
                basewidth = 300
                wpercent = (basewidth / float(image.size[0]))
                hsize = int((float(image.size[1]) * float(wpercent)))
                image_resized = image.resize((basewidth, hsize), Image.LANCZOS)
                st.image(image_resized, use_container_width=True)

            # Analysis button in sidebar
            if st.sidebar.button("🔎 Analyze Image", key="analyze_btn"):
                with st.spinner("Processing..."):
                    pred, conf, bbox = detector.predict(image)

                    if pred is None:
                        st.error("⚠️ No face detected in the image!")
                    else:
                        with img_col2:
                            st.markdown("<h3 style='text-align: center;'>Detected Face</h3>", unsafe_allow_html=True)
                            face, _ = detector.align_face(image)
                            if face:
                                # Resize face image
                                wpercent = (basewidth / float(face.size[0]))
                                hsize = int((float(face.size[1]) * float(wpercent)))
                                face_resized = face.resize((basewidth, hsize), Image.LANCZOS)

                                # Add overlay
                                is_fake = conf < 0.5
                                overlay = CROSS_IMG if is_fake else TICK_IMG
                                face_resized.paste(overlay, (face_resized.width - 50, 10), overlay)

                                st.image(face_resized, use_container_width=True)

                        # Analysis results below images
                        st.markdown("<div class='analysis-section'>", unsafe_allow_html=True)

                        # Metrics in columns
                        met_col1, met_col2, met_col3 = st.columns(3)
                        with met_col1:
                            st.metric("Confidence", f"{conf:.1%}")
                        with met_col2:
                            reliability, color, message, _ = calculate_reliability(conf)
                            st.metric("Reliability", reliability)
                        with met_col3:
                            st.metric("Prediction", "FAKE" if conf < 0.5 else "REAL")

                        # Detailed results
                        st.subheader("Analysis Details")
                        st.write(f"Interpretation: {message}")

                        # Display confidence gauge
                        _, _, _, fig = calculate_reliability(conf)
                        st.pyplot(fig)

                        st.markdown("</div>", unsafe_allow_html=True)

                        # Display model metrics
                        display_model_metrics()

        # Educational content in right column
        with right_col:
            display_educational_content()


def analyze_video_frame(conf):
    """Analyze a single frame's confidence score"""
    if conf >= 0.5:  # REAL prediction
        prediction = "REAL"
        if conf >= 0.7:
            reliability = "High"
            message = "Strong Real"
        else:
            reliability = "Medium"
            message = "Moderate Real"
    else:  # FAKE prediction
        prediction = "FAKE"
        if conf <= 0.39:
            reliability = "High"
            message = "Strong Fake"
        else:
            reliability = "Medium"
            message = "Moderate Fake"

    return prediction, reliability, message


def create_confidence_plots(df):
    """Create distribution and timeline plots"""
    # Set style
    sns.set_style("darkgrid")
    plt.style.use("dark_background")

    # Create distribution plot
    fig_dist, ax_dist = plt.subplots(figsize=(10, 2))

    # Plot regions
    ax_dist.axvspan(0, 0.39, alpha=0.3, color='red', label='Strong Fake')
    ax_dist.axvspan(0.39, 0.5, alpha=0.3, color='lightcoral', label='Moderate Fake')
    ax_dist.axvspan(0.5, 0.7, alpha=0.3, color='lightgreen', label='Moderate Real')
    ax_dist.axvspan(0.7, 1.0, alpha=0.3, color='green', label='Strong Real')

    # Plot confidence distribution
    sns.kdeplot(data=df['confidence'], ax=ax_dist, color='white', linewidth=2)
    ax_dist.set_xlim(0, 1)
    ax_dist.set_title('Confidence Distribution')
    ax_dist.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Create timeline plot
    fig_time, ax_time = plt.subplots(figsize=(12, 4))
    ax_time.axhspan(0, 0.39, alpha=0.2, color='red', label='Strong Fake')
    ax_time.axhspan(0.39, 0.5, alpha=0.2, color='lightcoral', label='Moderate Fake')
    ax_time.axhspan(0.5, 0.7, alpha=0.2, color='lightgreen', label='Moderate Real')
    ax_time.axhspan(0.7, 1.0, alpha=0.2, color='green', label='Strong Real')

    ax_time.plot(df['frame'], df['confidence'], color='white', linewidth=2)
    ax_time.set_ylabel('Confidence Score')
    ax_time.set_xlabel('Frame Number')
    ax_time.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_time.set_title('Confidence Timeline')

    return fig_dist, fig_time


def video_analysis(detector):
    """Handle video analysis mode"""
    st.session_state.mode = 'video'
    st.title("🎥 Demo: Deepfake Detection System")

    # Create two columns for main content
    main_col, edu_col = st.columns([0.7, 0.3])

    with main_col:
        # Video upload and controls
        uploaded_file = st.file_uploader("Upload a video", type=['mp4'])

        if uploaded_file:
            st.video(uploaded_file)

            # Analysis controls
            frame_sample_rate = st.slider(
                "Frame Sampling Rate",
                min_value=1,
                max_value=60,
                value=30,
                help="Analyze every Nth frame"
            )

            if st.button("🔎 Analyze Video"):
                try:
                    with st.spinner("Processing video..."):
                        # Save uploaded video temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            video_path = tmp_file.name

                        results = []
                        frames = {}

                        # Process video
                        cap = cv2.VideoCapture(video_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_count = 0

                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break

                            if frame_count % frame_sample_rate == 0:
                                # Update progress
                                progress = int((frame_count / total_frames) * 100)
                                progress_bar.progress(progress)
                                status_text.text(f"Analyzing frame {frame_count}/{total_frames}")

                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                pred, conf, bbox = detector.predict(frame_rgb)

                                if pred is not None:
                                    # Analyze frame
                                    prediction, reliability, message = analyze_video_frame(conf)

                                    results.append({
                                        'frame': frame_count,
                                        'prediction': prediction,
                                        'confidence': conf,
                                        'reliability': reliability,
                                        'interpretation': message
                                    })

                                    # Store sample frames
                                    if len(frames) < 5:
                                        frames[frame_count] = frame_rgb

                            frame_count += 1

                        cap.release()
                        progress_bar.empty()
                        status_text.empty()

                        if results:
                            df = pd.DataFrame(results)

                            # Calculate statistics
                            strong_fake = len(df[df['confidence'] <= 0.39])
                            strong_real = len(df[df['confidence'] >= 0.7])
                            total_frames = len(results)
                            avg_conf = df['confidence'].mean()

                            # Display metrics
                            st.subheader("Analysis Results")
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Strong Fake Frames", f"{strong_fake / total_frames:.1%}")
                            with col2:
                                st.metric("Strong Real Frames", f"{strong_real / total_frames:.1%}")
                            with col3:
                                st.metric("Average Confidence", f"{avg_conf:.3f}")
                            with col4:
                                st.metric("Frames Analyzed", total_frames)

                            # Display plots
                            fig_dist, fig_time = create_confidence_plots(df)

                            st.subheader("Confidence Distribution")
                            st.pyplot(fig_dist)

                            st.subheader("Confidence Timeline")
                            st.pyplot(fig_time)

                            # Display sample frames
                            if frames:
                                st.subheader("Sample Frames")
                                for frame_num, frame in frames.items():
                                    frame_result = df[df['frame'] == frame_num].iloc[0]

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.image(frame, caption=f"Frame {frame_num}")
                                    with col2:
                                        st.markdown(f"""
                                            Prediction: {frame_result['prediction']}<br>
                                            Confidence: {frame_result['confidence']:.3f}<br>
                                            Reliability: {frame_result['reliability']}
                                        """, unsafe_allow_html=True)

                            # Detailed results
                            st.subheader("Detailed Analysis")
                            if st.checkbox("Show frame-by-frame results"):
                                st.dataframe(df)
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "📊 Download Analysis Results",
                                    csv,
                                    "video_analysis.csv",
                                    "text/csv"
                                )
                        else:
                            st.error("No faces detected in the video!")

                except Exception as e:
                    st.error(f"Error analyzing video: {str(e)}")
                finally:
                    if 'video_path' in locals():
                        os.unlink(video_path)

    # Educational content in right column
    with edu_col:
        display_educational_content()
        display_model_metrics()


def main():
    # Initialize detector
    detector = load_model()
    if detector is None:
        st.error("Failed to load model. Please check the error message above.")
        return

    # Mode selection
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Mode", ["Image Analysis", "Video Analysis"])

    if mode == "Image Analysis":
        image_analysis(detector)
    else:
        video_analysis(detector)


if __name__ == "__main__":
    main()
