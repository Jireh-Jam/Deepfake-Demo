import streamlit as st
import torch
from PIL import Image, ImageDraw
import tempfile
import os
import pandas as pd
import numpy as np
# from deepfake_detector import DeepfakeDetectionSystem
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
MODEL_PATH = r"./deepfake_detector.onnx"

# Create overlay images
TICK_IMG, CROSS_IMG = create_overlay_images()


def ensure_rgb(image):
    """Convert RGBA image to RGB"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = Image.fromarray(image).convert("RGB")
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                image = Image.fromarray(image).convert("RGB")
            elif image.shape[2] == 3:
                image = Image.fromarray((cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    elif isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
    return image


class ONNXDeepfakeDetectionSystem:
    def __init__(self, model_path):
        self.min_image_size = 224  # Minimum size for reliable detection
        self.min_face_size = 64  # Minimum face size
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

    # def assess_image_quality(self, image):
    #     """ Assess image quality based on sharpness and brightness"""
    #     if isinstance(image,Image.Image):
    #         image = np.array(image)
    #
    #     issues = []
    #     # Check resolution
    #     if min(image.shape[:2]) < 224:
    #         issues.append("Low Resolution")
    #     # check contrast
    #     if np.std(image) < 20:
    #         issues.append("Low Contrast")
    #     # check for screenshot artifacts
    #     edges = cv2.Canny(image, 100, 200)
    #     edge_ratio = np.count_nonzero(edges)/edges.size
    #     if edge_ratio > 0.1:
    #         issues.append("Screenshot")
    #     #check for compression artifacts
    #     if len(np.unique(image)) < 1000:
    #         issues.append("Compression Artifacts")
    #     return issues

    def assess_image_quality(self, image):
        """Comprehensive image quality assessment"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        issues = []

        # Resolution check
        if min(image.shape[:2]) < self.min_image_size:
            issues.append("Low resolution (minimum 224x224 recommended)")

        # Contrast check
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        contrast = image_gray.std()
        if contrast < 20:
            issues.append("Low contrast - image may be too flat")

        # Brightness check
        brightness = np.mean(image_gray)
        if brightness < 40:
            issues.append("Image too dark")
        elif brightness > 215:
            issues.append("Image too bright")

        # Screenshot detection
        edges = cv2.Canny(image, 100, 200)
        edge_ratio = np.count_nonzero(edges) / edges.size
        if edge_ratio < 0.1:
            issues.append("Image appears to be a screenshot - use original image if possible")

        # Compression artifacts
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        if unique_colors < 5000:
            issues.append("Heavy compression detected - image quality may be compromised")

        # Blur detection
        laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            issues.append("Image appears blurry")

        # Add CSS styling for each issue type
        styled_issues = []
        for issue in issues:
            if "screenshot" in issue.lower():
                styled_issues.append(f"🖥️ {issue}")
            elif "compression" in issue.lower():
                styled_issues.append(f"🗜️ {issue}")
            elif "resolution" in issue.lower():
                styled_issues.append(f"📏 {issue}")
            elif "contrast" in issue.lower() or "bright" in issue.lower() or "dark" in issue.lower():
                styled_issues.append(f"💡 {issue}")
            elif "blurry" in issue.lower():
                styled_issues.append(f"🔍 {issue}")
            else:
                styled_issues.append(f"⚠️ {issue}")

        return styled_issues

    def align_face(self, image):
        """Enhanced face detection with RGB conversion"""
        try:
            # 1. Ensure RGB format
            image = ensure_rgb(image)
            if isinstance(image, Image.Image):
                image = np.array(image)

            print(f"Image shape after RGB conversion: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Image value range: [{image.min()}, {image.max()}]")

            # 2. Face detection
            landmarks = self.fa.get_landmarks(image)
            if not landmarks or len(landmarks) == 0:
                print("No faces detected in the image")
                return None, None

            # 3. Process detected face
            landmark = landmarks[0]
            margin = 40
            left = max(0, int(np.min(landmark[:, 0])) - margin)
            right = min(image.shape[1], int(np.max(landmark[:, 0])) + margin)
            top = max(0, int(np.min(landmark[:, 1])) - margin)
            bottom = min(image.shape[0], int(np.max(landmark[:, 1])) + margin)

            # 4. Extract and convert face region
            face = image[top:bottom, left:right]
            face = Image.fromarray(face)

            return face, (left, top, right, bottom)

        except Exception as e:
            print(f"Error in face alignment: {str(e)}")
            return None, None

    # def align_face(self, image):
    #     """Enhanced face detection with multi-face support"""
    #     try:
    #         # 1. Image format conversion
    #         if isinstance(image, str):
    #             image = Image.open(image)
    #         if isinstance(image, Image.Image):
    #             image = np.array(image)
    #
    #         # 2. Ensure RGB format
    #         if len(image.shape) == 3:
    #             if image.shape[2] == 4:  # RGBA image
    #                 # Convert RGBA to RGB
    #                 image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    #             elif image.shape[2] == 3:  # RGB/BGR image
    #                 image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if isinstance(image, np.ndarray) else image
    #         else:
    #             print(f"Unexpected image format: {image.shape}")
    #             return None, None
    #
    #         # 3. Enhance image preprocessing
    #         # Adjust contrast and brightness
    #         lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    #         l, a, b = cv2.split(lab)
    #         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    #         cl = clahe.apply(l)
    #         enhanced = cv2.merge((cl, a, b))
    #         enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    #
    #         # 4. Face detection with enhanced parameters
    #         try:
    #             landmarks = self.fa.get_landmarks(enhanced)
    #             print(f"Number of faces detected: {len(landmarks) if landmarks else 0}")
    #
    #             if not landmarks or len(landmarks) == 0:
    #                 # Try with original image if enhancement didn't help
    #                 landmarks = self.fa.get_landmarks(image_rgb)
    #                 if not landmarks or len(landmarks) == 0:
    #                     print("No faces detected in either enhanced or original image")
    #                     return None, None
    #
    #             # 5. Select the most prominent face
    #             # Calculate face sizes and centrality
    #             face_scores = []
    #             center_x = image_rgb.shape[1] / 2
    #             center_y = image_rgb.shape[0] / 2
    #
    #             for idx, lm in enumerate(landmarks):
    #                 # Calculate face size
    #                 width = np.max(lm[:, 0]) - np.min(lm[:, 0])
    #                 height = np.max(lm[:, 1]) - np.min(lm[:, 1])
    #                 size = width * height
    #
    #                 # Calculate distance from center
    #                 face_center_x = (np.max(lm[:, 0]) + np.min(lm[:, 0])) / 2
    #                 face_center_y = (np.max(lm[:, 1]) + np.min(lm[:, 1])) / 2
    #                 center_dist = np.sqrt((face_center_x - center_x) ** 2 + (face_center_y - center_y) ** 2)
    #
    #                 # Score based on size and centrality
    #                 score = size * (1 / (1 + center_dist / 100))
    #                 face_scores.append((idx, score))
    #
    #             # Select face with highest score
    #             best_face_idx = max(face_scores, key=lambda x: x[1])[0]
    #             landmark = landmarks[best_face_idx]
    #
    #             # 6. Extract face with larger margin
    #             margin = 40  # Increased margin
    #             left = max(0, int(np.min(landmark[:, 0])) - margin)
    #             right = min(image_rgb.shape[1], int(np.max(landmark[:, 0])) + margin)
    #             top = max(0, int(np.min(landmark[:, 1])) - margin)
    #             bottom = min(image_rgb.shape[0], int(np.max(landmark[:, 1])) + margin)
    #
    #             face = image_rgb[top:bottom, left:right]
    #             face_pil = Image.fromarray(face)
    #
    #             print(f"Successfully extracted face {best_face_idx + 1} of {len(landmarks)}")
    #             return face_pil, (left, top, right, bottom)
    #
    #         except Exception as e:
    #             print(f"Error during landmark detection: {str(e)}")
    #             return None, None
    #
    #     except Exception as e:
    #         print(f"Error in face alignment: {str(e)}")
    #         return None, None

    # def align_face(self, image):
    #     """Detect and align face in image"""
    #     try:
    #         if isinstance(image, str):
    #             image = Image.open(image)
    #         if isinstance(image, Image.Image):
    #             image = np.array(image)
    #
    #         if len(image.shape) == 3 and image.shape[2] == 3:
    #             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if isinstance(image, np.ndarray) else image
    #         else:
    #             image_rgb = image
    #
    #         landmarks = self.fa.get_landmarks(image_rgb)
    #         if not landmarks or len(landmarks) == 0:
    #             print("No face detected")
    #             return None, None
    #
    #         landmark = landmarks[0]
    #         margin = 30
    #         left = max(0, int(np.min(landmark[:, 0])) - margin)
    #         right = min(image.shape[1], int(np.max(landmark[:, 0])) + margin)
    #         top = max(0, int(np.min(landmark[:, 1])) - margin)
    #         bottom = min(image.shape[0], int(np.max(landmark[:, 1])) + margin)
    #
    #         face = image[top:bottom, left:right]
    #         face = Image.fromarray(face)
    #
    #         return face, (left, top, right, bottom)
    #
    #     except Exception as e:
    #         print(f"Error in face alignment: {str(e)}")
    #         return None, None

    def predict(self, image):
        """Predict if an image is fake or real"""
        try:
            quality_issues = self.assess_image_quality(image)
            if quality_issues:
                print(f"Quality issues detected: {quality_issues}")
                print("These issues can affect detection accuracy. Please use high-quality images.")
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
            <p>Deepfakes are synthetic media created using artificial intelligence.</p>
            <h3><span>🔍</span> How to Spot Deepfakes</h3>
            <ul>
                <li>Unnatural facial expressions and blinking</li>
                <li>Inconsistent lighting and shadows</li>
                <li>Blurry or misaligned facial features</li>
                <li>Artifacts around eyes, teeth, or jewelry</li>
            </ul>
            <h3><span>⚠️</span> Important Warning</h3>
            <ul>
                <li>Always verify content through multiple sources</li>
                <li>If in doubt, don't share the content</li>
                <li>Report suspicious content</li>
            </ul>
            <h3><span>🛡️</span> Best Practices</h3>
            <ul>
                <li>Use multiple verification tools</li>
                <li>Consider context and source credibility</li>
                <li>Excercise caution when uncertain</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# def display_educational_content():
#     st.markdown(
#         """
#         <div class="edu-content">
#             <h2>Understanding Deepfakes</h2>
#             <p>AI-generated synthetic media that manipulates faces, voices, and other elements in images/videos.</p>
#
#             <h3><span>🔍</span> Detection Tips</h3>
#             <ul>
#                 <li>Unnatural facial expressions and blinking</li>
#                 <li>Lighting/shadow inconsistencies</li>
#                 <li>Blurry features & unnatural textures</li>
#             </ul>
#
#             <h3><span>⚠️</span> Important Note</h3>
#             <ul>
#                 <li>Verify through multiple sources</li>
#                 <li>When in doubt, don't share</li>
#                 <li>Report suspicious content</li>
#             </ul>
#
#             <h3><span>🛡️</span> Best Practices</h3>
#             <ul>
#                 <li>Use multiple verification tools</li>
#                 <li>Check source credibility</li>
#                 <li>Exercise caution when uncertain</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)


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


# def image_analysis(detector):
#     """Handle image analysis mode"""
#     st.session_state.mode = 'image'
#     st.title("🔍 Demo: Deepfake Detection System")
#
#     # Sidebar controls
#     st.sidebar.subheader("Analysis Controls")
#     uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
#
#     if uploaded_file:
#         # Create main content columns
#         image = Image.open(uploaded_file)
#         image = ensure_rgb(image)
#         # st.sidebar.write("Image Information:")
#         # st.sidebar.write(f"Mode: {image.mode}")
#         # st.sidebar.write(f"Size: {image.size}")
#         quality_issues = detector.assess_image_quality(image)
#         if quality_issues:
#             st.sidebar.warning( f"Quality Issues Detected: {', '.join(quality_issues)}")
#         st.sidebar.info("""
#             📸 For best results:
#             - Use original images, not screenshots
#             - Avoid heavily compressed images
#             - Ensure good image quality
#             - Use images with clear, well-lit faces
#         """)
#         left_col, right_col = st.columns([0.7, 0.3])
#
#         # if quality_issues:
#         #     with st.expander("Quality Issues Detected", expanded=True):
#         #         st.warning(
#         #             f"""
#         #             ** Detected Issues:**
#         #             - {' '.join(quality_issues)}
#         #             These issues can affect detection accuracy.
#         #             \n- Avoid compressed images
#         #             \n- Use images with clear, well-lit faces
#         #             """)
#
#         with left_col:
#             # Create two sub-columns for original and analyzed images
#             img_col1, img_col2 = st.columns(2)
#
#             with img_col1:
#                 st.markdown("<h3 style='text-align: center;'>Original Image</h3>", unsafe_allow_html=True)
#                 image = Image.open(uploaded_file)
#                 # Resize image while maintaining aspect ratio
#                 basewidth = 300
#                 wpercent = (basewidth / float(image.size[0]))
#                 hsize = int((float(image.size[1]) * float(wpercent)))
#                 image_resized = image.resize((basewidth, hsize), Image.LANCZOS)
#                 st.image(image_resized, use_column_width=True)
#
#             # Analysis button in sidebar
#             if st.sidebar.button("🔎 Analyze Image", key="analyze_btn"):
#                 with st.spinner("Processing..."):
#                     pred, conf, bbox = detector.predict(image)
#
#                     if pred is None:
#                         st.error("⚠️ No face detected in the image!")
#                     else:
#                         with img_col2:
#                             st.markdown("<h3 style='text-align: center;'>Detected Face</h3>", unsafe_allow_html=True)
#                             face, _ = detector.align_face(image)
#                             if face:
#                                 # Resize face
#                                 face = ensure_rgb(face)
#                                 wpercent = (basewidth / float(face.size[0]))
#                                 hsize = int((float(face.size[1]) * float(wpercent)))
#                                 face_resized = face.resize((basewidth, hsize), Image.LANCZOS)
#
#                                 # Add overlay
#                                 is_fake = conf < 0.5
#                                 overlay = CROSS_IMG if is_fake else TICK_IMG
#                                 face_resized.paste(overlay, (face_resized.width - 50, 10), overlay)
#
#                                 st.image(face_resized, use_column_width=True)
#
#                         # Analysis results below images
#                         st.markdown("<div class='analysis-section'>", unsafe_allow_html=True)
#
#                         # Metrics in columns
#                         met_col1, met_col2, met_col3 = st.columns(3)
#                         with met_col1:
#                             st.metric("Confidence", f"{conf:.1%}")
#                         with met_col2:
#                             reliability, color, message, _ = calculate_reliability(conf)
#                             st.metric("Reliability", reliability)
#                         with met_col3:
#                             st.metric("Prediction", "FAKE" if conf < 0.5 else "REAL")
#
#                         # Detailed results
#                         st.subheader("Analysis Details")
#                         st.write(f"Interpretation: {message}")
#
#                         # Display confidence gauge
#                         _, _, _, fig = calculate_reliability(conf)
#                         st.pyplot(fig)
#
#                         st.markdown("</div>", unsafe_allow_html=True)
#
#                         # Display model metrics
#                         display_model_metrics()
#
#         # Educational content in right column
#         with right_col:
#             display_educational_content()


def image_analysis(detector):
    """Handle image analysis mode with unique button keys"""
    st.session_state.mode = 'image'
    st.title("🔍 Demo: Deepfake Detection System")

    # Sidebar controls
    st.sidebar.subheader("Analysis Controls")
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], key="image_uploader")

    if uploaded_file:
        try:
            # Load and convert image
            image = Image.open(uploaded_file)

            # Check image quality
            quality_issues = detector.assess_image_quality(image)
            if quality_issues:
                with st.expander("⚠️ Image Quality Issues Detected", expanded=True):
                    st.warning(
                        f"""
                        **Detected Issues:**
                        - {''.join(quality_issues)}

                        These issues may affect detection accuracy. For best results:
                        - Use original, unscreenshotted images
                        - Ensure good lighting and focus
                        - Avoid heavily compressed images
                        - Use clear, front-facing photos
                        """
                    )

            # Create main content columns
            left_col, right_col = st.columns([0.7, 0.3])

            with left_col:
                img_col1, img_col2 = st.columns(2)

                with img_col1:
                    st.markdown("<h3 style='text-align: center;'>Original Image</h3>", unsafe_allow_html=True)
                    # Resize image while maintaining aspect ratio
                    basewidth = 300
                    wpercent = (basewidth / float(image.size[0]))
                    hsize = int((float(image.size[1]) * float(wpercent)))
                    image_resized = image.resize((basewidth, hsize), Image.LANCZOS)
                    st.image(image_resized, use_column_width=True)

                # Analysis button with unique key
                if st.sidebar.button("🔎 Analyze Image", key="analyze_image_button"):
                    with st.spinner("Processing..."):
                        # Quality analysis section
                        st.markdown("""
                        <div class='analysis-section'>
                            <h3>Image Quality Analysis</h3>
                        """, unsafe_allow_html=True)

                        if quality_issues:
                            for issue in quality_issues:
                                st.markdown(f"• {issue}", unsafe_allow_html=True)

                            quality_score = max(0, 100 - (len(quality_issues) * 20))
                            st.metric(
                                "Image Quality Score",
                                f"{quality_score}%"
                            )
                        else:
                            st.success("✅ No image quality issues detected")

                        st.markdown("</div>", unsafe_allow_html=True)

                        # Deepfake analysis
                        pred, conf, bbox = detector.predict(image)

                        if pred is None:
                            st.error("⚠️ No face detected in the image!")
                        else:
                            with img_col2:
                                st.markdown("<h3 style='text-align: center;'>Detected Face</h3>",
                                            unsafe_allow_html=True)
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

                                    st.image(face_resized, use_column_width=True)

                            # Analysis results
                            st.markdown("<div class='analysis-section'>", unsafe_allow_html=True)

                            # Metrics with unique keys
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
        except Exception as e:

            st.error(f"Error processing image: {str(e)}")
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
