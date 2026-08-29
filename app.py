import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & MEDICAL THEME SYSTEM
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroClarity | Dementia Stage Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Clinical Styling
st.markdown("""
    <style>
        .main { background-color: #0F172A; }
        h1, h2, h3 { color: #00B4D8 !important; font-family: 'Helvetica Neue', sans-serif; }
        .stButton>button {
            background-color: #007A87;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            border: none;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #00B4D8;
            border: none;
            transform: scale(1.02);
        }
        .reportview-container .main .block-container { padding-top: 2rem; }
        .metric-card {
            background-color: #1E293B;
            border-left: 5px solid #00B4D8;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True) 

# -----------------------------------------------------------------------------
# 2. MACHINE LEARNING MODEL ARCHITECTURE
# -----------------------------------------------------------------------------
class DementiaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Safe Runtime Execution Target (Forcing safe CPU processing map for Cloud instances)
device = torch.device("cpu")
classes = ['Mild Dementia', 'Moderate Dementia', 'Non-Demented', 'Very Mild Dementia']

@st.cache_resource
def load_weights(weights_path):
    if not os.path.exists(weights_path):
        return None
    model = DementiaCNN()
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        st.error(f"Error reading model structure: {e}")
        return None
    model.eval()
    return model

# Locate the metrics file relative to execution path
MODEL_PATH = "best_model.pth"
model = load_weights(MODEL_PATH)

# Image evaluation pipeline matching parameters derived during your training process
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# -----------------------------------------------------------------------------
# 3. STREAMLIT FRONTEND & MEDICAL INTERFACE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2868/2868244.png", width=80)
    st.title("NeuroClarity Portal")
    st.markdown("---")
    st.markdown("### Clinical Protocol Checklist:")
    st.info("""
    - Ensure Scan is T1 or T2 Weighted Axial MRI.
    - Artifact check completed.
    - Image contains zero patient identifiable metadata.
    """)
    st.markdown("---")
    st.caption("Developed using Convolutional Neural Network Architectures running on PyTorch Core Engines.")

# Main Screen Real Estate Split
st.title("🧠 Dementia Stage Detection from Volumetric Brain MRI")
st.markdown("##### Clinical Decision Support System — Powered by deep-layer Convolutional Neural Networks (CNN)")
st.markdown("---")

if model is None:
    st.error(f"⚠️ **Deployment Error:** File target `{MODEL_PATH}` not detected at project base directory root.")
    st.info("Ensure that `best_model.pth` generated during training is pushed alongside the execution scripts inside your GitHub repository folder.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Diagnostic Sample Input")
        uploaded_file = st.file_uploader(
            "Load Patient Neuroimaging Target Scan (.jpg, .jpeg, .png)", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Patient Structural MRI Matrix", use_container_width=True)
            
            with st.spinner("Executing Pixel Vector Extraction & Tensor Computations..."):
                tensor_img = transform_pipeline(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor_img)
                    probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
                    prediction_idx = np.argmax(probabilities)
                    confidence = probabilities[prediction_idx] * 100

    with col2:
        st.subheader("Automated Diagnostics Engine Output")
        if uploaded_file is not None:
            # Classification Visual Callout Cards
            predicted_class = classes[prediction_idx]
            
            # Select Alert Levels for UI metrics styling based on status outputs
            if "Non" in predicted_class:
                status_color = "green"
                text_msg = "Neurotypical structural density ranges detected."
            elif "Very Mild" in predicted_class or "Mild" in predicted_class:
                status_color = "orange"
                text_msg = "Atrophy markers noted. Recommend clinical cognitive follow-up assays."
            else:
                status_color = "red"
                text_msg = "Significant structural degradation detected in global gray-matter matrices."

            st.markdown(f"""
                <div class="metric-card">
                    <h3>Identified Stage: <span style='color:#00B4D8;'>{predicted_class}</span></h3>
                    <h4>Confidence Metric: <span style='color:#00B4D8;'>{confidence:.2f}%</span></h4>
                    <p style='margin-top:10px; font-style:italic;'>Status Note: {text_msg}</p>
                </div>
            """, unsafe_allow_html=True)

            # Matplotlib Probability Metrics Bar-Plot
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#1E293B')
            ax.set_facecolor('#1E293B')
            
            y_pos = np.arange(len(classes))
            bars = ax.barh(y_pos, probabilities, align='center', color='#007A87', edgecolor='#00B4D8')
            bars[prediction_idx].set_color('#00B4D8') # Highlight winning class
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(classes, color='#F8FAFC', fontsize=10)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Probability Vector Density', color='#F8FAFC', fontsize=10)
            ax.xaxis.label.set_color('#F8FAFC')
            ax.tick_params(axis='x', colors='#F8FAFC')
            ax.spines['bottom'].set_color('#334155')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            
            plt.tight_layout() 
            st.pyplot(fig)
            
        else:
            st.warning("Awaiting Structural Matrix input. Upload an axial brain slice MRI file inside the diagnostic section to proceed.")