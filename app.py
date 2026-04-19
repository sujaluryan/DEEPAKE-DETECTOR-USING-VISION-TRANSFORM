import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from insightface.app import FaceAnalysis
import tempfile
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deepfake Detector", layout="wide")

st.title("🧠 Deepfake Video Detection System")

st.write("Upload a video and the AI model will determine whether it is **REAL or FAKE**.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():

    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)

    model.load_state_dict(torch.load("checkpoints/vit_deepfake_model.pth", map_location=device))
    model.to(device)
    model.eval()

    return model

model = load_model()

# ---------------------------
# Face detector
# ---------------------------
face_detector = FaceAnalysis(name="buffalo_l")
face_detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------------------
# Upload video
# ---------------------------
uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

if uploaded_video:

    st.video(uploaded_video)

    st.write("🔄 Processing video...")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress = st.progress(0)

    frame_probs = []
    frame_count = 0

    preview_frame = st.empty()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # update progress bar
        progress.progress(frame_count / total_frames)

        # process every 3th frame
        if frame_count % 3 != 0:
            continue

        faces = face_detector.get(frame)

        if len(faces) == 0:
            continue

        bbox = faces[0].bbox.astype(int)

        x1, y1, x2, y2 = bbox

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        preview_frame.image(face_rgb, caption="Detected Face", width=200)

        img = Image.fromarray(face_rgb)

        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(img)

            prob = torch.softmax(output, dim=1)[0][1].item()

        frame_probs.append(prob)

    cap.release()

    # ---------------------------
    # Final prediction
    # ---------------------------
    if len(frame_probs) == 0:

        st.error("❌ No face detected in video")

    else:

        video_score = np.mean(frame_probs)

        st.subheader("📊 Deepfake Probability")

        st.write(video_score)

        if video_score > 0.5:
            st.error("🚨 Prediction: FAKE")
        else:
            st.success("✅ Prediction: REAL")

        # ---------------------------
        # Probability graph
        # ---------------------------
        st.subheader("Frame Prediction Graph")

        fig, ax = plt.subplots()

        ax.plot(frame_probs)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Fake Probability")
        ax.set_title("Frame-wise Deepfake Probability")

        st.pyplot(fig)