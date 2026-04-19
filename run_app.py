import os
import sys

print("🚀 Starting Deepfake Detection Web App...")

python_exe = sys.executable

# ✅ FIX: add quotes around path
os.system(f'"{python_exe}" -m streamlit run app.py')