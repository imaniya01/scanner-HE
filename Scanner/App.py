import streamlit as st
import os
import cv2
import numpy as np
from main import process_image, manual_crop
from PIL import Image
from io import BytesIO

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Document Scanner", layout="wide")

st.title("Document Scanner menggunakan Metode Histogram Equalization")
uploaded_files = st.file_uploader("Upload maks. 5 gambar", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

save_format = st.selectbox("Format", ["jpg", "png"])
manual_crop_enabled = st.checkbox("Crop Manual")

processed_images = []

if uploaded_files:
    for uploaded_file in uploaded_files[:5]:  # Limit to 5 images
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.subheader(f"Original - {uploaded_file.name}")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if manual_crop_enabled:
            cropped = manual_crop(image)
        else:
            cropped = process_image(image)

        processed_images.append((uploaded_file.name, cropped))

        st.subheader("Processed")
        st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Save Section
    st.subheader("Simpan Gambar Hasil")
    if st.button("Simpan"):
        os.makedirs("output", exist_ok=True)
        for filename, img in processed_images:
            out_name = os.path.splitext(filename)[0] + f"_scanned.{save_format}"
            out_path = os.path.join("output", out_name)
            cv2.imwrite(out_path, img)
        st.success(f"{len(processed_images)} Gambar Hasil Scan telah Tersimpan di Folder.")
