import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import streamlit as st

def histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def auto_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break
    else:
        return image  # fallback jika tidak ditemukan dokumen

    pts = doc_cnt.reshape(4, 2)
    rect = order_points(pts)
    return four_point_transform(image, rect)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def process_image(image):
    cropped = auto_crop(image)
    equalized = histogram_equalization(cropped)
    return equalized

def manual_crop(image):
    st.subheader("🖼️ Draw rectangle to crop manually")

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        background_image=img_pil,
        update_streamlit=True,
        height=image.shape[0],
        width=image.shape[1],
        drawing_mode="rect",
        key="canvas",
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        obj = canvas_result.json_data["objects"][-1]  # kotak terakhir
        left = int(obj["left"])
        top = int(obj["top"])
        width = int(obj["width"])
        height = int(obj["height"])
        cropped = image[top:top+height, left:left+width]
        equalized = histogram_equalization(cropped)
        return equalized
    else:
        st.info("Silakan gambar kotak terlebih dahulu.")
        return image  # fallback (tidak dicrop)
