"""Streamlit web app: upload a document photo, scan it, and classify its type.

Run with:
    streamlit run src/document_classification/app.py
or:
    document-classification-web   (see README)
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from document_classification.pipeline import process_image

st.set_page_config(page_title="Document Scanner + Classifier", page_icon="📄", layout="wide")


@st.cache_data(show_spinner=False)
def _run(image_bytes: bytes, binarize: bool = False) -> dict:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    result = process_image(bgr, binarize=binarize)
    return {
        "flattened": cv2.cvtColor(result.flattened, cv2.COLOR_BGR2RGB),
        "cleaned": result.cleaned,
        "corners": result.corners.tolist(),
        "dewarp_method": result.dewarp_method,
        "denoise_method": result.denoise_method,
        "classification": result.classification,
        "original": np.array(pil),
    }


def _draw_corners(img_rgb: np.ndarray, corners) -> np.ndarray:
    out = img_rgb.copy()
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True, color=(255, 0, 0), thickness=max(2, img_rgb.shape[1] // 300))
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), max(4, img_rgb.shape[1] // 150), (0, 200, 0), -1)
    return out


st.title("📄 Document Scanner + Classifier")
st.caption(
    "Upload a photo of a document. The app detects the page, flattens perspective, "
    "cleans it into a scan, and predicts what kind of document it is."
)

uploaded = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png", "bmp", "webp"])
binarize = st.toggle(
    "Pure black & white",
    value=False,
    help="Off: clean grayscale scan (recommended). On: hard black-and-white — smaller files, but can break logos/photos.",
)

if uploaded is None:
    st.info("Upload a JPG/PNG photo of a receipt, note, invoice, form, ID, etc. to begin.")
    st.stop()

with st.spinner("Processing… (first run downloads the CLIP model, this can take a minute)"):
    res = _run(uploaded.getvalue(), binarize)

cls = res["classification"]
label = cls.get("label", "unknown")
category = cls.get("category", "")
conf = cls.get("confidence", 0.0)
cat_conf = cls.get("category_confidence", 0.0)
uncertain = cls.get("uncertain", False)
method = cls.get("method", "n/a")

st.subheader("Result")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Category", category or "—", help=f"{cat_conf*100:.1f}% confident")
c2.metric("Document type", label if not uncertain else "—",
          help="Specific type is uncertain — trust the category." if uncertain else None)
c3.metric("Confidence", f"{conf*100:.1f}%")
c4.metric("Classifier", method)

if uncertain:
    st.info(f"The category **{category}** is a confident call, but the specific "
            f"type is ambiguous here — treat the top guess below as a hint.")

if cls.get("top_k"):
    st.write("**Top guesses**")
    for item in cls["top_k"]:
        cat = item.get("category", "")
        prefix = f"{cat} › " if cat else ""
        st.progress(min(1.0, float(item["score"])), text=f"{prefix}{item['label']} — {item['score']*100:.1f}%")

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**1. Detected page**")
    st.image(_draw_corners(res["original"], res["corners"]), use_container_width=True)
    st.caption(f"Corner detection: {res['dewarp_method']}")
with col2:
    st.markdown("**2. Flattened**")
    st.image(res["flattened"], use_container_width=True)
with col3:
    st.markdown("**3. Cleaned scan**")
    st.image(res["cleaned"], use_container_width=True, clamp=True)
    st.caption(f"Enhancement: {res['denoise_method']}")

# Download button for the cleaned scan.
scan_pil = Image.fromarray(res["cleaned"])
buf = io.BytesIO()
scan_pil.save(buf, format="PNG")
st.download_button("⬇️ Download cleaned scan (PNG)", buf.getvalue(), file_name="scan.png", mime="image/png")
