import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import matplotlib.pyplot as plt

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_mobilenetv2_corrosion.keras')

model = load_model()

# --- Config ---
IMG_SIZE = (224, 224)
MAX_UPLOAD_SIZE_MB = 200


st.title("🔍 Demo Klasifikasi Corrosion vs No Corrosion")
st.write("""
Aplikasi ini mendeteksi kondisi benda apakah mengalami **corrosion** atau **no corrosion**  
menggunakan model MobileNetV2 yang sudah dilatih.  
""")

st.sidebar.title("📁 Contoh Gambar")
example_imgs = {
    "Corrosion Example": "examples/corrosion1.jpg",
    "No Corrosion Example": "examples/nocorrosion1.jpg"
}

selected_example = st.sidebar.selectbox("Pilih contoh gambar", list(example_imgs.keys()))

if os.path.exists(example_imgs[selected_example]):
    example_image = Image.open(example_imgs[selected_example])
    st.sidebar.image(example_image, caption=selected_example, use_container_width=True)
else:
    st.sidebar.error("File contoh tidak ditemukan.")

# --- How to Use ---
st.subheader("📚 Cara Menggunakan")
st.markdown("""
1. **Lihat Contoh Gambar**  
   Di sidebar kiri, ada gambar **Corrosion** atau **No Corrosion** sebagai referensi.

2. **Upload Gambar Anda**  
   Klik tombol **Browse files** di bawah ini untuk upload gambar. Format yang diterima: **JPG, JPEG, PNG** (maks. 200MB).

3. **Melihat Hasil Prediksi**  
   Aplikasi akan menampilkan hasil prediksi apakah gambar termasuk **corrosion** atau **no corrosion** dengan tingkat keyakinan (**confidence**).

4. **Unduh Gambar Hasil Prediksi**  
   Setelah prediksi ditampilkan, gambar yang telah diberi overlay teks hasil prediksi bisa diunduh.
""")

# --- Upload file ---
uploaded_file = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"])

def validate_file(file) -> bool:
    size_mb = file.size / (1024*1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        st.error(f"File terlalu besar! Maksimal {MAX_UPLOAD_SIZE_MB} MB.")
        return False
    return True

if uploaded_file and validate_file(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert('RGB')
    except:
        st.error("File bukan gambar yang valid!")
        st.stop()

    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='...', use_container_width=True)

    with st.spinner('Memproses gambar...'):
        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred_prob = model.predict(img_array)[0][0]
        pred_label = 1 if pred_prob > 0.5 else 0

        class_indices = {'CORROSION': 0, 'NO CORROSION': 1}  # ganti sesuai generator aslinya
        inv_class_indices = {v: k for k, v in class_indices.items()}
        label = inv_class_indices[pred_label]
        confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

    with col2:
        if label == 'CORROSION':
            st.markdown(f"<h2 style='color:red;'>⚠️ Prediksi: {label}</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:green;'>✅ Prediksi: {label}</h2>", unsafe_allow_html=True)

        st.markdown(f"Confidence: **{confidence:.2%}**")

        # --- Pie Chart ---
        fig, ax = plt.subplots()
        ax.pie([1 - pred_prob, pred_prob], labels=['Corrosion', 'No Corrosion'], autopct='%1.1f%%', colors=['#FF6347', '#32CD32'])
        ax.axis('equal')  
        st.pyplot(fig)

        # --- Rating Text for Confidence ---
        if confidence > 0.9:
            rating = "Sangat Yakin"
        elif confidence > 0.75:
            rating = "Yakin"
        elif confidence > 0.5:
            rating = "Cukup Yakin"
        else:
            rating = "Ragu-ragu"
        
        st.markdown(f"Rating Confidence: **{rating}**")

    def overlay_text(image, text):
        import PIL.ImageDraw as ImageDraw
        import PIL.ImageFont as ImageFont

        image = image.copy()
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arialbd.ttf", 28)
        except:
            font = ImageFont.load_default()

        display_text = f"{text} ({confidence:.2%})"
        pos = (10, 10)
        text_bbox = draw.textbbox(pos, display_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = 8
        bg_rect = [pos[0] - padding, pos[1] - padding,
                   pos[0] + text_width + padding, pos[1] + text_height + padding]
        draw.rectangle(bg_rect, fill=(0, 0, 0, 150))
        draw.text(pos, display_text, fill=(255, 255, 255), font=font)

        return image

    overlayed_image = overlay_text(image, label)
    buf = io.BytesIO()
    overlayed_image.save(buf, format='PNG')
    byte_im = buf.getvalue()

    st.download_button(
        label="Download",
        data=byte_im,
        file_name="hasil_prediksi_korosi.png",
        mime="image/png"
    )
else:
    if uploaded_file:
        st.warning("Mohon upload file yang valid dan ukuran tidak melebihi 200MB.")
