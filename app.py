import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
import gdown

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Sistem Klasifikasi Citra Sayuran",
    page_icon="🥦",
    layout="wide"
)

# Custom Styling (Charcoal Slate, Biru Muda Cerah & Layout Ramping untuk Screenshot)
st.markdown("""
    <style>
    /* 1. Header Atas Streamlit */
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    /* 2. Latar Belakang Utama Aplikasi */
    .stApp {
        background-color: #ffffff !important;
        color: #334155 !important;
    }
    
    /* 3. Bar Sisi (Sidebar) */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
        color: #334155 !important;
        border-right: 1px solid #cbd5e1;
    }
    
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] p {
        color: #334155 !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #cbd5e1 !important;
    }

    /* 4. Judul Utama Ringkas */
    .clean-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b !important;
        margin-top: -20px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* 5. Semua Label Teks & Paragraf */
    h1, h2, h3, h4, p, label, .stMarkdown, .stRadio label {
        color: #334155 !important;
        font-weight: 600;
    }

    /* 6. Area Kotak Upload Ringkas */
    [data-testid="stFileUploader"] section {
        padding: 12px 16px;
        min-height: 90px;
        border: 2px dashed #7dd3fc !important;
        border-radius: 10px;
        background-color: #f8fafc !important;
    }
    
    /* Tombol Upload */
    [data-testid="stFileUploader"] section button {
        background-color: #38bdf8 !important;
        border-radius: 8px !important;
        border: 1px solid #0ea5e9 !important;
        box-shadow: 0 2px 6px rgba(56, 189, 248, 0.3) !important;
    }
    [data-testid="stFileUploader"] section button * {
        color: #1e293b !important;
        font-weight: 700 !important;
    }

    /* 7. Tombol Prediksi Utama */
    div.stButton > button:first-child {
        background-color: #38bdf8 !important;
        color: #1e293b !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        border-radius: 8px !important;
        border: 1px solid #0ea5e9 !important;
        padding: 8px 16px !important;
        box-shadow: 0 3px 8px rgba(56, 189, 248, 0.3) !important;
        transition: all 0.2s ease !important;
        margin-top: 8px;
    }
    div.stButton > button:first-child:hover {
        background-color: #0ea5e9 !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }
    div.stButton > button:first-child * {
        color: #1e293b !important;
        font-weight: 800 !important;
    }

    /* 8. Radio Buttons */
    div[data-testid="stRadio"] div[role="radiogroup"] label div:first-child {
        border-color: #475569 !important;
        border-width: 2px !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] div:first-child {
        border-color: #334155 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label div:first-child > div,
    div[data-testid="stRadio"] div[role="radiogroup"] label div:first-child::after {
        background-color: #334155 !important;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        accent-color: #334155 !important;
    }

    /* 9. Batasi Ukuran Gambar Pratinjau agar Ramping */
    .preview-box {
        max-width: 280px;
        margin: 0 auto;
        text-align: center;
    }
    .preview-box img {
        max-height: 240px !important;
        object-fit: contain;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    </style>
""", unsafe_allow_html=True)

# Pemuatan Model Otomatis dari Google Drive
@st.cache_resource
def load_classification_model():
    model_name = "model_sayuran_custom_cnn.keras"
    if not os.path.exists(model_name):
        file_id = "1sEWOkVbdtFxc6-0J7oU2ITBE83HtIVVQ"
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("Mengunduh model CNN dari Google Drive... Harap tunggu sebentar."):
            gdown.download(url, model_name, quiet=False)
            
    if not os.path.exists(model_name):
        return None
        
    return tf.keras.models.load_model(model_name)

model = load_classification_model()
class_names = ['Carrot', 'Cucumber', 'Tomato']

# Informasi Detail Sayuran
vegetable_details = {
    'Carrot': {
        'nama': 'Wortel',
        'nutrisi': 'Kaya Vitamin A (Beta-karoten), Vitamin K1, Kalium, dan Serat.',
        'manfaat': 'Menjaga kesehatan penglihatan mata, meningkatkan imunitas, dan menjaga metabolisme tubuh.',
        'penyimpanan': 'Potong daun hijau, simpan di laci kulkas dengan wadah tertutup agar tahan hingga 3-4 minggu.'
    },
    'Cucumber': {
        'nama': 'Mentimun',
        'nutrisi': 'Mengandung kadar air hingga 95%, Vitamin K, Vitamin C, dan Mineral Magnesium.',
        'manfaat': 'Mencegah dehidrasi, membantu melancarkan pencernaan, dan meredakan peradangan.',
        'penyimpanan': 'Bungkus rapat dengan tisu atau kertas kering, simpan pada bagian kulkas yang tidak terlalu dingin.'
    },
    'Tomato': {
        'nama': 'Tomat',
        'nutrisi': 'Kaya Antioksidan Likopen, Vitamin C, Kalium, dan Folat.',
        'manfaat': 'Membantu menjaga kesehatan jantung, menangkal radikal bebas, dan merawat kulit.',
        'penyimpanan': 'Simpan pada suhu ruangan dengan posisi tangkai menghadap ke bawah agar tetap segar dan tidak lembek.'
    }
}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🥦 **Klasifikasi Sayuran**")
    st.caption("Sistem Deep Learning Computer Vision")
    st.markdown("---")
    
    menu = st.radio(
        "Menu Navigasi",
        options=["📤 Unggah & Prediksi", "ℹ️ Tentang Sistem"],
        index=0
    )
    
    st.markdown(
        """
        <div style="min-height: 240px;"></div>
        <div style="text-align: center; border-top: 1px solid #cbd5e1; padding-top: 14px;">
            <span style="display: inline-block; background-color: #ffffff; color: #334155; padding: 5px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 700; border: 1px solid #cbd5e1;">Oleh : Kelompok 6</span>
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- HALAMAN 1: UNGGAH & PREDIKSI ---
if menu == "📤 Unggah & Prediksi":
    st.markdown('<div class="clean-title">🥦 Klasifikasi Citra Sayuran</div>', unsafe_allow_html=True)

    if model is None:
        st.error("Model tidak ditemukan atau gagal diunduh. Pastikan link Google Drive sudah diatur publik.")
    else:
        input_source = st.radio(
            "Pilih Metode Masukan Gambar:",
            options=["📁 Unggah Gambar dari Galeri / Berkas", "📷 Ambil Foto Langsung Lewat Kamera"],
            horizontal=True
        )

        image_data = None
        source_name = "Kamera"

        if input_source == "📁 Unggah Gambar dari Galeri / Berkas":
            uploaded_file = st.file_uploader(
                "Pilih citra sayuran (.jpg, .jpeg, .png)", 
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                image_data = uploaded_file
                source_name = uploaded_file.name
        else:
            camera_file = st.camera_input("Arahkan kamera ke sayuran lalu ambil gambar")
            if camera_file is not None:
                image_data = camera_file
                source_name = "Foto_Kamera_Langsung.png"

        if image_data is not None:
            # Menggunakan proporsi kolom yang proporsional (kolom kiri lebih ramping untuk foto)
            col1, col2 = st.columns([1, 1.8], gap="medium")
            
            # Kolom Kiri: Pratinjau Citra yang Dikecilkan
            with col1:
                img_display = Image.open(image_data).convert("RGB")
                st.image(img_display, caption="Pratinjau Citra", width=280)
                run_btn = st.button("🔍 JALANKAN PREDIKSI", use_container_width=True)

            # Preprocessing Citra Sesuai Training (150x150)
            img_resized = img_display.resize((150, 150), Image.Resampling.BILINEAR)
            img_array = np.array(img_resized, dtype=np.float32)

            has_rescaling = any("rescaling" in layer.name.lower() for layer in model.layers)
            if has_rescaling:
                input_tensor = np.expand_dims(img_array, axis=0)
            else:
                input_tensor = np.expand_dims(img_array / 255.0, axis=0)

            # Kolom Kanan: Hasil Analisis
            with col2:
                if run_btn:
                    with st.spinner("Menganalisis citra..."):
                        preds = model.predict(input_tensor, verbose=0)[0]

                        if not np.isclose(np.sum(preds), 1.0, atol=1e-2):
                            preds = tf.nn.softmax(preds).numpy()

                        predicted_idx = int(np.argmax(preds))
                        label = class_names[predicted_idx]
                        confidence = float(preds[predicted_idx] * 100)
                        detail = vegetable_details[label]

                        st.markdown("#### 📊 Hasil Klasifikasi")

                        # Logika Batas Ambang 60%
                        if confidence < 60.0:
                            st.warning(f"⚠️ **Tingkat Keyakinan Rendah ({confidence:.2f}%)**\n\nCitra tidak dapat dikenali secara pasti. Pastikan sayuran berada di tengah dan pencahayaan jelas.")
                        else:
                            # Kartu Hasil Ramping
                            st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
                                    border: 1px solid #7dd3fc;
                                    border-radius: 10px;
                                    padding: 10px 16px;
                                    margin-bottom: 8px;
                                    display: flex;
                                    justify-content: space-between;
                                    align-items: center;
                                ">
                                    <div>
                                        <span style="font-size: 0.75rem; font-weight: 700; color: #0369a1; text-transform: uppercase;">Jenis Sayuran Terdeteksi:</span>
                                        <div style="font-size: 1.4rem; font-weight: 800; color: #1e293b;">{detail['nama']}</div>
                                    </div>
                                    <span style="
                                        background-color: #38bdf8;
                                        color: #1e293b !important;
                                        font-weight: 800;
                                        font-size: 1rem;
                                        padding: 4px 14px;
                                        border-radius: 16px;
                                        border: 1px solid #0ea5e9;
                                    ">{confidence:.2f}%</span>
                                </div>
                            """, unsafe_allow_html=True)

                            # Kartu Nutrisi & Manfaat
                            detail_col1, detail_col2 = st.columns(2)
                            with detail_col1:
                                st.markdown(f"""
                                    <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; min-height: 95px; font-size: 0.85rem;">
                                        <strong style="color: #1e293b;">🥗 Nutrisi:</strong><br>
                                        <span style="color: #475569;">{detail['nutrisi']}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                            with detail_col2:
                                st.markdown(f"""
                                    <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; min-height: 95px; font-size: 0.85rem;">
                                        <strong style="color: #1e293b;">💡 Manfaat:</strong><br>
                                        <span style="color: #475569;">{detail['manfaat']}</span>
                                    </div>
                                """, unsafe_allow_html=True)

                            st.markdown(f"""
                                <div style="background: #f1f5f9; border-left: 3px solid #38bdf8; padding: 8px 12px; border-radius: 4px; margin-top: 8px; font-size: 0.85rem;">
                                    <strong style="color: #1e293b;">❄️ Tips Penyimpanan:</strong>
                                    <span style="color: #475569;">{detail['penyimpanan']}</span>
                                </div>
                            """, unsafe_allow_html=True)

                        # Grafik Probabilitas yang Lebih Pendek (Tinggi 180px)
                        st.markdown("<p style='margin-bottom: 2px; margin-top: 10px; font-size: 0.9rem;'><b>Distribusi Skor Keyakinan</b></p>", unsafe_allow_html=True)
                        df_chart = pd.DataFrame({
                            'Sayuran': [vegetable_details[c]['nama'] for c in class_names],
                            'Probabilitas (%)': [round(p * 100, 2) for p in preds]
                        }).set_index('Sayuran')
                        st.bar_chart(df_chart, height=180)

# --- HALAMAN 2: TENTANG SISTEM ---
elif menu == "ℹ️ Tentang Sistem":
    st.markdown('<div class="clean-title">ℹ️ Tentang Sistem</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    ### Gambaran Proyek
    Aplikasi ini dikembangkan untuk mengidentifikasi dan mengklasifikasikan tiga jenis sayuran (**Wortel**, **Mentimun**, dan **Tomat**) secara otomatis dari citra digital.
    
    * **Arsitektur Model:** Custom Deep Convolutional Neural Network (CNN)
    * **Metodologi:** CRISP-DM
    * **Resolusi Masukan:** 150 x 150 piksel (RGB)
    * **Framework Antarmuka:** Streamlit
    """)
    st.info("Dikembangkan oleh **Kelompok 6** — Pengolahan Citra Digital")
