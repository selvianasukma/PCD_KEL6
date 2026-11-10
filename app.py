import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
import gdown

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Vegetable Image Classification System",
    page_icon="🥦",
    layout="wide"
)

# Custom Styling (Paksa Semua Bagian Atas & Tombol Menjadi Putih Bersih)
st.markdown("""
    <style>
    /* 1. Header Atas Streamlit Dipaksa Putih Bersih */
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    /* 2. Latar Belakang Utama Aplikasi Selalu Putih */
    .stApp {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* 3. Bar Sisi (Sidebar) Soft Ice Blue */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f4f8 0%, #e2e8f0 100%) !important;
        color: #334155 !important;
        border-right: 1px solid #cbd5e1;
    }
    
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] p {
        color: #334155 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #cbd5e1 !important;
    }

    /* 4. Judul Utama */
    .clean-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b !important;
        margin-top: -15px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* 5. Area Upload Citra */
    [data-testid="stFileUploader"] section {
        padding: 30px 20px;
        min-height: 180px;
        border: 2px dashed #94a3b8 !important;
        border-radius: 12px;
        background-color: #f8fafc !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #64748b !important;
        background-color: #f1f5f9 !important;
    }
    
    /* Tombol 'Browse files' diubah dari hitam menjadi abu-abu kalem */
    [data-testid="stFileUploader"] section button {
        background-color: #475569 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] section button:hover {
        background-color: #334155 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stFileUploader"] span, 
    [data-testid="stFileUploader"] small {
        color: #475569 !important;
    }

    /* 6. Tombol Prediksi */
    div.stButton > button:first-child {
        background-color: #475569 !important;
        color: #ffffff !important;
        font-size: 0.95rem;
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #334155;
        padding: 10px 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    div.stButton > button:first-child:hover {
        background-color: #334155 !important;
        color: #ffffff !important;
    }

    /* 7. Lencana Footer Sidebar */
    .sidebar-footer-badge {
        display: inline-block;
        background-color: #ffffff;
        color: #475569 !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        border: 1px solid #cbd5e1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Memastikan Seluruh Teks Terbaca Jelas */
    h1, h2, h3, h4, p, label {
        color: #1e293b !important;
    }
    </style>
""", unsafe_allow_html=True)

# Pemuatan & Pengunduhan Model Otomatis
@st.cache_resource
def load_classification_model():
    model_name = "model_sayuran_custom_cnn.keras"
    
    # Unduh otomatis dari Google Drive jika model belum ada di cloud
    if not os.path.exists(model_name):
        # Masukkan File ID Google Drive Anda di bawah ini
        file_id = "MASUKKAN_FILE_ID_GOOGLE_DRIVE_DISINI"
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("Mengunduh model bobot CNN dari Google Drive... Harap tunggu sebentar."):
            gdown.download(url, model_name, quiet=False)
            
    if not os.path.exists(model_name):
        return None
        
    return tf.keras.models.load_model(model_name)

model = load_classification_model()
class_names = ['Carrot', 'Cucumber', 'Tomato']

# Database Informasi Sayuran
vegetable_details = {
    'Carrot': {
        'nama': 'Wortel (Daucus carota)',
        'nutrisi': 'Vitamin A (Beta-karoten), Vitamin K1, Kalium, Serat.',
        'manfaat': 'Mendukung kesehatan retina mata, mengoptimalkan daya tahan tubuh, dan menjaga metabolisme.'
    },
    'Cucumber': {
        'nama': 'Mentimun (Cucumis sativus)',
        'nutrisi': 'Kadar Air Tinggi (95%), Vitamin K, Vitamin C, Magnesium.',
        'manfaat': 'Menjaga keseimbangan hidrasi tubuh, membantu sistem pencernaan, dan meredakan peradangan kulit.'
    },
    'Tomato': {
        'nama': 'Tomat (Solanum lycopersicum)',
        'nutrisi': 'Likopen (Antioksidan tinggi), Vitamin C, Kalium, Folat.',
        'manfaat': 'Menurunkan risiko penyakit kardiovaskular, menangkal radikal bebas, dan menjaga vitalitas kulit.'
    }
}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🥦 **Vegetable Classifier**")
    st.caption("Deep Learning Computer Vision System")
    st.markdown("---")
    
    menu = st.radio(
        "Navigation Menu",
        options=["📤 Upload & Predict", "ℹ️ About System"],
        index=0
    )
    
    st.markdown(
        """
        <div style="min-height: 280px;"></div>
        <div style="text-align: center; border-top: 1px solid #cbd5e1; padding-top: 16px;">
            <span class="sidebar-footer-badge">By : Kelompok 6</span>
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- HALAMAN 1: UPLOAD & PREDICT ---
if menu == "📤 Upload & Predict":
    st.markdown('<div class="clean-title">🥦 Vegetable Image Classification</div>', unsafe_allow_html=True)
    st.markdown("---")

    if model is None:
        st.error("Model 'model_sayuran_custom_cnn.keras' belum berhasil dimuat. Pastikan File ID Google Drive valid dan tautan disetel publik (Anyone with the link).")
    else:
        st.subheader("Unggah Citra Sayuran")
        uploaded_file = st.file_uploader(
            "Pilih atau seret file gambar sayuran (.jpg, .jpeg, .png)", 
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            st.write("")
            col1, col2 = st.columns([1, 1], gap="large")
            
            # Kolom Kiri: Tampilan Citra Input
            with col1:
                img_display = Image.open(uploaded_file).convert("RGB")
                st.image(img_display, caption="Pratinjau Citra Input", use_container_width=True)
                run_btn = st.button("🔍 Jalankan Prediksi", use_container_width=True)

            # Preprocessing Citra (150x150, normalisasi [0, 1])
            img_resized = img_display.resize((150, 150))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Kolom Kanan: Hasil Analisis
            with col2:
                if run_btn:
                    with st.spinner("Menganalisis karakteristik piksel citra..."):
                        preds = model.predict(img_array, verbose=0)[0]
                        predicted_idx = np.argmax(preds)
                        label = class_names[predicted_idx]
                        confidence = preds[predicted_idx] * 100
                        detail = vegetable_details[label]

                        st.subheader("📊 Hasil Klasifikasi")
                        st.success(f"**Komoditas Terdeteksi:** {detail['nama']}")
                        st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")
                        
                        with st.expander("Informasi Nutrisi & Manfaat Kesehatan", expanded=True):
                            st.markdown(f"**Kandungan Nutrisi:**\n{detail['nutrisi']}")
                            st.markdown(f"**Manfaat Kesehatan:**\n{detail['manfaat']}")

                        # Grafik Probabilitas
                        st.subheader("Distribusi Skor Keyakinan")
                        df_chart = pd.DataFrame({
                            'Kelas': ['Wortel (Carrot)', 'Mentimun (Cucumber)', 'Tomat (Tomato)'],
                            'Probabilitas (%)': [round(p * 100, 2) for p in preds]
                        }).set_index('Kelas')
                        st.bar_chart(df_chart)

# --- HALAMAN 2: ABOUT SYSTEM ---
elif menu == "ℹ️ About System":
    st.markdown('<div class="clean-title">ℹ️ Tentang Sistem</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ### Gambaran Proyek
    Aplikasi ini dikembangkan untuk mengidentifikasi dan mengklasifikasikan tiga jenis komoditas sayuran (**Wortel**, **Mentimun**, dan **Tomat**) secara otomatis dari citra digital.
    
    * **Arsitektur Model:** Custom Deep Convolutional Neural Network (CNN)
    * **Metodologi:** Cross-Industry Standard Process for Data Mining (CRISP-DM)
    * **Resolusi Input:** 150 x 150 piksel (RGB)
    * **Framework Antarmuka:** Streamlit
    """)
    st.info("Dikembangkan oleh **Kelompok 6** — Pengolahan Citra Digital")
