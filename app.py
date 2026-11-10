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

# Custom Styling (Tema Bersih & Elemen Biru Cerah)
st.markdown("""
    <style>
    /* 1. Header Atas Streamlit Putih Bersih */
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    /* 2. Latar Belakang Utama Aplikasi */
    .stApp {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* 3. Bar Sisi (Sidebar) */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f7ff 0%, #e0f2fe 100%) !important;
        color: #1e3a8a !important;
        border-right: 1px solid #bae6fd;
    }
    
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] p {
        color: #1e3a8a !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #bae6fd !important;
    }

    /* 4. Judul Utama */
    .clean-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a !important;
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
        border: 2px dashed #38bdf8 !important;
        border-radius: 12px;
        background-color: #f0f9ff !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #0284c7 !important;
        background-color: #e0f2fe !important;
    }
    
    /* Tombol 'Browse files' */
    [data-testid="stFileUploader"] section button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] section button:hover {
        background-color: #0369a1 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stFileUploader"] span, 
    [data-testid="stFileUploader"] small {
        color: #0369a1 !important;
    }

    /* 6. Tombol Prediksi Biru Cerah */
    div.stButton > button:first-child {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 8px;
        border: none !important;
        padding: 10px 20px;
        box-shadow: 0 4px 10px rgba(2, 132, 199, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #0369a1 !important;
        color: #ffffff !important;
        box-shadow: 0 6px 14px rgba(2, 132, 199, 0.35);
    }

    /* 7. Lencana Footer Sidebar */
    .sidebar-footer-badge {
        display: inline-block;
        background-color: #ffffff;
        color: #0284c7 !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        border: 1px solid #7dd3fc;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Warna Dasar Teks */
    h1, h2, h3, h4, p, label {
        color: #1e293b !important;
    }
    </style>
""", unsafe_allow_html=True)

# Pemuatan & Pengunduhan Model Otomatis dari Google Drive
@st.cache_resource
def load_classification_model():
    model_name = "model_sayuran_custom_cnn.keras"
    if not os.path.exists(model_name):
        file_id = "1sEWOkVbdtFxc6-0J7oU2ITBE83HtIVVQ"
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("Mengunduh model bobot CNN dari Google Drive... Harap tunggu sebentar."):
            gdown.download(url, model_name, quiet=False)
            
    if not os.path.exists(model_name):
        return None
        
    return tf.keras.models.load_model(model_name)

model = load_classification_model()
class_names = ['Carrot', 'Cucumber', 'Tomato']

# Database Informasi Lengkap Sayuran
vegetable_details = {
    'Carrot': {
        'nama_ilmiah': 'Daucus carota subsp. sativus',
        'nama_umum': 'Wortel (Carrot)',
        'keluarga': 'Apiaceae (Umbel-umbelan)',
        'ciri_fisik': 'Bentuk silindris memanjang dengan warna oranye cerah akibat pigmen karotenoid, tekstur renyah, dan berakar tunggang berdaging.',
        'nutrisi': 'Kaya Beta-karoten (Pro-Vitamin A), Vitamin K1, Kalium, Serat Pektin, serta Antioksidan Lutein.',
        'manfaat': 'Mendukung ketajaman penglihatan, meningkatkan sistem imunitas tubuh, dan memperlambat proses penuaan sel.',
        'penyimpanan': 'Potong tangkai daun hijau, simpan di laci bawah kulkas dengan wadah kedap air agar tahan 3-4 minggu.'
    },
    'Cucumber': {
        'nama_ilmiah': 'Cucumis sativus',
        'nama_umum': 'Mentimun (Cucumber)',
        'keluarga': 'Cucurbitaceae (Labu-labuan)',
        'ciri_fisik': 'Silindris memanjang dengan kulit hijau berbintik halus atau bergaris putih, berdaging renyah berair transparan.',
        'nutrisi': 'Kadar air mencapai 95-96%, Vitamin K, Vitamin C, Magnesium, serta Asam Kafeat dan Silika.',
        'manfaat': 'Membantu hidrasi tubuh secara optimal, menjaga keseimbangan tekanan darah, dan detoksifikasi.',
        'penyimpanan': 'Bungkus rapat per buah dengan tisu atau kertas kering, simpan di rak kulkas bagian tengah yang sejuk.'
    },
    'Tomato': {
        'nama_ilmiah': 'Solanum lycopersicum',
        'nama_umum': 'Tomat (Tomato)',
        'keluarga': 'Solanaceae (Terung-terungan)',
        'ciri_fisik': 'Bentuk bulat hingga oval berdinding buah halus, berwarna merah menyala saat matang, berair dengan biji berselaput lendir.',
        'nutrisi': 'Tinggi Likopen (Antioksidan kuat), Vitamin C, Kalium, Folat (Vitamin B9), dan Asam Klorogenat.',
        'manfaat': 'Menurunkan risiko penyakit kardiovaskular, menangkal radikal bebas, serta menjaga kesehatan sel kulit.',
        'penyimpanan': 'Simpan pada suhu ruang dengan pangkal tangkai menghadap ke bawah; hindari penyimpanan kulkas berlebih agar aroma dan tekstur tidak rusak.'
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
        <div style="text-align: center; border-top: 1px solid #bae6fd; padding-top: 16px;">
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
        st.error("Model gagal diunduh atau tidak ditemukan. Pastikan tautan Google Drive publik.")
    else:
        st.subheader("Unggah Citra Sayuran")
        uploaded_file = st.file_uploader(
            "Pilih atau seret file gambar sayuran (.jpg, .jpeg, .png)", 
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            st.write("")
            col1, col2 = st.columns([1, 1], gap="large")
            
            # Kolom Kiri: Pratinjau Citra Input
            with col1:
                img_display = Image.open(uploaded_file).convert("RGB")
                st.image(img_display, caption="Pratinjau Citra Input", use_container_width=True)
                run_btn = st.button("🔍 Jalankan Prediksi", use_container_width=True)

            # Preprocessing Citra (150x150, normalisasi [0, 1])
            img_resized = img_display.resize((150, 150))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Kolom Kanan: Hasil Analisis Lengkap
            with col2:
                if run_btn:
                    with st.spinner("Menganalisis karakteristik visual dan piksel citra..."):
                        preds = model.predict(img_array, verbose=0)[0]
                        predicted_idx = np.argmax(preds)
                        label = class_names[predicted_idx]
                        confidence = preds[predicted_idx] * 100
                        detail = vegetable_details[label]

                        st.subheader("📊 Hasil Klasifikasi")
                        
                        # Kartu Biru Cerah Komoditas Terdeteksi
                        st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
                                border: 1px solid #7dd3fc;
                                border-radius: 12px;
                                padding: 18px 22px;
                                margin-bottom: 12px;
                                box-shadow: 0 2px 6px rgba(14, 165, 233, 0.1);
                            ">
                                <span style="font-size: 0.82rem; font-weight: 700; color: #0284c7; text-transform: uppercase; letter-spacing: 0.8px;">Komoditas Terdeteksi</span>
                                <div style="font-size: 1.4rem; font-weight: 800; color: #0369a1; margin-top: 4px;">
                                    {detail['nama_umum']}
                                </div>
                                <div style="font-size: 0.95rem; font-style: italic; color: #075985;">
                                    {detail['nama_ilmiah']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        # Kartu Biru Cerah Tingkat Keyakinan
                        st.markdown(f"""
                            <div style="
                                background: #f0f9ff;
                                border: 1px solid #bae6fd;
                                border-radius: 10px;
                                padding: 12px 18px;
                                margin-bottom: 16px;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            ">
                                <span style="font-weight: 600; color: #0369a1; font-size: 0.95rem;">Tingkat Keyakinan (Confidence):</span>
                                <span style="
                                    background-color: #0284c7;
                                    color: #ffffff !important;
                                    font-weight: 800;
                                    font-size: 1.05rem;
                                    padding: 4px 14px;
                                    border-radius: 20px;
                                ">{confidence:.2f}%</span>
                            </div>
                        """, unsafe_allow_html=True)

                        # Kartu Informasi Detail (2 Kolom)
                        detail_col1, detail_col2 = st.columns(2)
                        with detail_col1:
                            st.markdown(f"""
                                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; min-height: 140px;">
                                    <strong style="color: #0284c7;">🥗 Kandungan Nutrisi:</strong><br>
                                    <span style="font-size: 0.9rem; color: #334155;">{detail['nutrisi']}</span>
                                </div>
                            """, unsafe_allow_html=True)
                        with detail_col2:
                            st.markdown(f"""
                                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; min-height: 140px;">
                                    <strong style="color: #0284c7;">💡 Manfaat Kesehatan:</strong><br>
                                    <span style="font-size: 0.9rem; color: #334155;">{detail['manfaat']}</span>
                                </div>
                            """, unsafe_allow_html=True)

                        st.write("")
                        st.markdown(f"""
                            <div style="background: #f8fafc; border-left: 4px solid #0284c7; padding: 12px 16px; border-radius: 6px;">
                                <strong style="color: #0f172a;">❄️ Rekomendasi Penyimpanan:</strong><br>
                                <span style="font-size: 0.9rem; color: #475569;">{detail['penyimpanan']}</span>
                            </div>
                        """, unsafe_allow_html=True)

                        # Grafik Probabilitas
                        st.write("")
                        st.subheader("Distribusi Skor Keyakinan")
                        df_chart = pd.DataFrame({
                            'Kelas': ['Wortel (Carrot)', 'Mentimun (Cucumber)', 'Tomat (Tomato)'],
                            'Probabilitas (%)': [round(p * 100, 2) for p in preds]
                        }).set_index('Kelas')
                        st.bar_chart(df_chart)

                        # Metadata Teknis Gambar
                        with st.expander(f"🔍 Detail Gambar Input ({uploaded_file.name})"):
                            st.write(f"- **Nama File:** `{uploaded_file.name}`")
                            st.write(f"- **Dimensi Citra Asli:** `{img_display.size[0]} × {img_display.size[1]} piksel`")
                            st.write(f"- **Resolusi Masukan Model:** `150 × 150 piksel (RGB)`")
                            st.write(f"- **Format Data:** `{uploaded_file.type}`")

# --- HALAMAN 2: ABOUT SYSTEM ---
elif menu == "ℹ️ About System":
    st.markdown('<div class="clean-title">ℹ️ Tentang Sistem</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ### Gambaran Proyek
    Aplikasi ini dikembangkan untuk mengidentifikasi dan mengklasifikasikan tiga jenis komoditas sayuran (**Wortel**, **Mentimun**, dan **Tomat**) secara otomatis dari citra digital.
    
    * **Arsitektur Model:** Custom Deep Convolutional Neural Network (CNN)
    * **Metodologi:** Cross-Industry Standard Process for Data Mining (CRISP-DM)
    * **Resolusi Masukan:** 150 x 150 piksel (RGB)
    * **Framework Antarmuka:** Streamlit
    """)
    st.info("Dikembangkan oleh **Kelompok 6** — Pengolahan Citra Digital")
