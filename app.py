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

# Custom Styling (Semua Teks Hitam Pekat & Elemen Kontras)
st.markdown("""
    <style>
    /* 1. Header Atas Streamlit */
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    /* 2. Latar Belakang Utama Aplikasi */
    .stApp {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    /* 3. Bar Sisi (Sidebar) */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
        color: #0f172a !important;
        border-right: 1px solid #cbd5e1;
    }
    
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] p {
        color: #0f172a !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #cbd5e1 !important;
    }

    /* 4. Judul Utama (Hitam Tebal) */
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

    /* 5. Semua Label Teks & Paragraf Menjadi Hitam */
    h1, h2, h3, h4, p, label, .stMarkdown, .stRadio label {
        color: #0f172a !important;
        font-weight: 600;
    }

    /* 6. Area Kotak Upload */
    [data-testid="stFileUploader"] section {
        padding: 24px 20px;
        min-height: 150px;
        border: 2px dashed #94a3b8 !important;
        border-radius: 12px;
        background-color: #f8fafc !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #475569 !important;
        background-color: #f1f5f9 !important;
    }
    
    /* Tombol Upload (Biru Cerah dengan Teks Hitam) */
    [data-testid="stFileUploader"] section button {
        background-color: #38bdf8 !important;
        border-radius: 8px !important;
        border: 1px solid #0284c7 !important;
        box-shadow: 0 2px 8px rgba(56, 189, 248, 0.35) !important;
    }
    [data-testid="stFileUploader"] section button:hover {
        background-color: #0ea5e9 !important;
    }
    
    /* Teks & Ikon Tombol Upload Menjadi Hitam */
    [data-testid="stFileUploader"] section button * {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    
    /* Keterangan Ukuran File Menjadi Hitam */
    [data-testid="stFileUploader"] span, 
    [data-testid="stFileUploader"] small {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* 7. Tombol Prediksi Utama */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
        border-radius: 10px !important;
        border: 1px solid #1e40af !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px rgba(30, 64, 175, 0.5) !important;
        transform: translateY(-2px);
    }

    /* 8. Lencana Footer Sidebar */
    .sidebar-footer-badge {
        display: inline-block;
        background-color: #ffffff;
        color: #0f172a !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 800;
        border: 1px solid #cbd5e1;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
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

# Informasi Detail Sayuran (Bahasa Indonesia)
vegetable_details = {
    'Carrot': {
        'nama': 'Wortel',
        'ciri_fisik': 'Bentuk silindris memanjang dengan warna oranye cerah, tekstur renyah padat.',
        'nutrisi': 'Kaya Vitamin A (Beta-karoten), Vitamin K1, Kalium, dan Serat.',
        'manfaat': 'Menjaga kesehatan penglihatan mata, meningkatkan imunitas, dan menjaga metabolisme tubuh.',
        'penyimpanan': 'Potong daun hijau, simpan di laci kulkas dengan wadah tertutup agar tahan hingga 3-4 minggu.'
    },
    'Cucumber': {
        'nama': 'Mentimun',
        'ciri_fisik': 'Bentuk memanjang silindris berwarna hijau, berdaging renyah dengan kadar air tinggi.',
        'nutrisi': 'Mengandung kadar air hingga 95%, Vitamin K, Vitamin C, dan Mineral Magnesium.',
        'manfaat': 'Mencegah dehidrasi, membantu melancarkan pencernaan, dan meredakan peradangan.',
        'penyimpanan': 'Bungkus rapat dengan tisu atau kertas kering, simpan pada bagian kulkas yang tidak terlalu dingin.'
    },
    'Tomato': {
        'nama': 'Tomat',
        'ciri_fisik': 'Bentuk bulat mengilap, berwarna merah cerah saat matang, berair dengan rasa segar.',
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
        <div style="min-height: 280px;"></div>
        <div style="text-align: center; border-top: 1px solid #cbd5e1; padding-top: 16px;">
            <span class="sidebar-footer-badge">Oleh : Kelompok 6</span>
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- HALAMAN 1: UNGGAH & PREDIKSI ---
if menu == "📤 Unggah & Prediksi":
    st.markdown('<div class="clean-title">🥦 Klasifikasi Citra Sayuran</div>', unsafe_allow_html=True)
    st.markdown("---")

    if model is None:
        st.error("Model tidak ditemukan atau gagal diunduh. Pastikan link Google Drive sudah diatur publik.")
    else:
        # Pilihan Metode Masukan Citra (Unggah Berkas atau Kamera)
        input_source = st.radio(
            "Pilih Metode Masukan Gambar:",
            options=["📁 Unggah Gambar dari Galeri / Berkas", "📷 Ambil Foto Langsung Lewat Kamera"],
            horizontal=True
        )

        image_data = None
        source_name = "Kamera"

        if input_source == "📁 Unggah Gambar dari Galeri / Berkas":
            uploaded_file = st.file_uploader(
                "Pilih atau seret gambar sayuran (.jpg, .jpeg, .png)", 
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                image_data = uploaded_file
                source_name = uploaded_file.name
        else:
            camera_file = st.camera_input("Arahkan kamera ke sayuran lalu klik tombol ambil gambar")
            if camera_file is not None:
                image_data = camera_file
                source_name = "Foto_Kamera_Langsung.png"

        if image_data is not None:
            st.write("")
            col1, col2 = st.columns([1, 1], gap="large")
            
            # Kolom Kiri: Pratinjau Citra Masukan
            with col1:
                img_display = Image.open(image_data).convert("RGB")
                st.image(img_display, caption="Pratinjau Citra Sayuran", use_container_width=True)
                run_btn = st.button("🔍 JALANKAN PREDIKSI", use_container_width=True)

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
                        
                        # Kartu Hasil - Teks Hitam
                        st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
                                border: 1px solid #7dd3fc;
                                border-radius: 12px;
                                padding: 18px 22px;
                                margin-bottom: 12px;
                                box-shadow: 0 2px 6px rgba(14, 165, 233, 0.1);
                            ">
                                <span style="font-size: 0.85rem; font-weight: 800; color: #0f172a; text-transform: uppercase; letter-spacing: 0.8px;">Jenis Sayuran Terdeteksi</span>
                                <div style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin-top: 4px;">
                                    {detail['nama']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        # Kartu Tingkat Keyakinan
                        st.markdown(f"""
                            <div style="
                                background: #f8fafc;
                                border: 1px solid #cbd5e1;
                                border-radius: 10px;
                                padding: 12px 18px;
                                margin-bottom: 16px;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            ">
                                <span style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Tingkat Keyakinan:</span>
                                <span style="
                                    background-color: #0f172a;
                                    color: #ffffff !important;
                                    font-weight: 800;
                                    font-size: 1.05rem;
                                    padding: 4px 14px;
                                    border-radius: 20px;
                                ">{confidence:.2f}%</span>
                            </div>
                        """, unsafe_allow_html=True)

                        # Kartu Informasi Gizi & Manfaat
                        detail_col1, detail_col2 = st.columns(2)
                        with detail_col1:
                            st.markdown(f"""
                                <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px; min-height: 130px;">
                                    <strong style="color: #0f172a;">🥗 Kandungan Nutrisi:</strong><br>
                                    <span style="font-size: 0.9rem; color: #1e293b;">{detail['nutrisi']}</span>
                                </div>
                            """, unsafe_allow_html=True)
                        with detail_col2:
                            st.markdown(f"""
                                <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px; min-height: 130px;">
                                    <strong style="color: #0f172a;">💡 Manfaat Kesehatan:</strong><br>
                                    <span style="font-size: 0.9rem; color: #1e293b;">{detail['manfaat']}</span>
                                </div>
                            """, unsafe_allow_html=True)

                        st.write("")
                        st.markdown(f"""
                            <div style="background: #f1f5f9; border-left: 4px solid #0f172a; padding: 12px 16px; border-radius: 6px;">
                                <strong style="color: #0f172a;">❄️ Tips Penyimpanan:</strong><br>
                                <span style="font-size: 0.9rem; color: #1e293b;">{detail['penyimpanan']}</span>
                            </div>
                        """, unsafe_allow_html=True)

                        # Grafik Probabilitas
                        st.write("")
                        st.subheader("Distribusi Skor Keyakinan")
                        df_chart = pd.DataFrame({
                            'Sayuran': ['Wortel', 'Mentimun', 'Tomat'],
                            'Probabilitas (%)': [round(p * 100, 2) for p in preds]
                        }).set_index('Sayuran')
                        st.bar_chart(df_chart)

                        # Detail Teknis Gambar
                        with st.expander(f"🔍 Detail Citra ({source_name})"):
                            st.write(f"- **Sumber Citra:** `{source_name}`")
                            st.write(f"- **Dimensi Citra Asli:** `{img_display.size[0]} × {img_display.size[1]} piksel`")
                            st.write(f"- **Resolusi Masukan Model:** `150 × 150 piksel (RGB)`")

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
