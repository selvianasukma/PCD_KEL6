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
        border: 1px solid #
