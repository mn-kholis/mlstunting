import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import predict_all  # pastikan tetap ada

st.set_page_config(page_title="Stunting Classifier", layout="centered")

# ==== Tambahkan CSS kustom untuk tombol sidebar ====
st.markdown("""
    <style>
    .custom-link {
        display: block;
        background-color: white;
        color: black !important;
        padding: 0.5em 1em;
        margin-bottom: 0.6em;
        border-radius: 6px;
        text-decoration: none !important;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        transition: all 0.2s ease-in-out;
    }

    .custom-link:hover {
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
        transform: translateY(-1px);
        background-color: #f8f9fa;
        color: black !important;
    }

    .custom-link:active {
        transform: translateY(1px);
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ==== Sidebar ====
st.sidebar.title("🟰 Menu")
menu = st.sidebar.selectbox("Pilih Halaman", ["Homepage", "Prediksi Stunting"])
st.sidebar.markdown("---")

# Tombol dengan gaya seperti gambar
st.sidebar.markdown('<a class="custom-link" href="https://docs.google.com/spreadsheets/d/1vHvOe_JRHC73PJC4UtcBdAnSutTukW59v6vbBg-jjg0/edit?usp=sharing" target="_blank">📂 Dataset</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a class="custom-link" href="https://github.com/mn-kholis/mlstunting" target="_blank">📂 Github</a>', unsafe_allow_html=True)

# ==== Homepage ====
if menu == "Homepage":
    col1, col2 = st.columns([1, 4])  

    with col1:
        st.image("assets/LOGOS.png", width=200)  
    with col2:
        st.title("Prediksi Stunting dan Berat Badan Pada Bayi")

    st.markdown("""
    Aplikasi ini memprediksi:
    - **Status stunting**
    - **Status berat badan**
    
    Berdasarkan jenis kelamin, umur, tinggi badan dan berat badan bayi
    ---
    """)

    st.markdown("""
    ### 📊 Tentang Dataset
    Dataset ini terdiri dari **100.000 data anak usia 0–60 bulan** yang digunakan untuk memprediksi kondisi **stunting** dan **berat badan (wasting)** berdasarkan data antropometri.

    Setiap entri dalam dataset memuat:
    - **Jenis Kelamin**: Laki-laki atau Perempuan  
    - **Umur**: Dalam satuan bulan (0–60 bulan)  
    - **Tinggi Badan**: Dalam cm  
    - **Berat Badan**: Dalam kg  
    - **Status Stunting**: Kategori seperti `Normal`, `Stunted`, `Severely Stunted`, dan `Tall`  
    - **Status Wasting (Kategori Berat Badan)**: Deskripsi seperti `Underweight`, `Normal`, `Overweight`, hingga `Risk of Overweight`

    Data ini menjadi dasar pelatihan model machine learning yang dapat membantu memprediksi kondisi **gizi buruk** secara dini pada anak balita, sebagai upaya mendukung program pencegahan stunting di Indonesia.
    """)

    df = pd.read_csv("stunting.csv")

    st.markdown("### Distribusi Kategori Stunting")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Stunting", order=df["Stunting"].value_counts().index, ax=ax1)
    ax1.set_ylabel("Jumlah")
    st.pyplot(fig1)

    st.markdown("### Distribusi Deskripsi Berat Badan")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="Wasting", order=df["Wasting"].value_counts().index, ax=ax2)
    ax2.set_ylabel("Jumlah")
    st.pyplot(fig2)

    st.markdown("### Distribusi Umur")
    fig3, ax3 = plt.subplots()
    sns.histplot(df["Umur (bulan)"], kde=True, bins=20, ax=ax3)
    ax3.set_xlabel("Umur (bulan)")
    ax3.set_ylabel("Frekuensi")
    st.pyplot(fig3)

    st.markdown("### Tinggi Badan vs Status Kategori Stunting")
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=df, x="Stunting", y="Tinggi Badan (cm)", ax=ax4)
    st.pyplot(fig4)

    st.markdown("### Berat Badan vs Status Deskripsi")
    fig5, ax5 = plt.subplots()
    sns.boxplot(data=df, x="Wasting", y="Berat Badan (kg)", ax=ax5)
    st.pyplot(fig5)

    st.markdown("### Korelasi Antar Fitur Numerik")
    corr = df[["Umur (bulan)", "Tinggi Badan (cm)", "Berat Badan (kg)"]].corr()
    fig6, ax6 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax6)
    st.pyplot(fig6)

    st.markdown("""
    Hasil evaluasi dari dua model machine learning yang digunakan untuk memprediksi:
    - **Kategori status stunting**: Normal, Stunted, Severely Stunted, dan Tall.
    - **Deskripsi berat badan**: Normal weight, Underweight, Severely Underweight, dan Risk of Overweight.

    Evaluasi mencakup metrik **precision, recall, f1-score**, dan **confusion matrix**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Model Kategori (Stunting)")
        st.image("assets/kategori.png", caption="Confusion Matrix & Classification Report - Kategori", use_column_width=True)
        st.success("""
        ✅ **Akurasi Model:** 97%  
        🔍 **Macro Avg Precision/Recall/F1:** 93%
        """)

    with col2:
        st.subheader("📌 Model Deskripsi (Berat Badan)")
        st.image("assets/deskripsi.png", caption="Confusion Matrix & Classification Report - Deskripsi", use_column_width=True)
        st.success("""
        ✅ **Akurasi Model:** 98%  
        🔍 **Macro Avg Precision/Recall/F1:** 97%
        """)

    st.markdown("---")
    st.markdown("Model dievaluasi menggunakan data validasi untuk memastikan performa dalam memprediksi kondisi stunting dan berat badan anak dengan akurat.")

# ==== Form Prediksi ====
elif menu == "Prediksi Stunting":
    st.title("📋 Form Prediksi Stunting")

    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    umur = st.number_input("Umur (bulan)", min_value=0, max_value=60)
    tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=130.0)
    berat = st.number_input("Berat Badan (kg)", min_value=2.0, max_value=30.0)

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            color: #28a745;
            border: 2px solid #28a745;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
            background-color: transparent;
            transition: all 0.3s ease;
        }

        div.stButton > button:first-child:hover {
            background-color: #28a745;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("Prediksi"):
        hasil_kategori, hasil_deskripsi = predict_all([jenis_kelamin, umur, tinggi, berat])

        ikon_kategori = "🟢" if hasil_kategori == "Normal" else "🔴"
        if "Normal" in hasil_deskripsi:
            ikon_deskripsi = "🟢"
        else:
            ikon_deskripsi = '<span style="color:red">🔴</span>'

        st.markdown("---")
        st.markdown("## 📑 Hasil Prediksi")
        st.markdown("""
            <style>
            .hasil-box {
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 15px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .kategori-box {
                background-color: #eaf4ff;
                border-left: 5px solid #1890ff;
            }
            .deskripsi-box {
                background-color: #f6ffed;
                border-left: 5px solid #52c41a;
            }
            .hasil-title {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 8px;
                color: black !important;
            }
            .hasil-value {
                font-size: 22px;
                font-weight: bold;
                color: black;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="hasil-box kategori-box">
                <div class="hasil-title">🧬 Status Stunting</div>
                <div class="hasil-value">{ikon_kategori} {hasil_kategori}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="hasil-box deskripsi-box">
                <div class="hasil-title">⚖️ Status Berat Badan</div>
                <div class="hasil-value">{ikon_deskripsi} {hasil_deskripsi}</div>
            </div>
        """, unsafe_allow_html=True)
