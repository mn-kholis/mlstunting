import streamlit as st
from utils import predict_all

st.set_page_config(page_title="Stunting Classifier", layout="centered")

st.sidebar.title("🧭 Menu")
menu = st.sidebar.radio("Pilih menu:", ["Main Menu", "Prediksi"])
st.sidebar.markdown("---")
st.sidebar.markdown("[📂 Dataset](https://docs.google.com/spreadsheets/d/1vHvOe_JRHC73PJC4UtcBdAnSutTukW59v6vbBg-jjg0/edit?usp=sharing)")
st.sidebar.markdown("[📂 Github](https://docs.google.com/spreadsheets/d/1vHvOe_JRHC73PJC4UtcBdAnSutTukW59v6vbBg-jjg0/edit?usp=sharing)")

if menu == "Main Menu":
    st.title("🧒 Prediksi Stunting dan Berat Badan")
    #st.image("assets/logo.png", width=200)
    st.markdown("""
    Aplikasi ini memprediksi:
    - **Status stunting** (kategori)
    - **Status berat badan** (deskripsi)
    """)

elif menu == "Prediksi":
    st.title("🤖 Form Prediksi")

    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    umur = st.number_input("Umur (bulan)", min_value=0, max_value=60)
    tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=130.0)
    berat = st.number_input("Berat Badan (kg)", min_value=2.0, max_value=30.0)
    if st.button("Prediksi"):
        hasil_kategori, hasil_deskripsi = predict_all([jenis_kelamin, umur, tinggi, berat])

        # Tambahkan ikon berdasarkan kategori
        ikon_kategori = "🟢" if hasil_kategori == "Normal" else "🔴"
        ikon_deskripsi = "⚖️" if "Normal" in hasil_deskripsi else "📉"

        st.markdown("---")
        st.markdown("## 🎯 Hasil Prediksi")
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
                <div class="hasil-title">🧬 Status Stunting (Kategori)</div>
                <div class="hasil-value">{ikon_kategori} {hasil_kategori}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="hasil-box deskripsi-box">
                <div class="hasil-title">⚖️ Status Berat Badan (Deskripsi)</div>
                <div class="hasil-value">{ikon_deskripsi} {hasil_deskripsi}</div>
            </div>
        """, unsafe_allow_html=True)