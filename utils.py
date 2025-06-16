import pickle
import numpy as np

def load_all():
    with open("knn_kategori_model.pkl", "rb") as f:
        model_kategori = pickle.load(f)
    with open("knn_deskripsi_model.pkl", "rb") as f:
        model_deskripsi = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    return model_kategori, model_deskripsi, scaler, label_encoders

def predict_all(input_data):
    model_kategori, model_deskripsi, scaler, label_encoders = load_all()

    #encode jenis kelamin
    input_encoded = input_data.copy()
    input_encoded[0] = label_encoders['Jenis Kelamin'].transform([input_data[0]])[0]

    scaled = scaler.transform([input_encoded])

    #prediksi
    pred_kategori = model_kategori.predict(scaled)[0]
    pred_deskripsi = model_deskripsi.predict(scaled)[0]

    hasil_kategori = label_encoders['kategori'].inverse_transform([pred_kategori])[0]
    hasil_deskripsi = label_encoders['deskripsi'].inverse_transform([pred_deskripsi])[0]

    return hasil_kategori, hasil_deskripsi