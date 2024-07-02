import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
import joblib

# Muat model saat server dimulai
model_path = r'D:\Lainnya\Rice Classification\core\models\rice_model.pkl'
model = joblib.load(model_path)

# Daftar label
labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

def classify_image(img_path):
    # Muat gambar
    img = image.load_img(img_path, target_size=(100, 100))  # Sesuaikan ukuran gambar sesuai dengan model kamu
    img_array = image.img_to_array(img)
    
    # Normalisasi gambar
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    img_array = preprocess_input(img_array)  # Gunakan preprocess_input sesuai dengan model yang digunakan

    # Prediksi menggunakan model
    prediction = model.predict(img_array)[0]  # Ambil array dari hasil prediksi
    
    # Mengubah prediksi menjadi persentase
    percentages = (prediction / np.sum(prediction)) * 100
    
    # Membuat dictionary hasil prediksi
    results = {}
    for idx, label in enumerate(labels):
        results[label] = percentages[idx]
    
    return results

def predict(request):
    if request.method == 'POST' and request.FILES['image']:
        img_file = request.FILES['image']
        file_name = default_storage.save(img_file.name, img_file)
        img_path = os.path.join(default_storage.location, file_name)
        
        try:
            # Lakukan klasifikasi gambar
            predicted_percentages = classify_image(img_path)
            # Temukan label dengan persentase tertinggi
            highest_label = max(predicted_percentages, key=predicted_percentages.get)
            highest_percentage = predicted_percentages[highest_label]
        finally:
            # Hapus file setelah klasifikasi
            default_storage.delete(file_name)
        
        return render(request, 'result.html', {
            'predicted_percentages': predicted_percentages,
            'highest_label': highest_label,
            'highest_percentage': highest_percentage
        })
    return render(request, 'index.html')

def index(request):
    return render(request, 'index.html')
