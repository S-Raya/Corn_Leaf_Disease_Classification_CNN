import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_trained_model():
    model = load_model('CornDiseaseClassifierModel3.h5')
    return model

model = load_trained_model()

# Daftar kelas penyakit
class_names = [
    "Blight",
    "Common Rust",
    "Gray Leaf Spot",
    "Healthy"
]

st.title("Prediksi Penyakit Daun Jagung dengan MobileNetV2")

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar Daun Jagung", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array_norm = img_array / 255.0  
    img_array_expanded = np.expand_dims(img_array_norm, axis=0) 

    img_resized_to_show = np.uint8(img_array_norm * 255)
    st.image(img_resized_to_show, caption="", use_container_width=True)

    # Prediksi kelas penyakit
    prediction = model.predict(img_array_expanded)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]

    # Tampilkan hasil prediksi
    st.write(f"**Prediksi Penyakit:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")
