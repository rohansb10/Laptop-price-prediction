import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model and label encoders
file_path_model = r"C:\Users\Rohan\Pictures\rohan\profile projects\laptop\model_save.pkl"

loaded_model = pickle.load(open(file_path_model, 'rb'))

# Streamlit App
st.title('Laptop Specification Prediction')

# Get user inputs through Streamlit widgets
brand = st.text_input("Brand")
cpu_core = st.text_input("CPU Core")
ram = st.text_input("RAM")
ram_type = st.text_input("RAM Type")
rom = st.text_input("ROM")
rom_type = st.text_input("ROM Type")
display_size = st.text_input("Display Size")
os = st.text_input("Operating System")
warranty = st.text_input("Warranty")
processor_generation = st.text_input("Processor Generation")
processor_model = st.text_input("Processor Model")
gpu_category = st.text_input("GPU Category")

# Create a DataFrame with user inputs


brand_mapping = {
    'HP': 0, 'Acer': 1, 'Lenovo': 2, 'Apple': 3, 'Dell': 4,
    'Asus': 5, 'Samsung': 6, 'Ultimus': 7, 'Primebook': 8, 'MSI': 9,
    'Infinix': 10, 'Wings': 11, 'Honor': 12, 'Zebronics': 13, 'Xiaomi': 14,
    'iBall': 15, 'Chuwi': 16, 'Realme': 17, 'Avita': 18, 'Walker': 19,
    'Huawei': 20, 'Tecno': 21, 'Gigabyte': 22, 'Vaio': 23, 'Microsoft': 24,
    'Fujitsu': 25, 'LG': 26, 'Ninkear': 27, 'Razer': 28, 'AXL': 29
}

ram_type_mapping = {
    'DDR4': 0, 'LPDDR5': 1, 'DDR5': 2, 'LPDDR4': 3, 'LPDDR5X': 4,
    'LPDDR4X': 5, 'DDR3': 6, 'LPDDR4x': 7, 'Unified': 8, 'DDR4-': 9,
    'LPDDR5x': 10, 'DDR': 11
}

rom_type_mapping = {
    'SSD': 0, 'Hard-Disk': 1
}

os_mapping = {
    'Windows 11': 0, 'Mac': 1, 'Android 11': 2, 'DOS': 3, 'Windows 10': 4,
    'Chrome': 5, 'Windows': 6, 'Ubuntu': 7, 'Mac Catalina': 8, 'DOS 3.0': 9,
    'Mac High Sierra': 10, 'Mac 10.15.3': 11
}

processor_model_mapping = {
    'AMD Ryzen 5': 0, 'Intel Core i3': 1, 'Intel Core i5': 2, 'Other Processor': 3,
    'Intel Core i7': 4, 'AMD Ryzen 3': 5, 'Intel Core i9': 6, 'AMD Ryzen 7': 7,
    'AMD Ryzen 9': 8
}

gpu_category_mapping = {
    'AMD': 0, 'other': 1, 'NVIDIA': 2
}


user_input = {
    'brand': brand,
    'cpu_core': cpu_core,
    'Ram': ram,
    'Ram_type': ram_type,
    'ROM': rom,
    'ROM_type': rom_type,
    'display_size': display_size,
    'OS': os,
    'warranty': warranty,
    'processor_generation': processor_generation,
    'processor_model': processor_model,
    'GPU_category': gpu_category
}




user_input_df = pd.DataFrame([user_input])
# Apply mappings to the DataFrame
user_input_df['brand'] = user_input_df['brand'].map(brand_mapping)
user_input_df['Ram_type'] = user_input_df['Ram_type'].map(ram_type_mapping)
user_input_df['ROM_type'] = user_input_df['ROM_type'].map(rom_type_mapping)
user_input_df['OS'] = user_input_df['OS'].map(os_mapping)
user_input_df['processor_model'] = user_input_df['processor_model'].map(processor_model_mapping)
user_input_df['GPU_category'] = user_input_df['GPU_category'].map(gpu_category_mapping)


# Make predictions
if st.button('Predict'):
    try:
        prediction = loaded_model.predict(user_input_df)
        st.write(f"Predicted Value: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
