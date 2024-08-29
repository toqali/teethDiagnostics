import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf  # or any other library you're using for the model

# Load the saved model
model_path = 'mobile_net.h5'  
model = tf.keras.models.load_model(model_path)  

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize if required by your model
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Welcome statement
st.title("🦷🔬  Advanced Classification for Teeth Health 🦷🪥")

# Add space
st.write("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

# Upload image file
uploaded_image = st.file_uploader("**Upload the image 📸👇✨**", type=["jpg", "jpeg", "png"])

# Check if an image file has been uploaded
if uploaded_image is not None:
    # Open the image file
    image = Image.open(uploaded_image).convert('RGB')  # Ensure image is in RGB format
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(preprocessed_image)
    
    # Process the prediction result (e.g., decode the prediction)
    class_names = ['OT', 'CoS', 'MC', 'CaS', 'OC', 'OLP', 'Gum']  
    predicted_class = class_names[np.argmax(prediction)]
    
    # Add space
    st.write("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

    # Display the prediction result
    st.title(f"**Prediction:** {predicted_class}")
