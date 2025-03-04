import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import load_model

import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np

import numpy as np
import tensorflow as tf

def model_prediction(image_input):
    # Ensure the image is in the correct format
    if isinstance(image_input, str):  
        # If input is a file path, load and preprocess
        image = tf.keras.preprocessing.image.load_img(image_input, target_size=(224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
    elif isinstance(image_input, np.ndarray):
        # If input is already a NumPy array, resize to match model's expected shape
        image_array = tf.image.resize(image_input, (224, 224)).numpy()
    else:
        raise TypeError("Invalid input type. Must be a file path (string) or NumPy array.")

    # Normalize image (convert to range 0-1)
    image_array = image_array / 255.0  

    # Expand dimensions to match (None, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)  # Shape will be (1, 224, 224, 3)

    # Remove any additional dimensions (fixes (1, 1, 224, 224, 3) issue)
    if image_array.shape[1] == 1:
        image_array = np.squeeze(image_array, axis=1)

    # Load model and predict
    model = tf.keras.models.load_model("C:/Users/Windows/Downloads/ELPHABA.keras")
    predictions = model.predict(image_array)

    # Get the predicted class index
    result_index = np.argmax(predictions)  # Extracts the class with the highest probability

    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Skin Disease Analysis"])

#Home Page
if (app_mode=="Home"):
    image_path = "LOGO.png"
    st.image(image_path, use_column_width=True)
    st.markdown("## Smart Skin Insights, Healthier Tomorrow!")
    st.write(
        "Our mission is to assist in identifying skin diseases quickly and accurately. "
        "Upload an image of your skin, and our system will analyze it to detect potential conditions. "
        "Together, let's promote better skin health!"
    )

    st.markdown("### How Elphie Works")
    st.write("1. **Upload Image:** Go to the **Skin Disease Analysis** page and upload an image of the affected skin area.")
    st.write("2. **Analysis:** Our deep learning model processes the image to identify potential skin conditions.")
    st.write("3. **Instant Results:** Get a quick assessment and insights to help you make informed decisions.")

    st.markdown("### Why Choose Elphie")
    st.write("- **Accuracy:** Elphie is built using state-of-the-art CNN models for precise skin disease detection.")
    st.write("- **Easy to Use:** Elphie utilizes intuitive interfaces for a hassle-free experience.")
    st.write("- **Fast & Reliable:** You will receive real-time results to take action quickly.")


# Prediction Page
elif app_mode == "Skin Disease Analysis":
    st.header("Skin Disease Analysis")
    test_image = st.file_uploader("Choose an Image:", type=["png", "jpg", "jpeg"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        # Convert image for model prediction
        image = Image.open(test_image)
        image = image.resize((128, 128))  # Resize to match model input shape
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict Button
        if st.button("Predict"):
            with st.spinner("Please Wait.."):
                result_index = model_prediction(image_array)  # Pass processed image

                # Define Class
                class_name = ['Acne', 'Chickenpox', 'Eczema', 'Healthy', 'Heat rash', 'Measles']
                st.success(f"Model is Predicting it's {class_name[result_index]}")
        