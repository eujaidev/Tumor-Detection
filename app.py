# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import cv2
# import numpy as np
# from PIL import Image

# model = load_model('/home/redleaf/Documents/DUK/Deep Learning/deep-learning/CNN/tumor_detection/results/model/cnn_tumor.keras')

# def make_prediction(img,model):
#     img=cv2.imread(img)
#     img=Image.fromarray(img)
#     img=img.resize((128,128))
#     img=np.array(img)
#     input_img = np.expand_dims(img, axis=0)
#     res = model.predict(input_img)
#     if res:
#         print("Tumor Detected")
#     else:
#         print("No Tumor")
#     return res

# st.title("Tumor Classification")
# st.write("Upload an image to classify if it has a tumor or not.")

# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     img = Image.open(uploaded_file)
#     st.image(img, caption="Uploaded Image.", use_column_width=True)
    
#     st.write("Classifying...")
#     label = make_prediction(img,model)
#     st.write(f"Prediction: Tumor - {label}")

# ------------------------------------------------------------------------------------------------------------------------------------------
# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image

# # Load your model
# model = load_model('/home/redleaf/Documents/DUK/Deep Learning/deep-learning/CNN/tumor_detection/results/model/cnn_tumor.keras')

# # Define the prediction function
# def make_prediction(img, model):
#     img = img.resize((128, 128))  # Resize image to match model input size
#     img = np.array(img)  # Convert to NumPy array
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     img = img / 255.0  # Normalize pixel values
#     res = model.predict(img)  # Make prediction
#     return res[0][0]  # Return the first result

# # Streamlit app
# st.title("Tumor Classification")
# st.write("Upload an image to classify if it has a tumor or not.")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     img = Image.open(uploaded_file)  # Open the uploaded image
#     st.image(img, caption="Uploaded Image.", use_column_width=True)
    
#     st.write("Classifying...")
#     label = make_prediction(img, model)  # Predict tumor
#     if label >= 0.5:
#         st.write("Prediction: Tumor Detected")
#     else:
#         st.write("Prediction: No Tumor")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('/home/redleaf/Documents/DUK/Deep Learning/deep-learning/CNN/tumor_detection/results/model/cnn_tumor.keras')

# Define the prediction function
def make_prediction(img, model):
    img = img.resize((128, 128))  # Resize image to model input size
    img = np.array(img)  # Convert image to NumPy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    res = model.predict(img)  # Get the model's prediction
    return res[0][0]  # Return probability for class

# Set page configuration for better UI
st.set_page_config(page_title="Tumor Classification", page_icon="🧠", layout="centered")

# Add custom CSS to style the UI
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stApp {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #1f77b4;
        }
        h2 {
            text-align: center;
            color: #ff7f0e;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #f0f2f6;
        }
    </style>
    """, unsafe_allow_html=True)

# Page title and subtitle
st.title("🧠 Tumor Detection CNN Model")
st.write("Upload an MRI scan to check for the presence of a tumor.")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an MRI scan image (JPG format)", type=["jpg", "jpeg", "png"])

# If a file is uploaded
if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI scan", use_column_width=True)

    # Add a submit button
    if st.button("Submit for Classification"):
        st.write("Analyzing the image...")

        # Make prediction
        label = make_prediction(img, model)

        # Display the result with enhanced UI
        if label >= 0.5:
            st.markdown(
                "<h2 style='color: red;'>🚨 Tumor Detected! 🚨</h2>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='color: green;'>✅ No Tumor Detected!</h2>", 
                unsafe_allow_html=True
            )

# Footer message
st.markdown(
    "<div class='footer'>Built with ❤️ using TensorFlow and Streamlit</div>", 
    unsafe_allow_html=True
)

