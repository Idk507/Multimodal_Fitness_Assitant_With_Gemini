import streamlit as st
import google.generativeai as genai
from PIL import Image
from google.generativeai import models
from IPython.display import Markdown

# Configure Google Generative AI
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit app title and description
st.title('BRATZLIFE - Fitness Image Assistance App')
st.write("""
Upload your fitness-related images and get personalized advice on your workout form, equipment usage, or tips!
""")

# Upload an image
uploaded_file = st.file_uploader("Upload a fitness image", type=["png", "jpg", "jpeg"])

# Check if an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Ask for user input related to the image
    user_query = st.text_input("Enter your query related to this image")

    # Proceed if user has provided a query
    if user_query:
        # Upload the file to Google Generative AI (mock function for example purposes)
        sample_file = genai.upload_file(path=uploaded_file.name, display_name="Image")
        st.write(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")

        # Prompt for the Generative Model
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        prompt = f"Using the uploaded image and the query '{user_query}', provide fitness-related advice or suggestions. The image shows a fitness-related activity or equipment. Assist the user with workout tips, form correction, or equipment usage guidance."

        # Generate a response from the AI model
        response = model.generate_content([sample_file, prompt])
        
        # Display the AI response
        st.markdown(f"> {response.text}")
