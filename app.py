
# Import libraries
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#loading model from tensorflow hub
model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

#performing style transfer using the loaded model and returning the stylized image
def perform_style_transfer(content_image, style_image, style_strength=0 ):
    # Resize the images to the desired dimensions
    content_image = content_image.resize((256, 256))
    style_image = style_image.resize((256, 256))

    # Convert the images to arrays and normalize to float values
    content_array = np.array(content_image).astype(np.float32) / 255.0
    style_array = np.array(style_image).astype(np.float32) / 255.0

    # Add a batch dimension to the input tensors
    content_array = np.expand_dims(content_array, axis=0)
    style_array = np.expand_dims(style_array, axis=0)

    # Convert the input tensors to float tensors
    content_tensor = tf.constant(content_array, dtype=tf.float32)
    style_tensor = tf.constant(style_array, dtype=tf.float32)

    # Perform neural style transfer using the loaded model
    stylized_array = model(content_tensor, style_tensor)[0]

    # Convert the stylized array to a numpy array with uint8 data type
    stylized_array = np.squeeze(stylized_array, axis=0)
    stylized_array = np.clip(stylized_array, 0, 1) * 255
    stylized_array = stylized_array.astype(np.uint8)

    # Convert the stylized array to an image
    stylized_image = Image.fromarray(stylized_array)

    return stylized_image






# Streamlit app 
def main():
    st.title("Neural Style Transfer App")

    # Display an example image
    image_url = "https://media.licdn.com/dms/image/C4E12AQEfjA-SVxYLVQ/article-cover_image-shrink_600_2000/0/1531630356496?e=2147483647&v=beta&t=kmO2CHjqruhnAASb4Ejpu5-GKwe-7L7HjYbwZD2N4oY"
    st.image(image_url, caption="Example Image", use_column_width=True)

    # Allow the user to upload the content and style images
    content_image = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png", "jfif"])
    style_image = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png", "jfif"])

    # Process the images and display the result
    if content_image is not None and style_image is not None:
        # Convert the uploaded images to PIL format
        content_img = Image.open(content_image)
        style_img = Image.open(style_image)

        # Display the uploaded images
        col1, col2, col3 = st.columns(3)  # Create three columns for image display
        col1.image(content_img, caption="Content Image", use_column_width=True)
        col2.image(style_img, caption="Style Image", use_column_width=True)

        # Perform neural style transfer
        if st.button("Apply Style"):
            with st.spinner("Styling in progress..."):
                stylized_img = perform_style_transfer(content_img, style_img)

            # Display the stylized image
            col3.image(stylized_img, caption="Stylized Image", use_column_width=True)

if __name__ == "__main__":
    main()


