import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = None
labels = ['real', 'fake']

def load_model():
    global model
    model = tf.keras.models.load_model('C:/Users/vamsh/OneDrive/Desktop/Deforgify/Deforgify/Model Training/fakevsreal_weights.h5')

def classify_image(file):
    if model is None:
        load_model()

    image = Image.open(file)  # reading the uploaded image
    image = image.resize((128, 128))  # resizing the image to fit the trained model
    image = image.convert("RGB")  # converting the image to RGB
    img = np.asarray(image)  # converting it to numpy array
    img = np.expand_dims(img, 0)
    predictions = model.predict(img)  # predicting the label
    label = labels[np.argmax(predictions[0])]  # extracting the label with maximum probability
    probab = float(round(predictions[0][np.argmax(predictions[0])] * 100, 2))

    return label, probab

# Streamlit app
def main():
    st.title('Deep Fake Detection')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Check if prediction button is clicked
        if st.button('Detect Deep Fake'):
            # Classify the uploaded image
            label, probability = classify_image(uploaded_file)

            # Display prediction result
            st.write(f"Prediction: {label}")
            st.write(f"Probability: {probability}%")
    st.markdown('---')
    st.write('© 2024 Deep Fake Detection Project')
    st.write('Made by Vamshi Nanduri and Divya P')
    st.write('Mentor by Shuvendu Rana')
            

if __name__ == '__main__':
    main()

