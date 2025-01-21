#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install -r requirements.txt

import numpy as np
import pickle
import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
from tensorflow import keras


# In[4]:


#loaded_model = pickle.load(open('C:/Users/ShaharyarAmjad/Downloads/face mask detection deployment model/face_mask_detection_model.sav','rb')

def load_model_from_drive(url):
    response = requests.get(url)
    with open("face_mask_detection_model.sav", "wb") as f:
        f.write(response.content)
    
    with open("face_mask_detection_model.sav", "rb") as f:
        return pickle.load(f)

model_url = "https://drive.google.com/uc?export=download&id=1UPCuyxIouOYMeQkrfBEO9OQ9TpwWo3u6"  # Replace <FILE_ID> with your file ID
st.write("Loading model from Google Drive...")
loaded_model = load_model_from_drive(model_url)
st.write("Model loaded successfully!")


                           


# In[ ]:


def face_detection(input_image):

    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
        
    #input_image = cv2.imread(input_image_path)
    #plt.imshow(input_image)
    #input_image_resized = cv2.resize(input_image, (128,128))
    
    # Convert the image to a Pillow Image object
    pil_image = Image.fromarray(input_image)

    # Resize the image using Pillow
    input_image_resized = pil_image.resize((128, 128))  # Resize to 128x128

    # Convert the resized image back to a NumPy array (if needed)
    input_image_resized = np.array(input_image_resized)
    
    input_image_scaled = input_image_resized/255
    
    input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])
    
    input_prediction = loaded_model.predict(input_image_reshaped)
    
    print(input_prediction)
    
    input_pred_label = np.argmax(input_prediction)
    
    if input_pred_label == 1:
      return 'The person in the image is wearing a mask'
    else:
      return 'The person in the image is not wearing a mask'                           
        


# In[ ]:


def main():

    #giving a title
    st.title('Face mask detection web app')

    #getting the input from user
    image = st.file_uploader('Please upload the image', type=["jpg", "jpeg", "png"])
    
    if image is not None:
        # Open image using PIL
        image = Image.open(image)
    
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        #prediction
        conclusion = ''

        #creating a button for prediction
        if st.button('Image result'):
            conclusion = face_detection(image)

        st.success(conclusion)

if __name__ == '__main__':
    main()

