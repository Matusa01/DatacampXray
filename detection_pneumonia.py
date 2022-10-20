#Importation of libraries
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import load_img, img_to_array 
from tensorflow.keras.models import Model
from keras.preprocessing import image
from PIL import Image
import os

# Index of the page
st.set_page_config(layout="wide", page_title="Pneumonie_Detection", page_icon="🫁")

#Sidebar
chart_visual = st.sidebar.selectbox('Select', 
                                    ('Home','Detection of pneumonia', 'About us'))

if chart_visual == 'Home':
  st.title('Pneumonia detection 🫁')
  st.caption(
      "A Streamlit web app by [Rukshini Rasakumaran](https://www.linkedin.com/in/rukshini-rasakumaran/), [Matusa Manohoran](https://www.linkedin.com/in/matusa-manoharan-31b392194/) and [Valentin Roy](https://www.linkedin.com/in/valentinroy94/)"
  )
  st.markdown('Pneumonia is an infection of the lungs, most often caused by a virus or bacteria. Specifically, the infection affects the lung alveoli, the tiny balloon-like sacs at the ends of the bronchioles (see diagram below). It usually affects only one of the 5 lobes of the lung (3 lobes in the right lung and 2 in the left), hence the term lobar pneumonia. When pneumonia also affects the bronchioles, it is called bronchopneumonia.')

  video_file = open('Video_pneumonia.mp4', 'rb')
  video_bytes = video_file.read()
  st.video(video_bytes)


if chart_visual == 'Detection of pneumonia':
  #Upload file
  uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)

  #Path of each files
  train_normal = os.path.join('./train_cloud/NORMAL_cloud') 
  train_pneumonia = os.path.join('./train_cloud/PNEUMONIA_cloud')
  test_normal = os.path.join('./test_cloud/NORMAL_cloud')
  test_pneumonia = os.path.join('./test_cloud/PNEUMONIA_cloud')

  train_normal_names = os.listdir(train_normal)
  train_pneumonia_names = os.listdir(train_pneumonia)
  test_normal_names = os.listdir(test_normal)
  test_pneumonia_names = os.listdir(test_pneumonia)

  model = tf.keras.models.Sequential([
    
      #1st concolution
      tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
      tf.keras.layers.MaxPooling2D(2, 2),

      #2nd convolution
      tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
    
      #3rd convolution
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
    
      #4thconvolution
      tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
    
      #5th convolution
      tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),

    
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])



  model.summary()
  model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])
  train_datagen = ImageDataGenerator(rescale = 1/255)
  test_datagen = ImageDataGenerator(rescale = 1/255)

  train_generator = train_datagen.flow_from_directory(
        './train_cloud/',
        target_size = (300,300),
        batch_size = 128,
        class_mode = 'binary'
    )

  validation_generator = test_datagen.flow_from_directory(
        './test_cloud/',
        target_size = (300, 300),
        batch_size = 128,
        class_mode = 'binary'
    )
  
  history = model.fit(
      train_generator,
      steps_per_epoch = 5,
      epochs = 5,
      validation_data = validation_generator
    )


  
  if uploaded_file is not None:
      path = "./test_cloud/PNEUMONIA_cloud/"+str(uploaded_file.name)
      img = keras.utils.load_img(path, target_size=(300, 300))
      x = keras.utils.img_to_array(img)
      x = np.expand_dims(x, axis =0)
      images = np.vstack([x])
      prediction = model.predict(images, batch_size = 128)
      print(prediction[0])
      if prediction[0]> 0.5:
          st.write("The person has a pneumonia")
          st.image(img)
      else:
          st.write("The person has not pneumonia")
          st.image(img)
          
