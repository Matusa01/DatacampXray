#Importation of libraries
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator #image generator label data based on the dir the image in contained in
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import load_img, img_to_array 
from tensorflow.keras.models import Model
from keras.preprocessing import image
from PIL import Image
import os

# Index of the page
st.set_page_config(layout="wide", page_title="Pneumonie_Detection", page_icon="🫁")

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
  uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)

  train_normal_dir = os.path.join('../Data Camp/train/NORMAL')
  train_pneumonia_dir = os.path.join('../Data Camp/train/PNEUMONIA')
  test_normal_dir = os.path.join('../Data Camp/test/NORMAL')
  test_pneumonia_dir = os.path.join('../Data Camp/test/PNEUMONIA')

  train_normal_names = os.listdir(train_normal_dir)
  train_pneumonia_names = os.listdir(train_pneumonia_dir)
  test_normal_names = os.listdir(test_normal_dir)
  test_pneumonia_names = os.listdir(test_pneumonia_dir)

  model = tf.keras.models.Sequential([
  
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
  
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
  
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
  
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
  
    # The fifth convolution
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

  
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for ('normal') clas and 1 for ('pneumonia') class
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

  model.summary()
  model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])
  train_datagen = ImageDataGenerator(rescale = 1/255)
  test_datagen = ImageDataGenerator(rescale = 1/255)

  train_generator = train_datagen.flow_from_directory(
        '../Data Camp/train/',
        target_size = (300,300),
        batch_size = 128,
        class_mode = 'binary'
    )

  validation_generator = test_datagen.flow_from_directory(
        '../Data Camp/test/',
        target_size = (300, 300),
        batch_size = 128,
        class_mode = 'binary'
    )
  
  history = model.fit(
      train_generator,
      steps_per_epoch = 10,
      epochs = 10,
      validation_data = validation_generator
    )

    # load new unseen dataset
  test_datagen = ImageDataGenerator(rescale = 1/255)

  test_generator = test_datagen.flow_from_directory(
      '../Data Camp/test',
      target_size = (300, 300),
      batch_size = 5, 
      class_mode = 'binary'
  )

  for i in uploaded_file:
      image_path = '../Data Camp' + i 
      img = keras.utils.load_img(image_path, target_size=(100, 100))
      x = keras.utils.img_to_array(img)
      x = np.expand_dims(x, axis =0)
      images = np.vstack([x])
      prediction = model.predict(images, batch_size = 10)
      if prediction[0]> 0.5:
          st.write("The person has a pneumonia")
          st.image(img)
      else:
          st.write("The person has not pneumonia")
          st.image(img)
          