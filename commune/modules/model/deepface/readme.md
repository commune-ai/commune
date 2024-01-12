# deepface v8 Module - README

## Introduction

[DeepFace](https://viso.ai/computer-vision/deepface/) is the most lightweight face recognition and facial attribute analysis library for Python. The open-sourced DeepFace library includes all leading-edge AI models for face recognition and automatically handles all procedures for facial recognition in the background.

If you run face recognition with DeepFace, you get access to a set of features: 
- Face Verification: The task of face verification refers to comparing a face with another to verify if it is a match or not. Hence, face verification is commonly used to compare a candidate’s face to another. This can be used to confirm that a physical face matches the one in an ID document. 

- Face Recognition: The task refers to finding a face in an image database. Performing face recognition requires running face verification many times. 

- Facial Attribute Analysis: The task of facial attribute analysis refers to describing the visual properties of face images. Accordingly, facial attributes analysis is used to extract attributes such as age, gender classification, emotion analysis, or race/ethnicity prediction. 

- Real-Time Face Analysis: This feature includes testing face recognition and facial attribute analysis with the real-time video feed of your webcam.


## How to Use DeepFace Module?

### Verify

`c model.deepface verify img1_path='image1.png' img2_path='image2.png'`

    This function verifies an image pair is same person or different persons. In the background, verification function represents facial images as vectors and then calculates the similarity between those vectors. Vectors of same person images should have more similarity (or less distance) than vectors of different persons.

### Analyze

`c model.deepface analyze img_path='image.png'`

    This function analyzes facial attributes including age, gender, emotion and race. In the background, analysis function builds convolutional neural network models to classify age, gender, emotion and race of the input image.
    
### Find

`c model.deepface find img_path='image.png' db_path="db/"`

    This function applies verification several times and find the identities in a database

### Represent

`c model.deepface represent img_path='image.png'`

    This function represents facial images as vectors. The function uses convolutional neural networks models to generate vector embeddings.

### Stream

`c model.deepface stream db_path='db/'`

    This function applies real time face recognition and facial attribute analysis

### Extract faces

`c model.deepface extract_faces img_path='image.png'`

    This function applies pre-processing stages of a face recognition pipeline including detection and alignment

## Fine-tuning for one person face

### Save one person face

`c model.deepface save_face img_src_dir='images' model_filename='one_person.h5'`

    This function fine-tunes the model using individual face images. You can upload one or multiple face images of the same person for fine-tuning. This process amplifies the data, especially when you have a small amount of it.

### Verify one person face

`c model.deepface verify face_image_path='image.jpeg' model_filename='one_person.h5'`

    This function verifies an individual's face using the fine-tuned model. You can specify the image path and select the model file saved in the 'save_face' function.

***You can refer the source code to check parameters.***

## Gradio UI

`c model.deepface gradio`

Testing the module on Gradio UI.