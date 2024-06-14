from deepface import DeepFace
import numpy as np
import cv2
from PIL import Image
import commune as c
import gradio as gr
from io import BytesIO
import base64
import os

MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "Dlib",
          "ArcFace", "SFace"]

DETECTORS = ["mtcnn", "retinaface", "opencv", "ssd",
             "dlib", "mediapipe", "yolov8"]

METRICS = ["cosine", "euclidean", "euclidean_l2"]

'''
EMOTIONS = {"angry":  "😡",
            "disgust": "😒",
            "fear":  "😱",
            "happy":  "😂",
            "sad":  "😭",
            "surprise":  "😲",
            "neutral":  "😐"}

GENDERS = {"Woman": "👩‍🦰", "Man": "👨‍🦳"}
'''

class DeepFaceModel(c.Module):

    whitelist = ['verify', 'analyze', 'find',
                 'represent', 'stream', 'extract_faces', 'gradio',
                 'save_face', 'verify_face']

    def __init__(self, api_key: str = None, cache_key: bool = True):
        config = self.set_config(kwargs=locals())

    def set_api_key(self, api_key: str, cache: bool = True):
        if api_key == None:
            api_key = self.get_api_key()

        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)

        assert isinstance(api_key, str)

    def verify(self,
               img1_path,
               img2_path,
               model_name="VGG-Face",
               detector_backend="mtcnn",
               distance_metric="cosine",
               enforce_detection=True,
               align=True,
               normalization="base",
               ):
        """
        This function verifies an image pair is same person or different persons. In the background,
        verification function represents facial images as vectors and then calculates the similarity
        between those vectors. Vectors of same person images should have more similarity (or less
        distance) than vectors of different persons.

        Parameters:
                img1_path, img2_path: exact image path as string. numpy array (BGR) or based64 encoded
                images are also welcome. If one of pair has more than one face, then we will compare the
                face pair with max similarity.

                model_name (str): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib
                , ArcFace and SFace

                distance_metric (string): cosine, euclidean, euclidean_l2

                enforce_detection (boolean): If no face could not be detected in an image, then this
                function will return exception by default. Set this to False not to have this exception.
                This might be convenient for low resolution images.

                detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
                dlib, mediapipe or yolov8.

                align (boolean): alignment according to the eye positions.

                normalization (string): normalize the input image before feeding to model

        Returns:
                Verify function returns a dictionary.

                {
                        "verified": True
                        , "distance": 0.2563
                        , "max_threshold_to_verify": 0.40
                        , "model": "VGG-Face"
                        , "similarity_metric": "cosine"
                        , 'facial_areas': {
                                'img1': {'x': 345, 'y': 211, 'w': 769, 'h': 769},
                                'img2': {'x': 318, 'y': 534, 'w': 779, 'h': 779}
                        }
                        , "time": 2
                }

        """
        return DeepFace.verify(
            img1_path,
            img2_path,
            model_name,
            detector_backend,
            distance_metric,
            enforce_detection,
            align,
            normalization)

    def analyze(self,
                img_path,
                detector_backend="mtcnn",
                output=None, # custom parameter: None: only json, "numpy": for gradio output, "base64": to open on a browser
                actions=("emotion", "age", "gender", "race"),
                enforce_detection=True,
                align=True,
                silent=False,
                ):
        """
        This function analyzes facial attributes including age, gender, emotion and race.
        In the background, analysis function builds convolutional neural network models to
        classify age, gender, emotion and race of the input image.

        Parameters:
                img_path: exact image path, numpy array (BGR) or base64 encoded image could be passed.
                If source image has more than one face, then result will be size of number of faces
                appearing in the image.

                actions (tuple): The default is ('age', 'gender', 'emotion', 'race'). You can drop
                some of those attributes.

                enforce_detection (bool): The function throws exception if no face detected by default.
                Set this to False if you don't want to get exception. This might be convenient for low
                resolution images.

                detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
                dlib, mediapipe or yolov8.

                align (boolean): alignment according to the eye positions.

                silent (boolean): disable (some) log messages

        Returns:
                The function returns a list of dictionaries for each face appearing in the image.

                [
                        {
                                "region": {'x': 230, 'y': 120, 'w': 36, 'h': 45},
                                "age": 28.66,
                                "dominant_gender": "Woman",
                                "gender": {
                                        'Woman': 99.99407529830933,
                                        'Man': 0.005928758764639497,
                                }
                                "dominant_emotion": "neutral",
                                "emotion": {
                                        'sad': 37.65260875225067,
                                        'angry': 0.15512987738475204,
                                        'surprise': 0.0022171278033056296,
                                        'fear': 1.2489334680140018,
                                        'happy': 4.609785228967667,
                                        'disgust': 9.698561953541684e-07,
                                        'neutral': 56.33133053779602
                                }
                                "dominant_race": "white",
                                "race": {
                                        'indian': 0.5480832420289516,
                                        'asian': 0.7830780930817127,
                                        'latino hispanic': 2.0677512511610985,
                                        'black': 0.06337375962175429,
                                        'middle eastern': 3.088453598320484,
                                        'white': 93.44925880432129
                                }
                        }
                ]
        """
        results = DeepFace.analyze(
            img_path,
            actions,
            enforce_detection,
            detector_backend,
            align,
            silent,
        )

        image = cv2.imread(img_path)
        for result in results:
            x, y, w, h = result["region"].values()
            cv2.rectangle(
                image, (x, y), (x + w, y + h), (255, 255, 255), 1
            )  # draw face rectangle to image

            cv2.putText(
                image,
                f"{result['dominant_emotion']}, {result['dominant_gender']} {int(result['age'])}, {result['dominant_race']}",
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            ) # put analyzed result on image

        # Convert the OpenCV image to RGB (PIL format)
        if output == None:
            return results       
        elif output == 'numpy': 
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return results, image_rgb
        elif output == "base64": 
            pil_image = Image.fromarray(image_rgb)

            # Convert the PIL image to base64
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return results, f'data:image/jpeg;base64,{base64_image}'
        else:
            return results

    def find(self,
             img_path,
             db_path,
             model_name="VGG-Face",
             distance_metric="cosine",
             enforce_detection=True,
             detector_backend="mtcnn",
             align=True,
             normalization="base",
             silent=False,
             ):
        """
        This function applies verification several times and find the identities in a database

        Parameters:
                img_path: exact image path, numpy array (BGR) or based64 encoded image.
                Source image can have many faces. Then, result will be the size of number of
                faces in the source image.

                db_path (string): You should store some image files in a folder and pass the
                exact folder path to this. A database image can also have many faces.
                Then, all detected faces in db side will be considered in the decision.

                model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID,
                Dlib, ArcFace, SFace or Ensemble

                distance_metric (string): cosine, euclidean, euclidean_l2

                enforce_detection (bool): The function throws exception if a face could not be detected.
                Set this to False if you don't want to get exception. This might be convenient for low
                resolution images.

                detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
                dlib, mediapipe or yolov8.

                align (boolean): alignment according to the eye positions.

                normalization (string): normalize the input image before feeding to model

                silent (boolean): disable some logging and progress bars

        Returns:
                This function returns list of pandas data frame. Each item of the list corresponding to
                an identity in the img_path.
        """
        return DeepFace.find(
            img_path,
            db_path,
            model_name,
            distance_metric,
            enforce_detection,
            detector_backend,
            align,
            normalization,
            silent
        )

    def represent(self,
                  img_path,
                  model_name="VGG-Face",
                  enforce_detection=True,
                  detector_backend="mtcnn",
                  align=True,
                  normalization="base",
                  ):
        """
        This function represents facial images as vectors. The function uses convolutional neural
        networks models to generate vector embeddings.

        Parameters:
                img_path (string): exact image path. Alternatively, numpy array (BGR) or based64
                encoded images could be passed. Source image can have many faces. Then, result will
                be the size of number of faces appearing in the source image.

                model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace

                enforce_detection (boolean): If no face could not be detected in an image, then this
                function will return exception by default. Set this to False not to have this exception.
                This might be convenient for low resolution images.

                detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
                dlib, mediapipe or yolov8.

                align (boolean): alignment according to the eye positions.

                normalization (string): normalize the input image before feeding to model

        Returns:
                Represent function returns a list of object with multidimensional vector (embedding).
                The number of dimensions is changing based on the reference model.
                E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
        """
        return DeepFace.represent(
            img_path,
            model_name,
            enforce_detection,
            detector_backend,
            align,
            normalization,
        )

    def stream(slef,
               db_path="",
               model_name="VGG-Face",
               detector_backend="opencv",
               distance_metric="cosine",
               enable_face_analysis=True,
               source=0,
               time_threshold=5,
               frame_threshold=5,
               ):
        """
        This function applies real time face recognition and facial attribute analysis

        Parameters:
                db_path (string): facial database path. You should store some .jpg files in this folder.

                model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace

                detector_backend (string): opencv, retinaface, mtcnn, ssd, dlib, mediapipe or yolov8.

                distance_metric (string): cosine, euclidean, euclidean_l2

                enable_facial_analysis (boolean): Set this to False to just run face recognition

                source: Set this to 0 for access web cam. Otherwise, pass exact video path.

                time_threshold (int): how many second analyzed image will be displayed

                frame_threshold (int): how many frames required to focus on face

        """

        return DeepFace.stream(
            db_path,
            model_name,
            detector_backend,
            distance_metric,
            enable_face_analysis,
            source,
            time_threshold,
            frame_threshold,
        )

    def extract_faces(self,
                      img_path,
                      target_size=(224, 224),
                      detector_backend="mtcnn",
                      enforce_detection=True,
                      align=True,
                      grayscale=False,
                      ):
        """
        This function applies pre-processing stages of a face recognition pipeline
        including detection and alignment

        Parameters:
                img_path: exact image path, numpy array (BGR) or base64 encoded image.
                Source image can have many face. Then, result will be the size of number
                of faces appearing in that source image.

                target_size (tuple): final shape of facial image. black pixels will be
                added to resize the image.

                detector_backend (string): face detection backends are retinaface, mtcnn,
                opencv, ssd or dlib

                enforce_detection (boolean): function throws exception if face cannot be
                detected in the fed image. Set this to False if you do not want to get
                an exception and run the function anyway.

                align (boolean): alignment according to the eye positions.

                grayscale (boolean): extracting faces in rgb or gray scale

        Returns:
                list of dictionaries. Each dictionary will have facial image itself,
                extracted area from the original image and confidence score.

        """        
        return DeepFace.extract_faces(
            img_path,
            target_size,
            detector_backend,
            enforce_detection,
            align,
            grayscale,
        )

    def save_face(self, 
                  img_src_dir='images', 
                  model_filename='one_person_face_model.h5', 
                  train_epochs=5, 
                  train_batch_size=32,
                  base_model_name='Facenet', 
                  train_optimizer='adam', 
                  train_loss='binary_crossentropy', 
                  train_metrics='accuracy',
                  ):
        """
        This function fine-tunes the model using individual face images. You can upload one or multiple face images of the same person for fine-tuning. This process amplifies the data, especially when you have a small amount of it.

        Parameters:
                img_src_dir (string): image src path.  Source dir can have many images.

                model_filename (string): model filename that will be stored.

                base_model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace

                train_optimizer (string): sgd, rmsprop, adam, adadelta, adagrad, adamax, nadam, ftrl

                train_loss (string): should be binary_corssentropy for binary classification

                train_metrics (string): would be splited by comma. e.x. you can put 'accuracy,binary_accuracy'
                
                train_epochs (number): number of training epochs

                train_batch_size (number): size of batch per training

        Returns:
                model filename that have been fine-tuned on given face images.
        """
        from deepface import DeepFace

        # load pretrained model from Deepface
        base_model = DeepFace.build_model(base_model_name)

        from tensorflow.keras.models import Model

        # Remove the top layer and use the output as feature extractor
        feature_extractor = Model(inputs=base_model.inputs, outputs=base_model.layers[-1].output)

        from tensorflow.keras.layers import Dense, Dropout, Flatten
        from tensorflow.keras.models import Sequential

        # Add custom layers
        model = Sequential([
            feature_extractor,
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Assuming binary classification (person/not person)
        ])

        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        def get_preprocessed_images(folder_path, 
                                    image_size=(160, 160), 
                                    label=1, 
                                    augment=True, 
                                    batch_size=32):
            images = []
            labels = []

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255.0
                    images.append(img)
                    labels.append(label)

            X = np.array(images)
            y = np.array(labels)


            if augment:
                # Define data augmentation
                datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest'
                )
                
                # Apply data augmentation (Note: This augments data in real-time, not in memory)
                return datagen.flow(X, y, batch_size=batch_size)
            else:
                return X, y

        # Usage
        folder_path = img_src_dir
        datagenerator = get_preprocessed_images(folder_path)

        # Freeze the layers of feature_extractor
        for layer in feature_extractor.layers:
            layer.trainable = False

        # Compile the model
        model.compile(optimizer=train_optimizer, loss='binary_crossentropy', metrics=train_metrics.split(','))

        # Train the model
        model.fit(datagenerator, epochs=train_epochstrain_batch_size, batch_size=train_batch_size)

        model.save(model_filename)
        return "Successfully saved!"

    def verify_face(self, 
                    face_image_path='image.jpeg', 
                    model_filename='one_person_face_model.h5', 
                    threshold=0.85):
        
        """
        This function verifies an individual's face using the fine-tuned model. You can specify the image path and select the model file saved in the 'save_face' function.

        Parameters:
                face_image_path (string): face image path.  

                model_filename (string): model filename that which is stored.

                threshold(number): threshold

        Returns:
                result of verification
        """
        from tensorflow.keras.models import load_model

        # load saved model
        model = load_model(model_filename)
        # get shape of input image
        input_shape = model.layers[0].input_shape if hasattr(model.layers[0], 'input_shape') else None

        # resize the image as input_shape
        def preprocess_image(img_path, image_size=(input_shape[1], input_shape[2])):

            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                return img

        # feed the image to the model to get reseult
        def verify_face(image_path, model):
            # Preprocess the image
            img= preprocess_image(image_path)  # This should match the preprocessing used during training

            # Predict
            prediction = model.predict(np.array([img]))
            print(prediction)
            return prediction[0][0]  # Threshold can be adjusted

        # Test with a new image
        result = verify_face(face_image_path, model)
        print("Is this the target person?", result)
        return result, result >= threshold

    def gradio(self):
        with gr.Blocks() as demo:
            with gr.Tab("Verify"):
                with gr.Row():
                    with gr.Column():
                        img1_path = gr.Image(label="Person 1", type="filepath")
                    with gr.Column():
                        img2_path = gr.Image(label="Person 2", type="filepath")
                with gr.Row():
                    with gr.Column(scale=1):
                        model_name = gr.Dropdown(
                            MODELS, label="Model", info="Face Recognition Model", value=MODELS[0])
                        detector_backend = gr.Dropdown(
                            DETECTORS, label="Detector", info="Face Detector", value=DETECTORS[0])
                        distance_metric = gr.Dropdown(
                            METRICS, label="distance_metric", value=METRICS[0])
                        verify_button = gr.Button('Verify')
                    with gr.Column(scale=4):
                        json = gr.JSON()
                verify_button.click(fn=self.verify, inputs=[img1_path, img2_path,
                                                            model_name, detector_backend, distance_metric
                                                            ], outputs=json)
            with gr.Tab("Analyze"):
                with gr.Row():
                    with gr.Column():
                        img_path = gr.Image(label="Person", type="filepath")
                    with gr.Column():
                        img_result = gr.Image(label="Analyzed")
                with gr.Row():
                    with gr.Column(scale=1):
                        detector_backend = gr.Dropdown(
                            DETECTORS, label="Detector", info="Face Detector", value=DETECTORS[0])
                        output_type = gr.Textbox("numpy", visible=False)
                        analyze_button = gr.Button('Analyze')
                    with gr.Column(scale=4):
                        json = gr.JSON()
                analyze_button.click(fn=self.analyze, inputs=[img_path,
                                                              detector_backend, output_type
                                                              ], outputs=[json, img_result])
            with gr.Tab("one_person_face_model"):                
                with gr.Row():
                    with gr.Column():
                        img_src_dir = gr.File(label="Person Face Dir", file_count='directory')
                        model_filename = gr.Textbox(label="Model Filename", value='one_person_face_model.h5')
                        train_epochs = gr.Slider(minimum=1, maximum=1000, value=5, label="Train Epochs", interactive=True)
                        train_batch_size = gr.Slider(minimum=1, maximum=1000, value=32, label="Train Batch Size", interactive=True)
                        save_button = gr.Button('Save') 
                        json = gr.Json()
                        save_button.click(fn=self.save_face, inputs=[img_src_dir, 
                                                                    model_filename, train_epochs, train_batch_size], outputs=json)
                    with gr.Column():
                        face_image_path = gr.Image(label="Person", type="filepath")
                        model_filename = gr.Textbox(label="Model Filename", value='one_person_face_model.h5')
                        threshold = gr.Slider(minimum=0.1, maximum=1.0, value=0.85, label="Threshold", interactive=True)
                        verify_button = gr.Button('Verify') 
                        json = gr.Json()
                        verify_button.click(fn=self.verify_face, inputs=[face_image_path
                                                                        model_filename, model_name, detector_backend, distance_metric ], outputs=json)
        demo.launch(quiet=True, share=True)
