import PySimpleGUI as sg
import cv2, os
import numpy as np
import pandas as pd
from datetime import datetime
from mtcnn import MTCNN as MTCNN
import tensorflow as tf



# Model Settings
THRESHOLD = 0.55

# Paths
DATABASE_PATH = 'C:/Users/Ng Wei Xiang/Desktop/ITI110 Deep Learning Project/app/database'
LOG_FILE_PATH = 'C:/Users/Ng Wei Xiang/Desktop/ITI110 Deep Learning Project/app/database/logs.csv'
MODEL_WEIGHTS_PATH = 'C:/Users/Ng Wei Xiang/Desktop/ITI110 Deep Learning Project/app/encoder_savedweights/weights'

# Camera Settings
camera_Width  = 640
camera_Heigth = 480
frameSize = (camera_Width, camera_Heigth)



def draw_box(img, bounding_box, text, colour):
    x,y,w,h = bounding_box
    x1, y1, x2, y2 = x, y, x+w, y+h
    img = cv2.rectangle(img,(x1,y1),(x2,y2),colour,2)
    if text != "":
        img = cv2.rectangle(img,(x1,y1),(x2,y1 + 12),colour,-1)
        img = cv2.putText(img, text, (x1, y1 + 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 1, cv2.LINE_AA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def MTCNN_step(img, face_detector_mtcnn):
    detected_faces = face_detector_mtcnn.detect_faces(img)
    bounding_box = []
    for detected_face in detected_faces:
        bounding_box = detected_face["box"]
        break
    if bounding_box:
        x,y,w,h = bounding_box
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face, bounding_box
    return None, None

def build_EfficientFaceNet(pretrained_path=''):
    Inp = tf.keras.layers.Input((224, 224, 3), name='input')
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=Inp, drop_connect_rate=0.5)
    x = base_model.output
    x = tf.keras.layers.Dropout(0.5, name='dropout1')(x)
    x = tf.keras.layers.DepthwiseConv2D((7,7), name='glb_depth_conv')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout2')(x)
    x = tf.keras.layers.Dense(128, name='non_norm_emb')(x)
    Out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='norm_emb')(x)
    EfficientFaceNet = tf.keras.models.Model(inputs=Inp, outputs=Out, name='EfficientFaceNet')
    if pretrained_path:
        EfficientFaceNet.load_weights(pretrained_path)
    return EfficientFaceNet

def EfficientFaceNet_step(cropped_face, EfficientFaceNet):
    img = cv2.resize(cropped_face, (224,224))
    img = np.asarray(img).astype("float32")
    webcam_embedding = EfficientFaceNet(img[np.newaxis,...])[0]
    return webcam_embedding

def verification_step(username, webcam_embedding):
    registered_embedding = np.load(os.path.join(DATABASE_PATH, f'{username}.npy'))
    similarity = np.dot(registered_embedding, webcam_embedding) / (np.linalg.norm(webcam_embedding) * np.linalg.norm(registered_embedding))
    return similarity > THRESHOLD, similarity



def main():

    sg.theme('DarkBlue')
    
    # define the window layout
    colwebcam_layout = [
        [sg.Text("Camera View", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="webcam")],
        [
            sg.Text('Username:', size=(10, 1)),
            sg.Input(key='-USERNAME INPUT-', size=(40, 1)),
            sg.Button('Register', size=(10, 1)),
            sg.Button('Verify', size=(10, 1))
        ]
    ]
    colwebcam = sg.Column(colwebcam_layout, element_justification='center')

    coloutput_layout = [
        [sg.Text("Output", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="outcam")],
        [sg.Text('', key='-OUTPUT TEXT-')]
    ]
    coloutput = sg.Column(coloutput_layout, element_justification='center')
    
    layout = [[colwebcam, coloutput]]
    
    # create the window and show it without the plot
    window = sg.Window("ITI110 Demo App", layout, location=(100, 100))

    
    # initiate models and variables
    face_detector_mtcnn = MTCNN()
    EfficientFaceNet = build_EfficientFaceNet(MODEL_WEIGHTS_PATH)
    outcam_current_display = False
    black_frame = np.zeros(frameSize[::-1])
    black_imgbytes = cv2.imencode(".png", black_frame)[1].tobytes()
    if not os.path.isfile(LOG_FILE_PATH):
        logs_df = pd.DataFrame({
            'Time': [],
            'Event': [],
            'Outcome': [],
        })
        logs_df.to_csv(LOG_FILE_PATH, index=False)
    
    
    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)

    while True:
        event, values = window.read(timeout=20)
        if event == sg.WIN_CLOSED:
            cap.release()
            cv2.destroyAllWindows()
            return

        # get camera frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, frameSize)
        
        # update webcam
        webcam_imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["webcam"].update(data=webcam_imgbytes)
        if not outcam_current_display:
            window["outcam"].update(data=black_imgbytes)
        
        
        # meaningful events
        if event in {'Register', 'Verify'}:
            
            # check for valid username
            valid_username = True
            username = values['-USERNAME INPUT-']
            if username == '':
                outcome = "Username is not given"
                outcam_imgbytes = black_imgbytes
                valid_username = False
            elif not os.path.exists(os.path.join(DATABASE_PATH, f'{username}.npy')):
                outcome = f"User {username} is not registered"
                outcam_imgbytes = black_imgbytes
                valid_username = False
            
            if valid_username:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cropped_face, bounding_box = MTCNN_step(img, face_detector_mtcnn)
                
                # check for detected face
                if cropped_face is None:
                    outcome = "No face detected"
                    outcam_imgbytes = black_imgbytes

                # register event
                elif event == 'Register':
                    webcam_embedding = EfficientFaceNet_step(cropped_face, EfficientFaceNet)
                    np.save(os.path.join(DATABASE_PATH, f'{username}.npy'), webcam_embedding)
                    
                    outcome = f"User {username} has been registered"
                    img_boxed = draw_box(img, bounding_box, username, (255,255,0))
                    outcam_imgbytes = cv2.imencode(".png", img_boxed)[1].tobytes()

                # verify event
                elif event == 'Verify':
                    webcam_embedding = EfficientFaceNet_step(cropped_face, EfficientFaceNet)
                    verified, sim_score = verification_step(username, webcam_embedding)
                    
                    outcome = f"User {username} is {'' if verified else 'not '}verified (score: {sim_score:.2f})"
                    img_boxed = draw_box(img, bounding_box, username, (0,255,0) if verified else (255,0,0))
                    outcam_imgbytes = cv2.imencode(".png", img_boxed)[1].tobytes()

            # log event
            logs_df = pd.read_csv(LOG_FILE_PATH, index_col=False)
            new_logs_df = pd.DataFrame({
                'Time': [datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
                'Event': [event],
                'Outcome': [outcome],
            })
            logs_df = pd.concat([logs_df, new_logs_df])
            logs_df.to_csv(LOG_FILE_PATH, index=False)
    
    
            # update output pane
            window['-OUTPUT TEXT-'].update(outcome)
            window["outcam"].update(data=outcam_imgbytes)
            outcam_current_display = True
        
    
    
if __name__ == "__main__":
    main()