from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np  # Normalization
import streamlit as st
st.set_page_config(page_title="Mask Detection System",page_icon="data:image/png;base64,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")
st.title("FACE MASK DETECTION SYSTEM")
choice=st.sidebar.selectbox("My Menu",("HOME","IP CAMERA","CAMERA"))
if(choice=="HOME"):
    st.image("https://act-my.com/wp-content/uploads/2021/12/faceRecognition.gif")
elif(choice=="IP CAMERA"):
    url=st.text_input("ENTER IP CAMERA URL")
    btn=st.button("Start")
    window=st.empty()
    if btn:
        vid = cv2.VideoCapture(url) # (0) for primary camera of laptop
        btn2=st.button("Stop")
        if btn2:
            vid.release()
            st.rerun()
        facemodel = cv2.CascadeClassifier("face.xml")
        maskmodel = load_model("mask.h5", compile=False)
        i=0
        while True:
            flag, frame = vid.read()  # Read individual frames
            if flag:
                faces = facemodel.detectMultiScale(frame)
                for (x, y, l, w) in faces:
                    face_img = frame[y:y+w, x:x+l]  # Crop the face from the frame
                    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)  # Resize the cropped face image
                    face_img = np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)  # Reshape the image for the model
                    face_img = (face_img / 127.5) - 1  # Normalize the image

            # Predict if the person is wearing a mask or not
                    pred = maskmodel.predict(face_img)[0][0]
                    if pred < 0.9:  # If prediction indicates no mask (adjust threshold if needed)
                            path = "Data/" + str(i) + ".jpg"
                            cv2.imwrite(path, frame[y:y+w, x:x+l])  # Capture and save the face image
                            i = i + 1
                            cv2.rectangle(frame, (x, y), (x + l, y + w), (0, 0, 255), 3)  # Red rectangle for no mask
                    else:
                        cv2.rectangle(frame, (x, y), (x + l, y + w), (0, 255, 0), 3)  #
                window.image(frame,channels=("BGR")) #colour of opencv
elif(choice=="CAMERA"):
    cam=st.selectbox("ENTER 0 for primary camera or 1 for secondary camera",("NONE",0,1))
    btn=st.button("Start")
    window=st.empty()
    if btn:
        vid = cv2.VideoCapture(cam) # (0) for primary camera of laptop
        btn2=st.button("Stop")
        if btn2:
            vid.release()
            st.rerun()
        facemodel = cv2.CascadeClassifier("face.xml")
        maskmodel = load_model("mask.h5", compile=False)
        i=0
        while True:
            flag, frame = vid.read()  # Read individual frames
            if flag:
                faces = facemodel.detectMultiScale(frame)
                for (x, y, l, w) in faces:
                    face_img = frame[y:y+w, x:x+l]  # Crop the face from the frame
                    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)  # Resize the cropped face image
                    face_img = np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)  # Reshape the image for the model
                    face_img = (face_img / 127.5) - 1  # Normalize the image

            # Predict if the person is wearing a mask or not
                    pred = maskmodel.predict(face_img)[0][0]
                    if pred < 0.9:  # If prediction indicates no mask (adjust threshold if needed)
                            path = "Data/" + str(i) + ".jpg"
                            cv2.imwrite(path, frame[y:y+w, x:x+l])  # Capture and save the face image
                            i = i + 1
                            cv2.rectangle(frame, (x, y), (x + l, y + w), (0, 0, 255), 3)  # Red rectangle for no mask
                    else:
                        cv2.rectangle(frame, (x, y), (x + l, y + w), (0, 255, 0), 3)  #
                window.image(frame,channels=("BGR")) #colour of opencv 

