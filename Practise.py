from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np  # Normalization

# Initialize variables
i = 0

# Load video and models
vid = cv2.VideoCapture("facemask.mp4") # (0) for primary camera of laptop
facemodel = cv2.CascadeClassifier("face.xml")
maskmodel = load_model("mask.h5", compile=False)

# Process the video frames
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
                cv2.rectangle(frame, (x, y), (x + l, y + w), (0, 255, 0), 3)  # Green rectangle for mask

        # Display the frame
        cv2.namedWindow("riya", cv2.WINDOW_NORMAL)
        cv2.imshow("riya", frame)

        # Exit on 'x' key press
        k = cv2.waitKey(33)
        if k == ord("x"):
            break
    else:
        break

# Release resources
vid.release()
cv2.destroyAllWindows()


    
