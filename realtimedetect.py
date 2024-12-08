import cv2
from keras.models import model_from_json
import numpy as np

# Load pre-trained model
json_file = open('D:/Python/Facial Emotion Detection/emotiondetector.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('D:/Python/Facial Emotion Detection/emotiondetector.h5')

# Load haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    print("Error loading haar cascade file")
    exit()

# define labels for emotions
labels = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}

# feature extraction function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    feature = np.repeat(feature, 3, axis=-1)
    return feature/255.0

# access webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Could not access the webcam")
    exit()

# real time emotion detection
while True:
    i, im = webcam.read()
    if im is None:
        print("Frame capture failed")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p,q,r,s) in faces:
        image = gray[q:q+s, p:p+r]
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        cv2.putText(im, '%s' % prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
    
    cv2.imshow('image', im)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()