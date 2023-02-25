from fastapi import FastAPI, Request
import uvicorn
import os
import base64
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd
import pickle
import sklearn
import shutil
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
parent_dir = "/home/saadi09/FastApi/dataset/"
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("work_price_prediction.pkl", "rb"))


@app.get("/")
async def root():
    return {"message": "Hello World"}


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


def face_training():
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(parent_dir)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


@app.post('/recognize_face')
async def recognize_face(request: Request):
    data = await request.json()
    dict_data = dict(data)

    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    # names = ['None', 'saad', 'nawaf', 'Zeeshan', 'Moin', 'Sameer']
    # font = cv2.FONT_HERSHEY_SIMPLEX

    decoded_data = base64.b64decode(dict_data['image_base64'])
    np_data = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    # im = cv2.imread(img)  # Read image
    # img = cv2.imread("2.jpg")
    # Convert into grayscale
    # im = cv2.imread("test/test1.jpeg")
    # cv2.imshow('image', im)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    print(faces)
    if type(faces) == tuple:
        data = {'status': 400, 'response': {"confidence": 'unknown'}, 'message': 'Fail'}
        return data
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            confidence = format(round(100 - confidence))
            data = {'status': 200, 'response': {"confidence": confidence, 'id': id}, 'message': 'Success'}
        else:
            confidence = "  {0}%".format(round(100 - confidence))
            data = {'status': 400, 'response': {"confidence": 'unknown'}, 'message': 'Fail'}

    return data
    #    id = "unknown"
    #   confidence = "  {0}%".format(round(100 - confidence))

    # cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
    # cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
    # imS = cv2.resize(img, (1000, 1000))
    # cv2.imshow('img', imS)
    #  cv2.waitKey()


@app.post('/detect_face')
async def detect_face(request: Request):
    # Getting request data into dictionary
    data = await request.json()
    dict_data = dict(data)
    count = 0
    # Iterating through all images from request
    for img in dict_data['baseImg']:

        # Decoding base64 Image to detect face
        decoded_data = base64.b64decode(img)
        np_data = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

        # Loading face detection model using cv2
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Converting image to grey scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detecting face from grey scale image using model haarcascade
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if type(faces) == tuple:
            data = {'status': 400, 'response': "Face not Found", 'message': 'Fail'}
            return data
        else:
            count += 1
            # Getting Cropped face image
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Saving image in dataset folder
                image_path = "dataset/User." + str(dict_data['user']) + '.' + str(count) + ".jpg"
                cv2.imwrite(image_path, gray[y:y + h, x:x + w])
    face_training()
    return {'status': 200, 'response': "Face Detected", 'message': 'Success'}


@app.post('/estimate_price')
async def cost_estimation(request: Request):
    # if request.method == "POST":
    # worker_type =  date_arr = request.form["work_type"]
    data = await request.json()
    dict_data = dict(data)
    print(dict_data)

    work_hour = int(pd.to_datetime(dict_data['work_time'], format="%H:%M").hour)
    work_min = int(pd.to_datetime(dict_data['work_time'], format="%H:%M").minute)

    worker_type = "Handyman"
    if dict_data["work_type"] == 'Plumber':
        work_type = 0
    elif dict_data["work_type"] == 'Electrician':
        work_type = 1
    elif dict_data["work_type"] == 'Painter':
        work_type = 0
    elif dict_data["work_type"] == 'Handyman':
        work_type = 1
    elif dict_data["work_type"] == 'Cleaner':
        work_type = 1
    else:
        work_type = 0

    if dict_data["work_city"] == 'Islamabad':
        work_city = 0
    elif dict_data["work_city"] == 'Lahore':
        work_city = 1
    else:
        work_city = 2

    if dict_data["worker_expertise"] == 'Low':
        work_expertise = 0
    elif dict_data["worker_expertise"] == 'medium':
        work_expertise = 1
    else:
        work_expertise = 2

   # prediction = model.predict([[0, 0, 1, 2, 0]])
    prediction = model.predict([[work_type, work_city, work_expertise, work_hour, work_min]])
    #print({'status': 200, 'data': prediction[0], 'message': 'Success'})
    return {'status': 200, 'data': prediction[0], 'message': 'Success'}


def extract_data_cnic1():
    image = cv2.imread('Cnic/cnic.jpg')

    # --- dilation on the green channel ---
    dilated_img = cv2.dilate(image[:, :, 1], np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)

    # --- finding absolute difference to preserve edges ---
    diff_img = 255 - cv2.absdiff(image[:, :, 1], bg_img)

    # --- normalizing between 0 to 255 ---
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow('ImageWindow', cv2.resize(norm_img, (0, 0), fx=0.5, fy=0.5))

    # --- Otsu threshold ---
    th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(th, lang='eng')
    print(text)
    cv2.imshow('ROI', cv2.resize(th, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey()


@app.post('/extract_data_cnic')
async def extract_data_cnic(request: Request):
    data = await request.json()
    dict_data = dict(data)
    print(dict_data)
    decoded_data = base64.b64decode(dict_data['baseImg'])
    np_data = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    #img = cv2.imread("Cnic/cnic.jpg")
   # image_path = "Cnic/cnic" + ".jpg"
    #cv2.imwrite(image_path, img)
   # cv2.imshow("Image", img)
    #cv2.waitKey(0)

    dilated_img = cv2.dilate(img[:, :, 1], np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)

    # --- finding absolute difference to preserve edges ---
    diff_img = 255 - cv2.absdiff(img[:, :, 1], bg_img)

    # --- normalizing between 0 to 255 ---
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # cv2.imshow('ImageWindow', cv2.resize(norm_img, (0, 0), fx=0.5, fy=0.5))

    # --- Otsu threshold ---
    th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(th, lang='eng')
    #print(text)
    # cv2.imshow('ROI', cv2.resize(th, (0, 0), fx=0.5, fy=0.5))
    # cv2.waitKey()
    s = text.split('\n')
    print(s)
    #print(len(s))
    for i in s:
        if len(i) > 10:
            if i[5] == '-' and i[13] == '-':
                print(i[0:15])
                data = {'status': 200, 'response': "Extracted", 'message': 'Pass', 'Data': i[0:15]}
                return data
            else:
                data = {'status': 400, 'response': "Not Extracted, try again!", 'message': 'Fail'}
        else:
            data = {'status': 400, 'response': "Not Extracted, try again!", 'message': 'Fail'}

    return data

if __name__ == '__main__':
    #extract_data_cnic1()
    uvicorn.run(app, host='0.0.0.0', port=5000)
