import cv2

from  ultralytics import YOLO
import  supervision as sv
# def parse_arguments()->argparse.Namespace:
#     details=argparse.ArgumentParser(title="YOLOV8")
#     details.add_argument("--webcam-resoluation"
#                          ,default=[1280,720],
#                          nargs=2,
#                          type=int)


from pyrebase import pyrebase

config = {
    "apiKey": "AIzaSyCbS2zxws8NTgd8KD7wmRfvzKSq53ZAycM",

    "authDomain": "smart-green-house-ef358.firebaseapp.com",

    "databaseURL": "https://smart-green-house-ef358-default-rtdb.firebaseio.com",

    "projectId": "smart-green-house-ef358",

    "storageBucket": "smart-green-house-ef358.appspot.com",

    "messagingSenderId": "951967419183",

    "appId": "1:951967419183:web:d46e50d9fbecf7d1ae2fda",

    "measurementId": "G-SB14B0FV36",
}

cap=cv2.VideoCapture(0)
# cap=cv2.VideoCapture("rtsp://admin:NOMADI@192.168.1.6:554/H.264")
while (cap.isOpened()):
    ret,frame=cap.read()
    model=YOLO("C:/Users/20121/Downloads/best.pt")
    # model=YOLO("C:/Users/20121/Downloads/best.pt")
    box_annotator=sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    if ret==True:
       res=model(frame,agnostic_nms=True)[0]
       detections=sv.Detections.from_yolov8(res)
       labels = [
           f"{model.model.names[class_id]}"

for _,_,confidence,class_id,_ in detections
       ]
       firebase = pyrebase.initialize_app(config)
       database = firebase.database()
       data = {"one": labels}
       database.child("Detection Results").update(data)
       # while i>=0:
       #
       #
       #     i+=1
       frame=box_annotator.annotate(scene=frame,detections=detections,labels=labels)

       cv2.imshow('Diseased',frame)
       if cv2.waitKey(25) & 0xFF==ord('q'):
           break


