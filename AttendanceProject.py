import cv2
import numpy as np
import face_recognition 
import os
from datetime import datetime


# Path to the directory containing images of student
path = 'C:\\Users\\ASANTE PHILEMON\\Desktop\\Academics\\Level 400\\L 400 2nd, Sem\\my_work\\ImageBasic'

images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#functions that will compute the encoding 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#Mark Attendance
def markAttendance(name):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    with open(current_date+'.csv','w+',newline = '') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)

#finding the matches 
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#finding the encoding of the webcam 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

#finding matches in the frame 
    for encodeFace, faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)


        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)
            #print(name)
            y1,x2,y2,x1 = faceLoc
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
     
    
    



     



    cv2.imshow('webcam',img)
    cv2.waitKey(1)
    



