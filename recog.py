import cv2
import face_recognition
import numpy as np
import os 
from datetime import datetime
from numpy import genfromtxt

encoded_list=[]
home = os.getcwd()
datafiles = []
people =[]

def getdata():# To read and data form csv files
    os.chdir('Data')
    if len(os.listdir())==0:
        print("No face data exist in the Data Directory")
        exit()
    for file in  os.listdir():
        if file =='Names.csv':
            continue
        if os.path.isfile(file):
            datafiles.append(file)
            file = file.split(".")[-2]
            people.append(file)
    
    for file in datafiles:
        my_data = genfromtxt(file, delimiter=',')
        encoded_list.append(my_data)
    
    os.chdir(home)

getdata()

fileName =  "Attendance.csv"
try:
    if not os.path.exists(fileName):
      with open(fileName,'w') as thekey:
        thekey.write(f'Name,Time,Date\n')
        print("File " , fileName ,  " Created ") 
except FileExistsError:
    print("File" , fileName ,  " already exists")

def attendance(name): # used for attendance
    with open('Attendance.csv', 'r+') as f:
        myCsvList = f.readlines()
        peoplename = []
        for line in myCsvList:
            entry = line.split(',')
            peoplename.append(entry[0])
        if name not in peoplename: # to prevent the overwirting the attedance again and again
            time_now = datetime.now()
            current_time = time_now.strftime('%H:%M:%S')
            date_today = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{current_time},{date_today}')

cam = cv2.VideoCapture(0)

while True : # To match th faces on cam with the avaible recongnition data 
    ret , frame =cam.read()
    faces = cv2.resize(frame, (0,0), None, 0.25,0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    faces_on_cam=face_recognition.face_locations(faces)
    encode_on_cam = face_recognition.face_encodings(faces, faces_on_cam)
    
    
    for encodeFace, faceLoc in zip(encode_on_cam, faces_on_cam):
        matches = face_recognition.compare_faces(encoded_list, encodeFace)
        faceDis = face_recognition.face_distance(encoded_list, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] :
            name = people[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.56, (255, 255, 255), 2)
            attendance(name)
    cv2.imshow('Watcher' , frame)
    if cv2.waitKey(1) == 13:
        break
cam.release()
cv2.destroyAllWindows()
