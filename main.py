import cv2
import face_recognition
import numpy as np
import os 
from datetime import datetime
from time import process_time


loc ='images' # images location
images = [] # to store images
peopele =[] # to store names
mylist  = os.listdir(loc)
print(mylist)
for ts_img in mylist :
    curr_img = cv2.imread(f'{loc}/{ts_img}')
    images.append(curr_img)
    peopele.append(os.path.splitext(ts_img)[0])


fileName =  "Attendance.csv"
try:
    if not os.path.exists(fileName):
      with open(fileName,'w') as thekey:
        thekey.write(f'Name,Time,Date\n')
        print("File " , fileName ,  " Created ") 
except FileExistsError:
    print("File" , fileName ,  " already exists")

def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myCsvList = f.readlines()
        peoplename = []
        for line in myCsvList:
            entry = line.split(',')
            peoplename.append(entry[0])
        if name not in peoplename:
            time_now = datetime.now()
            current_time = time_now.strftime('%H:%M:%S')
            date_today = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{current_time},{date_today}')

def faceencode(images):
    encoded_obj =[] # use to store images details such as details , distance etc.
    for img in images :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded = face_recognition.face_encodings(img)[0]
        encoded_obj.append(encoded)
    return encoded_obj # encode all photo feature and reutern all of them as a array

#print(faceencode(images))
t1 = process_time()
encoded_list = faceencode(images)
t2 = process_time()
print("Encoding complted in",t2-t1)

cam = cv2.VideoCapture(0)

while True :
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
            name = peopele[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)
    cv2.imshow('Watcher' , frame)
    if cv2.waitKey(1) == 13:
        break
cam.release()
cv2.destroyAllWindows()
