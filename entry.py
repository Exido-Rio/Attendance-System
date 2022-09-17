import cv2
import numpy as np
import face_recognition
import os 
from time import process_time
from datetime import datetime


t1 = process_time()
loc ='images' # images file location
images = [] # list of images
peopele =[] # name of people 
mylist  = os.listdir(loc)
home = os.getcwd()
existing_ppl = []

# To get the list of csv files in Data dir except Names.csv one 
os.chdir('Data')
fileName = "Names.csv"
try:
    if not os.path.exists(fileName):
      with open('Names.csv','w') as thekey:
        thekey.write(f'Name,Time,Date\n')
        print("File " , fileName ,  " Created ") 
except FileExistsError:
    print("File" , fileName ,  " already exists")
    
for file in os.listdir():
    if file =='Names.csv':
        continue
    if os.path.isfile(file):
        file = file.split(".")[-2]
        existing_ppl.append(file)
os.chdir(home)



for ts_img in mylist : # used to detect images files inside the images directory
    curr_img = cv2.imread(f'{loc}/{ts_img}')
    images.append(curr_img)
    peopele.append(os.path.splitext(ts_img)[0])

print("list people who's data is stored in data.csv ",existing_ppl)

def facedata(images):
    new_enrty=[]
    os.chdir("Data")
    with open('Names.csv', 'r+') as f:
        myCsvList = f.readlines()
        peoplename = []
        for line in myCsvList:
            entry = line.split(',')
            peoplename.append(entry[0])
        for img,ppl in zip(images,peopele) :
            if ppl not in existing_ppl and ppl not in peoplename :
                new_enrty.append(ppl)
                time_now = datetime.now()
                current_time = time_now.strftime('%H:%M:%S')
                date_today = time_now.strftime('%d/%m/%Y')
                f.writelines(f'{ppl},{current_time},{date_today}\n')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encoded = face_recognition.face_encodings(img)[0]
                np.savetxt(f"{ppl}.csv", encoded, delimiter = ",") 
        print("List of new people added",new_enrty)     
    os.chdir(home)

facedata(images)

t2 = process_time()
print("Encoding complted time took :",t2-t1)
