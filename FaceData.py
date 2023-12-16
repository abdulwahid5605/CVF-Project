import cv2
import numpy as np
cap=cv2.VideoCapture(0)

# we are using haarcascade classifier algorithm to fetch data from an image
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") 

# to store data after every 10th second we use skip
skip=0

# list jismay data store hoga
face_data=[]

# this is the path of the folder jismay hum image ka data store krayengay
dataset_path="./face_dataset/"

file_name=input("Enter the name of the person")

while True:
    # ret is boolean variable that checks the camera is open or not
    ret,frame=cap.read()
    # converting the colorful camera image into grayscale image
    # why we are converting images into grayscale? Because it is easier for the algorithm to read data from grayscale image as compared to colorful image
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # all the coordinates(top left, top right, bottom left , bottom right) of the face is returned by the "face" variable

    if ret==False:
        continue

    # using functionality of harcascade algo
    # "detectMultiScale" detects the block in which we are having the coordinates of the faces and the coordinates are x,y,w,h
    # faces is a list storing coordinates
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5) #parameters frame_name , scaling factor , k (number of neighbours)  

    if len(faces) == 0:
        continue

    # the next step is soring of faces in decreasing orders(there can be multiple faces in an image)
    # faces=[x,y,w,h] => Area=w*h => Area[2] * Area[3]
    # image having larger area has obviously the greater size
    # lambda x is representing faces
    # x[2] * x[3] = Area[2] * Area[3]
    # reverse true means descending order

    k=1

    faces=sorted(faces,key=lambda x: x[2]*x[3],reverse=True)

    skip=skip+1

    for i in faces[:1]:
        # fetching image coordinates from an image
        x,y,w,h=faces

        # applying padding
        offset=5
        # we are subtracting the padding in order to get the face
        face_offset=frame[y-offset:y+h+offset, x-offset:x+w+offset]

        # converting the image into 100 by 100 size
        face_selection=cv2.resize(face_offset,(100,100))

        # after every tenth second the data is stored in an array

        if skip % 10 == 0:
            face_data.append(face_selection)
            print(len(face_data))
        
        cv2.imshow(str(k), face_selection)

        # creating a rectangle near the faces(green color)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("faces",frame)

    # if q is pressed then camera will simply switch off
    key_pressed=cv2.waitKey(1) 
    if key_pressed== ord("q"):
        break

    
# now we are having list of face stored in face_data 
# we will now convert list into array for an efficient work
# we have to apply "knn" algorithm and it applies only on numpy array

face_data=np.array(face_data)
# now we have to reshape this array
face_data=face_data.reshape(face_data.shape[0],1)
# we are printing the shape of data
print(face_data.shape)

# now saving the data in "face_dataset" folder
np.save(dataset_path+file_name+".npy")
print("Dataset saved at : {}".format(dataset_path + file_name + ".npy"))

cap.release()
cv2.destroyAllWindows()
   
