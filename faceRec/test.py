
##import cv2
##
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^WEBCAM OPENCV^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##cv2.namedWindow("preview")
##vc = cv2.VideoCapture(0)
##
##if vc.isOpened(): # try to get the first frame
##    rval, frame = vc.read()
##else:
##    rval = False
##
##while rval:
##    cv2.imshow("preview", frame)
##    rval, frame = vc.read()
##    key = cv2.waitKey(20)
##    if key == 27: # exit on ESC
##        break
##    
##cv2.destroyWindow("preview")


##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^FACIAL RECOGNITION^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##import numpy as np
##import cv2
##
##face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
##eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
##
##img = cv2.imread('sachin.jpg')
##gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##faces = face_cascade.detectMultiScale(img, 1.3, 5)
##for (x,y,w,h) in faces:
##    img = cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
##    roi_gray = gray[y:y+h, x:x+w]
##    eyes = eye_cascade.detectMultiScale(roi_gray)
##    for (ex,ey,ew,eh) in eyes:
##        cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
##
##cv2.imshow('gray',gray)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^webcam WORK^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, 0)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

