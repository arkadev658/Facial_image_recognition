from tensorflow.keras.models import load_model
from keras.preprocessing import image
import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

haar_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model=load_model('F:\\Downloads\\emotion_detection.h5')
age_model=load_model('F:\\Downloads\\age_prediction_new.h5')
gender_model=load_model('F:\\Downloads\\gender_v2.h5')

emotion_labels={0:'angry',
                1:'disgusted',
                2:'fearful',
                3:'happy',
                4:'neutral',
                5:'sad',
                6:'surprised'}

gender_labels={0:'Male',1:'Female'}

capture =  cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi=frame[y:y+h,x:x+h]
        roi=roi.astype('float')/255.0
        
        roi_em=cv.resize(roi,(48,48),interpolation=cv.INTER_AREA)
        #roi_em=img_to_array(roi_em)
        roi_em=np.expand_dims(roi_em,axis=0)
        em_pred=emotion_model.predict(roi_em).argmax(axis=1)
        em_label=emotion_labels[int(em_pred)]
        cv.putText(frame,em_label,(x,y),cv.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),1)
        
        roi_gen=cv.resize(roi,(200,200),interpolation=cv.INTER_AREA)
        #roi_gen=img_to_array(roi_gen)
        roi_gen=np.expand_dims(roi_gen,axis=0)
        gen_pred=gender_model.predict(roi_gen).argmax(axis=1)
        gen_label=gender_labels[int(gen_pred)]
        cv.putText(frame,gen_label,(x+w,y+h),cv.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),1)
        
        roi_age=cv.resize(roi,(200,200),interpolation=cv.INTER_AREA)
        #roi_age=img_to_array(roi_age)
        roi_age=np.expand_dims(roi_age,axis=0)
        age_pred=int(age_model.predict(roi_age))
        cv.putText(frame,f'{age_pred} years',(x,y+h),cv.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),1)
        
    cv.imshow('Video',frame)
    if cv.waitKey (20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()

cv.waitKey(0)