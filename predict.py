from keras.models import load_model
import cv2
import numpy as np

model = load_model('tarp_final.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('t1 (15).jpg')
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])

classes = model.predict_classes(img)

if classes==0:
    print('This image is of class Ant ')
elif classes==1:
    print('This image is of class Aphids ')
elif classes==2:
    print('This image is of class Mites ')
elif classes==3:
    print('This image is of class Thrips ')
    
print (classes)
