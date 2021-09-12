import cv2
from PIL import Image
from numpy import asarray,expand_dims
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions


def extract_faces(filename,req_size = (224,224)):
    face_cascade = cv2.CascadeClassifier
    ("/Users/shreyashrivastava/opt/anaconda3/pkgs/libopencv-4.5.2-py39h852ad08_1/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face = img[y:y+h,x:x+w]
        image = Image.fromarray(face)
        image = image.resize(req_size)
        face_array= asarray(image)

    return face_array
        

img='/Users/shreyashrivastava/Desktop/XYZ.png'
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixels = extract_faces(img)
# convert one face into samples
pixels = pixels.astype('float32')
samples = expand_dims(pixels, axis=0)
# prepare the face for the model, e.g. center pixels
samples = preprocess_input(samples, version=2)
# create a vggface model
model = VGGFace(model='resnet50')
# perform prediction
yhat = model.predict(samples)
# convert prediction into names
results = decode_predictions(yhat)
# display most likely results
for result in results[0]:
	print('%s: %.3f%%' % (result[0], result[1]*100))
'''pixels =cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
# plot the extracted face
plt.imshow(pixels)
# show the plot
plt.show()

model = VGGFace(model='resnet50')
# summarize input and output shape
print('Inputs: %s' % model.inputs)
print('Outputs: %s' % model.outputs)
yhat = model.predict(samples)

'''








































