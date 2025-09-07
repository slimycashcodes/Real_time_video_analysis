import cv2

def draw_boundary(img,classifier,scaleFactor,minNeighbours,color,text):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img,scaleFactor,minNeighbours)
    coords = []
    for(x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        coords = [x,y,w,h]
    return coords

def detect(img,faceCascade,eyeCascade,smileCascade):
    color = {"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0)}
    coords = draw_boundary(img,faceCascade,1.1,10,color['blue'],"Face")
    if len(coords)==4:
        roi_img = img[coords[1]:coords[1]+coords[3]+5,coords[0]:coords[0]+coords[2]+5]
        #coords = draw_boundary(roi_img,eyeCascade,1.1,1,color['red'],"Eyes")
        coords = draw_boundary(roi_img,smileCascade,1.7,20,color['green'],"Smile")

    return img

video_capture = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

while True:
    _, img = video_capture.read()
    img = cv2.flip(img,2)
    img = detect(img,faceCascade,eyeCascade,smileCascade)

    cv2.imshow("face detection",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
