import cv2
import face
import win32com.client as win

def speak(name, score):
    speaker = win.Dispatch("SAPI.SpVoice")
    if name == '0001':
        name = '李 '
    elif name == '0002':
        name = '李 '
    elif name == '0003':
        name = '刘 '
    elif name == '0004':
        name = '卢 '
    if score == -1:
        content = '未检测到人脸'
    else:
        content = name + ',你好，人脸相似度是 百分之' + str(int(score))
    speaker.Speak(content)


cap = cv2.VideoCapture(0)   #   调用USB摄像头
classifier = cv2.CascadeClassifier('cv2_need.xml')  #   加载本地人脸特征数据集


while True:
    image = cap.read()
    image = image[1]
    faceData = classifier.detectMultiScale(image,scaleFactor=1.5, minNeighbors=2, minSize=(10,100))


    for x, y, w, h, in  faceData:

        cv2.rectangle(image, (x, y), (x+h, y+w), (0, 255, 0), 2)  #   绘制识别人脸的方框
    # print(faceData)
    cv2.imshow('test', image)   #   创建摄像头画面窗口，展示出来
    if cv2.waitKey(10) == 13:
        image = image[y:y+w, x:x+h]
        name, score = face.search(image)
        speak(name, score)


