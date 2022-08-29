import cv2

cap = cv2.VideoCapture(1)  # 使用第0个摄像头
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 加载人脸特征库
face_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
smile = cv2.CascadeClassifier('haarcascade_smile.xml')
while True:
    ret, frame = cap.read()  # 读取一帧的图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转灰
    frame = cv2.flip(frame, 180)

    # 边缘检测 GaussianBlur = cv2.blur(gray, (x, y), 0)
    GaussianBlur = cv2.blur(gray, (7, 7), 0)  # （x,y）越小 图像边缘检测越复杂
    GaussianBlur = cv2.Canny(GaussianBlur, 0, 30, 30)  # 差值越大 图像边缘越不清楚
    # scaleFactor表示每次图像尺寸减小的比例
    # minNeighbors表示每一个目标至少要被检测到10次才算是真的目标
    # minSize为目标的最小尺寸与最大尺寸

    # 检测人脸
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.15, minNeighbors=10, minSize=(5, 5))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 用矩形圈出人脸 name,
        # 坐标(x, y), (x+w, y+h), 颜色rgb, 粗细
        face_area = frame[y:y + h, x:x + w]
        cv2.putText(frame, 'face', (x, y - 4), 3, 1.2, (0, 0, 255), 1)
        # x,y ------
        # |          |
        # |          |
        # |          |
        # --------x+w， y+h
        eyes = face_eye.detectMultiScale(frame, scaleFactor=1.15, minNeighbors=20, minSize=(5, 5))
        for (ex, ey, ew, eh) in eyes:
            # 画出眼眶
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)  # 用矩形圈出人脸
            # 用矩形画出眼眶
            # name,
            # 坐标(x, y), (x+w, y+h), 颜色rgb, 粗细

            # 微笑检测
        smiles = smile.detectMultiScale(face_area, scaleFactor=1.16, minNeighbors=65, minSize=(25, 25))
        for (ex, ey, ew, eh) in smiles:
            # 画出微笑框
            cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
            # 用矩形画出微笑框
            # name,
            # 坐标(x, y), (x+w, y+h), 颜色rgb, 粗细

    cv2.imshow('face list', frame)
    cv2.imshow("bianyuan", GaussianBlur)  # 显示边缘检测画面

    # 获取关键词
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 按下q退出
        break

cap.release()  # 释放摄像头
cv2.destroyAllWindows() # 关闭窗口
