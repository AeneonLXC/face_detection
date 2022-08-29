import cv2

cap = cv2.VideoCapture(0)

MINthreshold = 10
MAXthreshold = 100

while True:
    photo, vedio = cap.read()  # 获取画面
    gray = cv2.cvtColor(vedio, cv2.COLOR_BGR2GRAY)  # 转灰
    GaussianBlur = cv2.blur(gray, (7, 7), 0)  # (x, y,)x,y越大，图像的模糊程度越大;参数0表示标准差取0。

    Canny = cv2.Canny(GaussianBlur, MINthreshold, MAXthreshold)  # 50是最小阈值,150是最大阈值

    #   Sobel
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)

    # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
    # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像

    Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
    Scale_absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)

    cv2.imshow("CannyThreshold", Canny)  # 显示边缘检测画面
    cv2.imshow("SobelThreshold", Sobel)
    key_word = cv2.waitKey(60)  # 获取键盘按键

    if cv2.waitKey(2) & 0xff == ord('A'):  # 按下a键进行图片保存
        cv2.imwrite(r'C:\Users\23608\Desktop\IMG\CannyThreshold\canny.PNG', Canny)
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
