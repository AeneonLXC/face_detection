#修改正样本像素
import pandas as pd
import cv2
for n in range(1000,1606):#代表正数据集中开始和结束照片的数字
    path='D:\\facemask\\mask\\have_mask\\'+str(n)+'.jpg'
    # 读取图片
    img = cv2.imread(path)
    img=cv2.resize(img,(20,20)) #修改样本像素为20x20
    cv2.imwrite('D:\\facemask\\mask\\have_mask\\' + str(n) + '.jpg', img)
    n += 1