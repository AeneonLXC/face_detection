# Python训练本地的人脸数据集

## 一、人脸口罩数据集下载处理

### （1）人脸数据集下载

​		下载人脸口罩数据集，利用opencv进行模型训练

![image-20201215235955091](C:%5CUsers%5C23608%5CDesktop%5COPENCV%E7%AC%94%E8%AE%B0%5Cimage-20201215235955091.png)



### （2）人脸数据集处理

#### 	1）对正样本数据集进行连续排序

```python
import os
path = r"C:\Users\23608\Desktop\facemask\have_mask" #人脸口罩数据集正样本的路径
filelist = os.listdir(path) #以列表的形式读取path里面的文件
count=1000 #开始文件名1000.jpg
for file in filelist:   
    Olddir=os.path.join(path,file)  #连接两个或更多的路径名组件
    if os.path.isdir(Olddir):   #返回一个列表，其中包含有指定路径下的目录和文件的名称
        continue
    filename=os.path.splitext(file)[0]   
    filetype=os.path.splitext(file)[1]
 
    Newdir=os.path.join(path,str(count)+filetype)  
    os.rename(Olddir,Newdir)
    count+=1
```

#### 	2）对负样本数据集进行连续排序

```python
import os
path = r"C:\Users\23608\Desktop\facemask\no_mask" #人脸口罩数据集正样本的路径
filelist = os.listdir(path)
count=10000 #开始文件名10000.jpg
for file in filelist:   
    Olddir=os.path.join(path,file)  
    if os.path.isdir(Olddir):  
        continue
    filename=os.path.splitext(file)[0]   #分离文件名与扩展名；默认返回(fname,fextension)元组
    filetype=os.path.splitext(file)[1]
 
    Newdir=os.path.join(path,str(count)+filetype)  #路径拼接
    os.rename(Olddir,Newdir)	#重命名
    count+=1
```

#### 	3）正样本图片剪切

```python
import pandas as pd
import cv2
for n in range(1000,1606):#代表正数据集中开始和结束照片的数字
    path=r'C:\Users\23608\Desktop\facemask\have_mask'+str(n)+'.jpg'
    # 读取图片
    img = cv2.imread(path)
    img=cv2.resize(img,(20,20)) #修改样本像素为20x20
    cv2.imwrite(r'C:\Users\23608\Desktop\facemask\have_mask\' + str(n) + '.jpg', img)
    n += 1
```

#### 	4）负样本图片剪切

```python
import pandas as pd
import cv2
for n in range(10000,11790):#代表正数据集中开始和结束照片的数字
    path=r'C:\Users\23608\Desktop\facemask\no_mask'+str(n)+'.jpg'
    # 读取图片
    img = cv2.imread(path)
    img=cv2.resize(img,(80,80)) #修改样本像素为20x20
    cv2.imwrite(r'C:\Users\23608\Desktop\facemask\no_mask\' + str(n) + '.jpg', img)
    n += 1
```

4)创建TXT正负样本数据集

在facemask目录下，输入

```
dir /b/s/p/w *.jpg > have_mask.txt
dir /b/s/p/w *.jpg > no_mask.txt
/b表示去除摘要信息，且顶格显示完整路径
/s表示枚举嵌套文件夹中的内容
```

#### 5)安装opencv

将opencv安装路径\opencv\build\x64\vc14\bin下的opencv_createsamples.exe、opencv_createsamples.exe放在facemask目录下

#### 6）对正样本txt文档进行预处理

```python
import os #处理文件和目录

Houzui=r" 1 0 0 20 20" #后缀
filelist = open(r'C:\Users\23608\Desktop\facemask\have_mask','r+')
line = filelist.readlines()
for file in line:
    file=file.strip('\n')+Houzui+'\n' #删除开头或是结尾的字符
    print(file)
    filelist.write(file)
```

#### 7）对负样本txt文档进行预处理

```python
import os

Houzui=r" 1 0 0 80 80" #后缀
filelist = open(r'C:\Users\23608\Desktop\facemask\no_mask','r+')
line = filelist.readlines()
for file in line:
    file=file.strip('\n')+Houzui+'\n'
    print(file)
    filelist.write(file)
```

#### 8)生成正负样本havemask.vec文件和nomask.vec文件,在cmd终端中进行，进入mask文件夹下，输入以下内容：

生成正样本havemask.vec文件：

```python
opencv_createsamples.exe -vec havemask.vec -info have_mask.txt -num 410 -w 20 -h 20
opencv_createsamples.exe -vec nomask.vec -info no_mask.txt -num 1688 -w 80 -h 80

info：样本说明文件
vec：样本描述文件名和路径
num：样本个数，这里为410个样本
w h：样本尺寸，这里为20x20
```



## 二、训练人脸口罩数据集模型

```
opencv_createsamples.exe 
用来生成正样本vec的，用来准备训练用的正样本数据和测试数据。他的输出为以 *.vec 为扩展名的文件，该文件以二进制方式存储图像。
```

## 三、进行人脸口罩识别

```python
import cv2

detector= cv2.CascadeClassifier(r'C:\Users\23608\Desktop\facemask\haarcascade_frontalface_default.xml')
mask_detector=cv2.CascadeClassifier(r'C:\Users\23608\Desktop\facemask\xml\cascade.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 3) #调用detectMultiScale()函数检测
   #scale_factor参数可以决定两个不同大小的窗口扫描之间有多大的跳跃，这个参数设置的大，则意味着计算会变快，但如果窗口错过了某个大小的人脸，则可能丢失物体。
   #指示寻找人脸的最小区域。设置这个参数过大，会以丢失小物体为代价减少计算量。
    for (x, y, w, h) in faces:
        #参数分别为 图片、左上角坐标，右下角坐标，颜色，厚度
        face=img[y:y+h,x:x+w]  # 裁剪坐标为[y0:y1, x0:x1]
        mask_face=mask_detector.detectMultiScale(gray, 1.1, 5)
        for (x2,y2,w2,h2) in mask_face:
            cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

    cv2.imshow('Mask Detector', img)
    cv2.waitKey(3)

cap.release()
cv2.destroyAllWindows()

```

