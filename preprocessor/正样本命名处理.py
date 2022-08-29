import os
#   对数据进行命名
path = "D:\\facemask\\mask\\have_mask"  # 人脸口罩数据集正样本的路径
filelist = os.listdir(path)
count = 1000  # 开始文件名1000.jpg
for file in filelist:
    Olddir = os.path.join(path, file)
    if os.path.isdir(Olddir):
        continue
    filename = os.path.splitext(file)[0]
    filetype = os.path.splitext(file)[1]

    Newdir = os.path.join(path, str(count) + filetype)
    os.rename(Olddir, Newdir)
    count += 1