from aip import AipFace
import cv2
import base64
Face = AipFace(' ', ' ', ' ')  #调用百度aip （App账号， API密码，）


def toBase(img):
    retval, buffer = cv2.imencode('.jpg', img)
    pic_str = base64.b64encode(buffer)
    pic_str = pic_str.decode()
    return pic_str


def add(image, name):
    image = toBase(image)
    Face.addUser(image, 'BASE64', 'python', name) # 图片信息 图片类型 用户组 图片名字。百度Aip支持的图片格式是base64 将图片转换为base64格式


def search(image):
    image = toBase(image)
    result = Face.search(image, 'BASE64', 'python')
    try:
        result = result['result']['user_list'][0]
        return result['user_id'], result['score']
    except:
        return -1, -1