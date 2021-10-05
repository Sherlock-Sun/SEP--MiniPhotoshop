import cv2
import os
import time

def Button_camera(dirname):
    # 创建目录
    if (not os.path.isdir(dirname)):
        os.makedirs(dirname)

    # 打开摄像头进行人脸图像采集
    camera = cv2.VideoCapture(0)
    count = 0
    T = time.localtime()
    while (True):
        ret, frame = camera.read()
        cv2.imshow("camera", frame)
        # 输入q键退出
        if cv2.waitKey(100) & 0xff == ord("q"):
            break
        # 输入空格键时拍照
        if cv2.waitKey(100) & 0xff == ord(" "):
            cv2.imwrite(dirname + '/%s.jpg' % str(str(T.tm_year) + str(T.tm_mon) + str(T.tm_mday) + str(T.tm_hour) + str(T.tm_min) + str(count)), frame)
            count += 1

    camera.release()
    cv2.destroyAllWindows()

def Smile_photo(dirname):
    # 创建目录
    if (not os.path.isdir(dirname)):
        os.makedirs(dirname)
    # 打开摄像头进行人脸图像采集

    smileCascade = cv2.CascadeClassifier(r'venv\Lib\site-packages\cv2\data\haarcascade_smile.xml')
    face_cascade = cv2.CascadeClassifier(r'venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

    camera = cv2.VideoCapture(0)
    while(True):
        ret, frame = camera.read()
        cv2.imshow("camera", frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 10, flags = cv2.CASCADE_SCALE_IMAGE)

        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            smile = smileCascade.detectMultiScale(
                gray,
                scaleFactor=1.16,
                minNeighbors=35,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x1, y1, w1, h1) in smile:
                T = time.localtime()
                cv2.imwrite(dirname + '/%s.jpg' % str(str(T.tm_year) + str(T.tm_mon) + str(T.tm_mday) + str(T.tm_hour) + str(T.tm_min) + str(T.tm_sec)), frame)
                camera.release()
                cv2.destroyAllWindows()
                return;
        if cv2.waitKey(100) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Smile_photo("images")
    #print("test")
    #Button_camera("images") # 你生成的图片放在的电脑中的地方