import dlib
import cv2
import numpy as np
import math
import time

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    land_marks = []

    rects = detector(img_gray, 1)
    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
        land_marks.append(land_marks_node)
    return land_marks

'''
reference: CSDN, https://blog.csdn.net/grafx/article/details/70232797?locationNum=11&fps=1
'''
# m = (endX, endY), 圆心c = (startX, startY), x = (i, j)
def localTranslationWarp(srcImg, startX, startY, endX, endY, radius):
    ddradius = float(radius * radius)
    copyImg = srcImg.copy()

    #计算公式中的|m-c|^2
    ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    dleft = startX - int(radius)
    dright = startX + int(radius)
    dup = startY + int(radius)
    ddown = startY - int(radius)
    H, W, C = srcImg.shape

    for i in range(dleft, dright):
        for j in range(ddown, dup):

            if i < 0 and j < 0 and i >= W and j >= W:
                continue

            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)
            # 再判断是否在形变圆中
            if (distance < ddradius):
                # 计算出(i, j)的原坐标
                # 计算平方号里的值
                ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                ratio = ratio * ratio
                parameter = float(math.sqrt(ddradius / ((endX - startX) * (endX - startX)
                                                 + (endY - startY) * (endY - startY))))
                ratio = ratio * parameter

                # 映射原位置
                UX = i - ratio * (endX - startX)
                UY = j - ratio * (endY - startY)

                if int(UX) < W - 1 and int(UY) < H - 1:
                    # 根据双线性插值法得到UX和UY的值
                    value = BilinearInsert(srcImg, UX, UY)
                    # 改变当前i, j的值
                    copyImg[j, i] = value

    return copyImg

# 双线性插值法
def BilinearInsert(src, ux, uy):
    h, w, c = src.shape
    if c == 3 and int(ux) + 1 < w and int(uy) + 1 < h:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1

        part1 = src[y1, x1].astype(np.float64) * (float(x2) - ux) * (float(y2) - uy)
        part2 = src[y1, x2].astype(np.float64) * (ux - float(x1)) * (float(y2) - uy)
        part3 = src[y2, x1].astype(np.float64) * (float(x2) - ux) * (uy - float(y1))
        part4 = src[y2, x2].astype(np.float64) * (ux - float(x1)) * (uy - float(y1))

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.int8)

def face_thin_auto(src):
    landmarks = landmark_dec_dlib_fun(src)
    thin_image = src.copy()

    # 如果检测不到人脸就退出
    if len(landmarks) == 0:
        return

    for step in range(len(landmarks)):
        landmarks_node = landmarks[step]
        left_landmark = landmarks_node[3]  # 瘦脸的起始位置
        left_landmark_down = landmarks_node[5]

        right_landmark = landmarks_node[13]  # 瘦脸的起始位置
        right_landmark_down = landmarks_node[15]

        endPt = landmarks_node[30]

        # 计算第四个点到第六个点的距离作为瘦脸距离
        r_left = math.sqrt(
            (left_landmark[0, 0] - left_landmark_down[0, 0]) * (left_landmark[0, 0] - left_landmark_down[0, 0]) +
            (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1])
        )
        # 计算第12个点到第14个点的距离作为瘦脸距离
        r_rihgt = math.sqrt(
            (right_landmark[0, 0] - right_landmark_down[0, 0]) * (
                    right_landmark[0, 0] - right_landmark_down[0, 0]) +
            (right_landmark[0, 1] - right_landmark_down[0, 1]) * (right_landmark[0, 1] - right_landmark_down[0, 1])
        )

        # 瘦左脸
        thin_image = localTranslationWarp(thin_image, left_landmark[0, 0], left_landmark[0, 1],
                                          endPt[0, 0], endPt[0, 1], r_left)
        # 瘦右脸
        thin_image = localTranslationWarp(thin_image, right_landmark[0, 0], right_landmark[0, 1],
                                          endPt[0, 0], endPt[0, 1], r_rihgt)

        # 显示
        cv2.imshow('thin', thin_image)
        cv2.imwrite('thin_single.jpg', thin_image)

if __name__ == '__main__':
    src = cv2.imread('R-C.jfif')
    cv2.imshow('sample', src)
    #p1 = time.time()
    face_thin_auto(src)
    #p2 = time.time()
    #print(p2 - p1)
    cv2.waitKey(0)