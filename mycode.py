import cv2  # 图像处理的库 OpenCv
import dlib  # 人脸识别的库 dlib
import numpy as np  # 数据处理的库 numpy

# 开心 惊讶 正常 愤怒 悲伤 恐惧 厌恶
"""
surprise：不皱眉张嘴
disgust：咧嘴皱眉
anger：撅嘴
fear：皱眉眼睛大张嘴
sad：撅嘴，眼睛小
happy：嘴宽度大

"""


import os


def cartoonise(picture_name, j):

    num_bilateral = 7  # 定义双边滤波的数目

    img_rgb = cv2.imread(picture_name)

    # 用高斯金字塔降低取样
    img_color = img_rgb

    img_color = cv2.pyrDown(img_color)

    # 重复使用小的双边滤波代替一个大的滤波
    for _ in range(10):
        img_color = cv2.bilateralFilter(img_color, d=9,
                                        sigmaColor=9,
                                        sigmaSpace=7)
    # 升采样图片到原始大小
    img_color = cv2.pyrUp(img_color)

    # 转换为灰度并使其产生中等的模糊
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 3)
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=5,C=7)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    #img_cartoon = cv2.bitwise_and(img_color, img_edge)
    img_cartoon = cv2.addWeighted(img_color,0.8, img_edge,0.2,0)
    cv2.imwrite('zzcartoon' + str(j) + '.jpg', img_cartoon)



# 使用特征提取器 get_frontal_face_detector
detector = dlib.get_frontal_face_detector()
# dlib 的68点模型，使用官方训练好的特征预测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 建cv2摄像头对象，参数0表示打开电脑自带的摄像头，如果换了外部摄像头，则自动切换到外部摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# set(propId, value),设置视频参数，propId设置视频参数， value设置参数值
#cap.set(3, 480)
# 截取 screenshoot 的计数器
w = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")

    kk = cv2.waitKey(1)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 人脸 / Faces
    faces = detector(frame, 0)

    # 待会要写的字体 / Font to write
    font = cv2.FONT_ITALIC
    # 复制，用于照相
    draw_img = frame.copy()

    # 检测到人脸 / Face detected
    if len(faces) != 0:
        # 矩形框 / Show the rectangle box of face
        for k, d in enumerate(faces):
            # 计算矩形大小
            # Compute the width and height of the box
            # (x,y), (宽度width, 高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # 计算矩形框大小 / compute the size of rectangle box
            face_height = (d.bottom() - d.top())
            face_width = (d.right() - d.left())
            # 用红色矩阵框出人脸, 光的三原色Red(0,0,255), Green(0,255,0), Blue(255,0,0)
            #  rectangle(img, pt1, pt2, color), 其中pt1为矩阵上顶点，pt2为矩阵下顶点
            cv2.rectangle(frame, pos_start, pos_end, (0, 255, 255), 2)
            shape = predictor(frame, d)
            for i in range(68):
                # circle(img, center, radius, color),
                # img,Image where the circle is drawn
                # center,Center of the circle
                # radius,Radius of the circle (半径)
                # color，Circle color
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), 1)
                #cv2.putText(frame, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            # 分析任意 n 点的位置关系来作为表情识别的依据
            # 嘴中心	66，嘴左角48，嘴右角54
            mouth_width = (shape.part(54).x - shape.part(48).x) / face_width  # 嘴巴张开程度
            mouth_height = (shape.part(66).y - shape.part(62).y) / face_height # 嘴巴张开程度
            # print("嘴巴宽度与识别框宽度之比：" , mouth_width)
            # print("嘴巴高度与识别框宽度之比：" , mouth_height)
            eye_height = (shape.part(41).y - shape.part(37).y) / face_height
            #print("eye", eye_height)
            frown_sum = 0  # 两边眉毛距离之和
         
            for j in range(17, 21):
                frown_sum += shape.part(j + 5).x - shape.part(j).x
            # 与脸宽度比例
            frown_sum = frown_sum / face_width
                #print(frown_sum/face_width)


            if mouth_width >= 0.34:
                if frown_sum <= 1.8:
                        cv2.putText(frame, "disgust", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, 4)
                else:
                        cv2.putText(frame, "happy", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)
            elif mouth_height >= 0.05:
                if frown_sum <= 1.8:
                    cv2.putText(frame, "fear", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, 4)
                else:
                        cv2.putText(frame, "surprise", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)
            elif mouth_width <= 0.26:
                if eye_height < 0.03:
                    cv2.putText(frame, "sad", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                  2, 4)
                else:
                    cv2.putText(frame, "angry", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2, 4)

            else:
                cv2.putText(frame, "nature", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2, 4)





    else:
        # 没有检测到人脸
        cv2.putText(frame, "No Face", (20, 50), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # 6. 按下 'q' 键退出 / Press 'q' to exit
    if kk == 27:
        print('goodbye')
        break
    elif kk == ord('q'):

        cv2.imwrite('zz' + str(w) + '.jpg', draw_img)
        cartoonise('zz' + str(w) + '.jpg', w)

        print(w)
        w += 1


    cv2.imshow("camera", frame)

# 释放摄像头 / Release camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
