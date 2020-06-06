import cv2  
import dlib  
import random

# 开心 惊讶 自然 愤怒 悲伤 恐惧 厌恶
"""
happy：咧嘴不皱眉
disgust：咧嘴皱眉
surprise：不皱眉张嘴
fear：皱眉眼睛大张嘴
angry：撅嘴皱眉眼睛大
sad：撅嘴眼睛小
"""



def puttext(emotion, cartoon, name):
    happy_word = ["hahahaha", "Today is a good day!", "Aha"]
    angry_word = ["piss me off", "Don't bother me!"]
    disgust_word = ["disgusting", "hate you", "stay away from me", "Ugh"]
    sad_word = ["Life is so hard.", "TAT", "feel blue now..."]
    fear_word = ["It's' horrible!", "That sounds awful."]
    surprise_word = ["What a surprise!", "This shocked me", "Wow!", "Awesome", "can't believe it"]
    nature_word = ["just so so", "emmmm"]
    emotion_list = [happy_word, angry_word, disgust_word, sad_word, fear_word, surprise_word, nature_word]
    choice = ["happy", "angry", "disgust", "sad", "fear", "surprise", "nature"]
    location = choice.index(str(emotion))
    cv2.putText(cartoon, random.choice(emotion_list[location]), (d.left(), d.bottom() + 50), font, 1,
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    cv2.imwrite(str(name), cartoon)


def cartoonise(emotion, picture_name, num):
    num_bilateral = 10  # 定义双边滤波的数目

    img_rgb = cv2.imread(picture_name)

    # 用高斯金字塔降低取样
    img_color = img_rgb

    img_color = cv2.pyrDown(img_color)

    # 重复使用小的双边滤波代替一个大的滤波
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9,
                                        sigmaColor=9,
                                        sigmaSpace=7)
    # 升采样图片到原始大小
    img_color = cv2.pyrUp(img_color)

    # 转换为灰度并使其产生中等的模糊
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 3)
    # 使用自适应阈值处理灰度图像创建轮廓
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=5, C=7)
    # 转换回色彩图像
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    # 叠加彩色图像和轮廓
    img_cartoon = cv2.addWeighted(img_color, 0.8, img_edge, 0.2, 0)
    # 命名并保存图像
    name = 'cartoon/' + str(emotion) + str(num) + '.jpg'
    puttext(emotion, img_cartoon, name)


def screenshoot(emotion, num):
    name = f'{emotion}/picture{num}' + '.jpg'
    cv2.imwrite(name, draw_img)
    cartoonise(str(emotion), name, num)
    print(f"{emotion},{num}")


# 使用特征提取器 get_frontal_face_detector
detector = dlib.get_frontal_face_detector()
# dlib 的68点模型，使用官方训练好的特征预测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 建cv2摄像头对象，参数0表示打开电脑自带的摄像头，如果换了外部摄像头，则自动切换到外部摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 截取 screenshoot 的计数器
happy, disgust, fear, sad, angry, surprise, nature = 1, 1, 1, 1, 1, 1, 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")

    kk = cv2.waitKey(1)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 人脸 / Faces
    faces = detector(frame, 0)

    # 字体
    font = cv2.FONT_ITALIC
    # 复制，用于照相
    draw_img = frame.copy()
    cv2.putText(frame, "Press 'q' to take a picture", (10, 20), font, 0.6, (0, 200, 0), 2)
    cv2.putText(frame, "Press 'Esc' to end", (10, 50), font, 0.6, (0, 200, 0), 2)
    # 检测到人脸 / Face detected
    if len(faces) != 0:
        # 矩形框 / Show the rectangle box of face
        for k, d in enumerate(faces):
            # 计算矩形大小
            # (x,y), (宽度width, 高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # 计算矩形框大小 / compute the size of rectangle box
            face_height = (d.bottom() - d.top())
            face_width = (d.right() - d.left())
            # 用黄色矩阵框出人脸, 光的三原色Red(0,0,255), Green(0,255,0), Blue(255,0,0)
            #  rectangle(img, pt1, pt2, color)
            cv2.rectangle(frame, pos_start, pos_end, (0, 255, 255), 2)
            shape = predictor(frame, d)
            for i in range(68):
                # circle(img, center, radius, color),
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), 1)
            # 分析任意 n 点的位置关系来作为表情识别的依据
            # 嘴中心	66，嘴左角48，嘴右角54
            mouth_width = (shape.part(54).x - shape.part(48).x) / face_width  # 嘴巴张开宽度
            mouth_height = (shape.part(66).y - shape.part(62).y) / face_height  # 嘴巴张开高度
            eye_height = (shape.part(41).y - shape.part(37).y) / face_height  # 眼睛睁开程度
            frown_sum = 0  # 两边眉毛距离之和
            for j in range(17, 21):
                frown_sum += shape.part(j + 5).x - shape.part(j).x
            frown_sum = frown_sum / face_width

            if mouth_width >= 0.34:
                if frown_sum <= 1.8:
                    cv2.putText(frame, "disgust", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, 4)
                    if kk == ord('q'):
                        screenshoot("disgust", disgust)
                        disgust += 1

                else:
                    cv2.putText(frame, "happy", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, 4)
                    if kk == ord('q'):
                        screenshoot("happy", happy)
                        happy += 1

            elif mouth_height >= 0.05:
                if frown_sum <= 1.8:
                    cv2.putText(frame, "fear", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, 4)
                    if kk == ord('q'):
                        screenshoot("fear", fear)
                        fear += 1
                else:
                    cv2.putText(frame, "surprise", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, 4)
                    if kk == ord('q'):
                        screenshoot("surprise", surprise)
                        surprise += 1

            elif mouth_width <= 0.26:
                if eye_height < 0.03:
                    cv2.putText(frame, "sad", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, 4)
                    if kk == ord('q'):
                        screenshoot("sad", sad)
                        sad += 1
                else:
                    cv2.putText(frame, "angry", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, 4)
                    if kk == ord('q'):
                        screenshoot("angry", angry)
                        angry += 1
            else:
                cv2.putText(frame, "nature", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, 4)
                if kk == ord('q'):
                    screenshoot("nature", nature)
                    nature += 1

    else:
        # 没有检测到人脸
        cv2.putText(frame, "No Face", (20, 50), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # 按下 'Esc' 键退出
    if kk == 27:
        print('goodbye')
        break

    cv2.imshow("emotion", frame)

# 释放摄像头
cap.release()
cv2.destroyAllWindows()


