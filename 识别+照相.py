import cv2  # 图像处理的库 OpenCv
import dlib  # 人脸识别的库 dlib



# 使用特征提取器 get_frontal_face_detector
detector = dlib.get_frontal_face_detector()
# dlib 的68点模型，使用官方训练好的特征预测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 建cv2摄像头对象，参数0表示打开电脑自带的摄像头，如果换了外部摄像头，则自动切换到外部摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# set(propId, value),设置视频参数，propId设置视频参数， value设置参数值
#cap.set(3, 480)
# 截取 screenshoot 的计数器
i = 1

while cap.isOpened():
    flag, img_rd = cap.read()

    kk = cv2.waitKey(1)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    # 人脸 / Faces
    faces = detector(img_rd, 0)

    # 复制，用于照相
    frame = img_rd.copy()

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
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())
            cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)




    # 6. 按下 'q' 键退出 / Press 'q' to exit
    if kk == 27:
        print('goodbye')
        break
    elif kk == ord('q'):

        cv2.imwrite('zz' + str(i) + '.jpg', frame)
        print(i)
        i += 1

   


    cv2.imshow("camera", img_rd)

# 释放摄像头 / Release camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
