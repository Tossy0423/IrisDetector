"""
    Program Description:
    ----------
    iBPL2019(Yuntech & OIT, 18-31 Aug 2019)で, Group_Fが作成した"Iris Detector(虹彩認識)"を行うためのプログラム.
    iBPL2019 (Yuntech & OIT, 18-31 Aug 2019) is a program for performing "Iris Detector" created by Group_F.
    角膜の位置の検出を確認. 病気の判断まではできていない.
    Confirmed the detection of the position of the cornea.


    using PythonModules:
    ----------
    * numpy
    * opencv3
    * time
    * datetime


    Reference Documets:
    ----------
    * 【入門者向け解説】openCV顔検出の仕組と実践(detectMultiScale)
    https://qiita.com/FukuharaYohei/items/ec6dce7cc5ea21a51a82

    * Python+OpenCV で顔検出 - OpenCV に付属の評価器を試す
    https://blog.mudatobunka.org/entry/20162/10/03/014520

    * Canny法によるエッジ検出
    http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_canny/py_canny.html

    * Canny
    https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html

    * 物体検出（detectMultiScale）をパラメータを変えて試してみる（scaleFactor編)
    http://workpiles.com/2015/04/opencv-detectmultiscale-scalefactor/


"""

# ==========Import Modules========== #

import numpy as np
import cv2 as cv3
import time
from datetime import datetime as dt

print("----------Import Modules Clear----------")



# ==========Define GlobalValue========== #

# Set Camera Parameta
class CameraSetting:

    # Resolution
    ## Width
    S_width = 1920

    ## Height
    S_height = 1080

    # FPS
    S_fps = 30

    # Result
    ## Width
    R_width = 0x00

    ## Height
    R_height = 0x00

    ## FPS
    R_fps = 0x00

# Create Instance
CamSet = CameraSetting()


# Cascades File Import
"""
    Description: 
    ----------
    openCVが提供している, 顔の各部分を検出する比較器(カスケードファイル)をインポート.
    Imports a comparator (cascade file) that detects each part of the face provided by openCV.
    
    複数のカスケードファイルをインポートできる.
    You can import multiple cascade files.
    
    今回は"目"を認識するためのカスケードファイルをインポートしている.
    This time, a cascade file for recognizing "eyes" is imported.
    
"""
face_cascade = cv3.CascadeClassifier("./haarcascades/haarcascade_eye.xml")


print("----------Define GlobalValue Clear----------")


# ==========Define Function========== #

print("Set Import Location Cam")

"""
    Description:
    ----------
    画像をインポートする場所を指定するフラグ.
    Flag that specifies where to import images.
    
    Parameters:
    ----------
    * _Flag_ImportCam
        webカメラからインポート.
        Import from webcam.
        デバイス番号は, "VideoCapture(x)"より変更可能.
        Device number can be changed from "VideoCapture (x)".
                
    * _Flag_ImportGynoii
        無線カメラ"Gynoii"を使う.
        Use the wireless camera "Gynoii".
        今回はこのカメラを暗視カメラ(赤外線カメラ)として使い, 虹彩認識を行った.
        This time, this camera was used as a night vision camera (infrared camera) to perform iris recognition.
        
    * _Flag_ImportPic
        任意のファイルパスから画像ファイルをインポート.
        Import image files from any file path.
        logデータを使って処理を再現するために使用.
        Used to reproduce processing using log data.
"""
_Flag_ImportCam = False
_Flag_ImportGynoii = True
_Flag_ImportPic = False


# Preparation Capture
if(_Flag_ImportCam == True):
    cap1 = cv3.VideoCapture(1);

elif(_Flag_ImportGynoii == True):
    cap1 = cv3.VideoCapture("rtsp://192.168.0.1:554/live/0");

elif(_Flag_ImportPic == True):
    cap1 = cv3.imread("_img_src_20190829_134229.png", 1)


##Setting Camera
if(_Flag_ImportCam == True):

    cap1.set(3, CamSet.S_width)
    cap1.set(4, CamSet.S_height)
    cap1.set(5, CamSet.S_fps)

    # Result Camera Parameta
    CamSet.R_width = cap1.get(3)
    CamSet.R_height = cap1.get(4)
    CamSet.R_fps = cap1.get(5)
    print("[CameraPrameta]: width=%d, height=%d, fps=%d" % (CamSet.R_width,CamSet.R_height, CamSet.R_fps))



print("----------Define Function Clear----------")

# ====================Program Start==================== #
def main():


    # init function
    # Flags that the program needs to process
    _Flag_detect_eye = False
    _Flag_detect_circle = False


    """
        Description:
        ----------
        画像のlogを取得するためのフラグ.
        Flag to get image log.
        
        * True: 
            Hough変換が成功するごとにデータを出力する.(注意: 気がついたら大量のlogデータが実行ディレクトリにあるかも.)
            Output data every time Hough conversion succeeds (Note: If you notice, there may be a lot of log data in the execution directory)
    """
    _Flag_export_log = True


    """
        Description:
        ----------
        プログラムの動作周波数を設定する変数.
        Variable that sets the operating frequency of the program.
        
        "Gynoii"を使用する場合, ストリーミング速度より, プログラムの動作速度が早いためプログラムが強制終了しやすい.
        When "Gynoii" is used, the program is forcibly terminated because the program operation speed is faster than the streaming speed.
        
        ここで動作速度を制限することで, 少し改善した.
        By limiting the operating speed here, we improved a little.
        
        単位は, [Hz].
        The unit is [Hz].    
    """
    # Running Clock [Hz]
    _Param_RUNCLOCK = 50


    # ===== Loop ===== #
    while True:


        # ===== Import Image ===== #
        if((_Flag_ImportCam == True) | (_Flag_ImportGynoii == True)):
            ret, _img_src = cap1.read()
            if ret != True:
                print("[ERROR]: Please Check Conection PC to Camera")
        else:
            _img_src = cap1.copy()




        # ===== Convert to "grayscale" ===== #
        _img_gray = np.array((_img_src.shape[1], _img_src.shape[0]), dtype=np.uint8)
        _img_gray = cv3.cvtColor(_img_src, cv3.COLOR_BGR2GRAY)



        # ===== Detect eyes ===== #
        faces = face_cascade.detectMultiScale(_img_gray,            # img
                                              scaleFactor = 1.15,   # scale factor
                                              minNeighbors = 50,    # Minimum neighborhood rectangle size
                                              minSize = (10, 10),   # Min size of object
                                              maxSize = (300, 300)  # Max size of object
                                              )



        # ===== Triming image with eyes ===== #

        # Copy the input image to the image to draw the detected part.
        _img_drowrect = _img_src.copy()

        # init flag
        _Flag_detect_eye = False

        # Count variable to detect only one eye.
        _cnt_eyeloop = 0

        for (x, y, w, h) in faces:

            _Flag_detect_eye = True

            # 1回のみ実行
            if(1 <= _cnt_eyeloop):
                break

            else:
                # drow rect
                cv3.rectangle(_img_drowrect, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Triming Image
                _img_triming = _img_src[y : y+h, x : x+w]

                # Process Desiplay
                cv3.imshow("_img_triming", _img_triming)
                cv3.namedWindow("_img_triming", cv3.WINDOW_NORMAL)

            _cnt_eyeloop += 1


        # Process Display
        cv3.imshow("_img_drowrect", _img_drowrect)
        cv3.namedWindow("_img_drowrect", cv3.WINDOW_NORMAL)




        # ===== If you can detect your eyes ===== #
        if(_Flag_detect_eye == True):

            # Convert "BGR Color Space" to "Gray Scale"
            _img_gray_eye = cv3.cvtColor(_img_triming, cv3.COLOR_BGR2GRAY)


            # Emphasize contours using Canny edge detection
            _img_eye_canny = cv3.Canny(_img_gray_eye, 60, 60)

            # Invert Canny image
            _img_eye_canny = cv3.bitwise_not(_img_eye_canny)

            # Logical AND with cropped image of eyes
            _img_eye_canny = cv3.bitwise_and(_img_eye_canny, _img_gray_eye)

            # Process Display
            cv3.imshow("_img_eye_canny", _img_eye_canny)
            cv3.namedWindow("_img_eye_canny", cv3.WINDOW_NORMAL)

            # Extract circles with Hough transform.
            _hough_circle = cv3.HoughCircles(_img_eye_canny,        # image
                                        cv3.HOUGH_GRADIENT,         # method("cv3.HOUGH_GRADIENT" only use)
                                        dp          = 25,           # Voting resolution
                                        minDist     = 20,           # Minimum distance between detected circle centers
                                        param1      = 30,           # canny's larger threshold
                                        param2      = 50,           # Threshold for detecting the center of a circle
                                        minRadius   = 5,            # The minimum circle radius.
                                        maxRadius   = 10)           # The maximum circle radius.


            # ===== When a circle (cornea detection) is formed in the eye image ===== #
            if(_hough_circle is not None):


                # Define arrays for the number of detected data
                _pos_eye = np.zeros((_hough_circle.shape[1], 3), dtype=np.int8)

                # To display the result after Hough transform, copy the cropped image.
                _img_hough_dst = _img_triming.copy()


                # drow circle
                cv3.circle(_img_hough_dst,                                      # image
                           (_hough_circle[0][0][0], _hough_circle[0][0][1]),    # pos
                           (_hough_circle[0][0][2]),                            # radius
                           (255, 255, 255),                                     # color
                           1)                                                   # thinkness


                cv3.imshow("_img_hough_dst", _img_hough_dst)
                cv3.namedWindow("_img_hough_dst", cv3.WINDOW_NORMAL)


                # ===== Organize data obtained from Hough transform ===== #
                for i in range(0, _hough_circle.shape[1]):

                    for j in range(0, 3):

                        _pos_eye[i][j] = _hough_circle[0][i][j]

                    print("[Info, EyePos.]: [{}] x={}, y={}, Radius={}".format(i, _pos_eye[i][0], _pos_eye[i][1], _pos_eye[i][2]))


                # drow circle
                cv3.circle(_img_triming,                        # image
                           (_pos_eye[i][0], _pos_eye[i][1]),    # pos
                           (_pos_eye[i][2]),                    # radius
                           (0, 0, 255),                         # color
                           3)                                   # thinkness


                cv3.imshow("_img_hough_dst", _img_hough_dst)
                cv3.namedWindow("_img_hough_dst", cv3.WINDOW_NORMAL)

                _Flag_detect_circle = True

            else:
                _Flag_detect_circle = False



        # export log data
        if((_Flag_detect_circle == True) & (_Flag_export_log == True)):
            _time_event = dt.now()
            _str_time = _time_event.strftime("%Y%m%d_%H%M%S")

            print("exported picture")

            # Input image
            cv3.imwrite("_img_src_" + _str_time + ".png", _img_src)

            # Eye position detected by "detectMultiScale"
            cv3.imwrite("_img_drowrect_" + _str_time + ".png", _img_drowrect)

            # Result of Hough transform of eye trimming image
            cv3.imwrite("_img_hough_dst_" + _str_time + ".png", _img_hough_dst)



        # Control Program Running Clock
        time.sleep( 1/ _Param_RUNCLOCK )


        # Stop Program
        key = cv3.waitKey(1)
        if (key == ord('q')):
            print("[Info]: quit program")
            break

    cv3.destroyAllWindows()

# ====================Program End==================== #
print("----------Finish Program----------")

if __name__ == "__main__":
    main()
