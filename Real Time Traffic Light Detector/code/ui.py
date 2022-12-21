import argparse
import platform
import shutil
import time
from numpy import random
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv7 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import numpy as np
import time

def load_model(
        weights=ROOT / 'best.pt',  # model.pt path(s)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference

):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    return model, stride, names, pt, jit, onnx, engine


def run(model, img, stride, pt,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.1,  # confidence threshold
        iou_thres=0.15,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        ):

    cal_detect = []

    device = select_device(device)
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    # Set Dataloader
    im = letterbox(img, imgsz, stride, pt)[0]

    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=augment)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()

            # Write results

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]}'
                lbl = names[int(cls)]
                #if lbl not in ['motorcycle','car', 'bus', 'truck']:
                    #continue
                cal_detect.append([label, xyxy])
    return cal_detect


def det_yolov7v6(info1):
    global model, stride, names, pt, jit, onnx, engine
    if info1[-3:] in ['jpg','png','jpeg','tif','bmp']:
        image = cv2.imread(info1)  # Read Items
        try:
            results = run(model, image, stride, pt)  # Recognize, return the results and location
            for i in results:
                box = i[1]
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                color = [255, 0, 0]
                xm1 = (p1[0] + p2[0]) / 2
                ym1 = (p1[1] + p2[1]) / 2

                k1 = (755 - 714) / (1749 - 0)
                b1 = 714 - k1 * 0
                if (k1 * xm1 + b1 - ym1) > 0:
                    color = [0, 0, 255]

                k2 = (700 - 682) / (1443 - 329)
                b2 = 682 - k2 * 329
                if (k2 * xm1 + b2 - ym1) < 0:
                    color = [0, 0, 255]

                    # ui.printf(
                    # str(time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))) + 'Detected Illegal parking of vehicles')
                cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
        except:
            pass
        cv2.line(image, (0, 714), (1749, 755), (0, 255, 0), 2)
        cv2.line(image, (329, 682), (1443, 700), (0, 255, 0), 2)
        ui.showimg(image)
    if info1[-3:] in ['mp4','avi']:
        es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
        kernel = np.ones((5, 5), np.uint8)
        background = None
        capture = cv2.VideoCapture(info1)
        while True:
            _, image = capture.read()

            if image is None:
                break

            imagecopy = image.copy()
            frame_lwpCV = imagecopy
            # The frame is preprocessed, the gray image is first transformed, and then the Gaussian filter is
            # carried out.
            # Gaussian filter is used for fuzzy processing. Reason for processing: each input video will generate noise
            # due to natural vibration, illumination changes or the camera itself. Noise is smoothed to avoid detection
            # during movement and tracking.
            gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
            gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

            # Set the first frame as the background for the entire input
            if background is None:
                background = gray_lwpCV
                continue
            # For each frame read from the background, the difference between it and Beijing is calculated and a
            # different map is obtained.
            # You also need to apply the threshold to get a black and white image and dilate the image by the code
            # below so that the holes and imperfection can be normalized
            diff = cv2.absdiff(background, gray_lwpCV)
            diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]  # Binarized threshold processing
            diff = cv2.dilate(diff, es, iterations=2)  # Morphological dilatation
            #ui.showimg(diff)
            #QApplication.processEvents()
            # 显示矩形框
            contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)  # This function computes the outline of an object in an image
            for c in contours:
                if  cv2.contourArea(c) < 3000:  # For rectangular areas, only Outlines that are larger than a given threshold are shown, so some minor changes will not be shown. For cameras with constant illumination and low noise, the threshold of minimum contour size can not be set
                    continue
                #print(cv2.contourArea(c))
                (x, y, w, h) = cv2.boundingRect(c)  # This function evaluates the bounding box of a rectangle
                # cv2.rectangle(frame_lwpCV, (x, y), (x+w, y+h), (0, 255, 0), 2)
                tl = round(0.002 * (frame_lwpCV.shape[0] + frame_lwpCV.shape[1]) / 2) + 1  # line/font thickness
                c1, c2 = (int(x), int(y)), (int(x + w), int(y + h))
                label = str(cv2.contourArea(c) * 24 / (4096 * 5 * 11))[:4] + 'm/s'
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

                yyy = (c1[1] + c2[1])/2 + 100
                xxx = (c1[0] + c2[0]) / 2


                if yyy > 700 and yyy < 785 and (xxx < 585 or xxx > 595) :
                    print(label)
                    print(xxx)
                    cv2.rectangle(image, c1, c2, [0,255,255], -1, cv2.LINE_AA)  # filled
                    cv2.putText(image, label, (c1[0], c1[1] - 2), 0, 1, [0, 0, 255], thickness=tf,
                                lineType=cv2.LINE_AA)


            try:
                results = run(model, imagecopy, stride, pt)  # Recognize, return multiple arrays each with the first result and the second coordinate position
                is_green = False
                for i in results:
                    if i[0] == 'G':
                        is_green = True
                        break


                for i in results:
                    box = i[1]
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    xm1 = p2[0] #(p1[0] + p2[0]) / 2
                    ym1 = p2[1] -20  #(p1[1] + p2[1]) / 2

                    k1 = (755 - 714)/(1749 - 0)
                    b1 = 714 - k1 * 0

                    k2 = (700 - 682) / (1443 - 329)
                    b2 = 682 - k2 * 329
                    # (k2 * xm1 + b2 - ym1) < 0 and

                    if (k1*xm1 +b1 - ym1) > 0 and  xm1 > 400 and xm1 < 1400:
                        if i[0] == 'person':
                            if is_green:
                                color = [0, 255, 0] # The green box marks pedestrians when the light is green
                                cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                            else:
                                color = [0, 0, 255] # Red boxes mark pedestrians at red lights
                                cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                                ui.printf(time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time())) + 'Alert! The light is red and someone is crossing the road')
                                print(time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time())) + 'Alert! The light is red and someone is crossing the road')
                            #ui.printf(
                                #str(time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))) + 'Warning! Vehicle parking is detected')
                        if i[0] == 'cane' or i[0] == 'wheelchair':
                            if is_green:
                                color = [255, 0, 255] # Purple for special groups
                                cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                            else:
                                color = [0, 0, 255]  # Red boxes mark pedestrians at red lights
                                cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                                ui.printf(time.strftime('%Y.%m.%d %H:%M:%S ',
                                                        time.localtime(time.time())) + 'Alert! At this time, the red light, there are special people crossing the road')
                                print(time.strftime('%Y.%m.%d %H:%M:%S ',
                                                    time.localtime(time.time())) + 'Alert! At this time, the red light, there are special people crossing the road')

                    if i[0] == 'R':
                        color = [0, 0, 255] # Red box for Red light
                        cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)

                    if i[0] == 'G':
                        color = [0, 255, 0] # Green box for Green light
                        cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)

                    if i[0] == 'Y': # Yellow box for Yellow light
                        color = [0, 144, 255]
                        cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)



            except:
                pass
            cv2.line(image, (0, 714), (1749, 755), (0, 255, 0), 2)
            cv2.line(image, (329, 682), (1443, 700), (0, 255, 0), 2)
            ui.showimg(image)
            QApplication.processEvents()

class Thread_1(QThread):  # Thread 1
    def __init__(self,info1):
        super().__init__()
        self.info1=info1
        self.run2(self.info1)

    def run2(self, info1):
        result = []
        result = det_yolov7v6(info1)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        MainWindow.setStyleSheet("background-image: url(\"./template/carui.png\")")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(29, 43, 1030, 105))
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("")
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setStyleSheet("font-size:50px;font-weight:bold;font-family:SimHei;background:rgba(255,255,255,0.2);border-radius:20px")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 188, 751, 501))
        self.label_2.setStyleSheet("background:rgba(255,255,255,0.7);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(73, 746, 851, 174))
        self.textBrowser.setStyleSheet("background:rgba(255,255,255,0.9);")
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1020, 750, 150, 40))
        self.pushButton.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1020, 810, 150, 40))
        self.pushButton_2.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(1020, 870, 150, 40))
        self.pushButton_3.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_3.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", ""))
        self.label.setText(_translate("MainWindow", ""))
        self.label_2.setText(_translate("MainWindow", "Click to add video"))
        self.pushButton.setText(_translate("MainWindow", "Choose Item"))
        self.pushButton_2.setText(_translate("MainWindow", "Start Detection"))
        self.pushButton_3.setText(_translate("MainWindow", "Quit"))

        # Click the text box Bind slot event
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.click_1)
        self.pushButton_3.clicked.connect(self.handleCalc3)

    def openfile(self):
        global sname, filepath
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        if fname == '':
            return
        filepath = os.path.normpath(fname)
        sname = filepath.split(os.sep)
        ui.printf("Current Path：%s" % filepath)


    def handleCalc3(self):
        os._exit(0)

    def printf(self,text):
        self.textBrowser.append(text)
        self.cursor = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    def showimg(self,img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 800
        else:
            ratio = n_height / 800
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.label_2.setPixmap(QPixmap.fromImage(new_img))

    def click_1(self):
        global filepath
        try:
            self.thread_1.quit()
        except:
            pass
        self.thread_1 = Thread_1(filepath)  # Create thread
        self.thread_1.wait()
        self.thread_1.start()  # Start thread


if __name__ == "__main__":
    global model, stride, names, pt, jit, onnx, engine
    model, stride, names, pt, jit, onnx, engine = load_model()  # Load model
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
