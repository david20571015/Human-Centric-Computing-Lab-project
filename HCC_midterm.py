import sys
sys.path.append('./yolov5')

import cv2
import numpy as np
from yolov5.models import *

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

import time
import argparse

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize

# from djitellopy import Tello
from droneHCC import Drone


def detect(frame, model, opt, img_size=736):
    frame = frame.unsqueeze(0)
    frame = Resize((img_size, img_size))(frame)

    # Get detections
    with torch.no_grad():
        out, _ = model(frame, augment=opt.augment)
        out = non_max_suppression(out, opt.conf_thres, opt.nms_thres, multi_label=True)

    xy1, xy2, conf, labels, center= [], [], [], [], []
    x_scale, y_scale = 960 / img_size, 720 / img_size
    for pred in out[0]:
        x1, x2 = map(lambda x: int(x * x_scale), pred[:3:2])
        y1, y2 = map(lambda y: int(y * y_scale), pred[1:4:2])
        xy1.append([x1, y1])
        xy2.append([x2, y2])
        center.append([(x1 + x2) // 2, (y1 + y2) // 2])
        conf.append(pred[4].item())
        labels.append(int(pred[5]))

    return xy1, xy2, conf, labels, center, len(labels)

def render(frame, xy1, xy2, conf, center, labels, classes):
    # x_scale, y_scale = 960 / 640, 720 / 640
    bbox_num = len(labels)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5

    for i in range(bbox_num):
        x1, y1 = xy1[i]
        x2, y2 = xy2[i]
        label = classes[labels[i]]

        # bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.circle(frame, tuple(center[i]), 3, (0, 255, 0))

        # label
        label_size, _ = cv2.getTextSize(label, font, font_size, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), (255, 0, 0), -1)
        cv2.putText(frame, label, (x1, y1), font, font_size, (255, 255, 255), 1)

    def pekora():
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="weights/best.pt", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="classes.txt", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument('--augment', action='store_true', default=False, help='augmented inference')
    parser.add_argument("--img_size", type=int, default=640, help="size of each image dimension")
    parser.add_argument("--height", type=int, default=0, help="height level")
    stella = ['Aries', 'Leo', 'Sagittarius', 'Taurus', 'Virgo', 'Capricorn', \
                     'Gemini', 'Libra', 'Aquarius', 'Cancer', 'Scorpio', 'Pisces']
    parser.add_argument("-t", "--target", type=int, default=0, help="target")
    opt = parser.parse_args()
    print(opt)

    target = stella[opt.target]
    print(f'target = {target}')

    with open(opt.class_path, 'r') as class_file:
        classes = class_file.read().split("\n")[:-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = attempt_load(opt.weights_path, map_location=device)
    model.eval()

    fr = 0
    r = 3

    drone = Drone(
        # stable parameters
        kp = np.array([0.115, 0.115, 0.1, 25]),
        ki = np.array([0.001, 0.001, 0, 0]),
        kd = np.array([0.12, 0.12, 0, 0])
    )

    err = np.zeros((4,))
    errSum = np.zeros((4,))
    prevErr = np.zeros((4,))

    drone.takeoff()
    drone.send_rc_control(0,0,70,0)
    height = 0 if opt.height == 0 else 80
    while drone.get_height() < height:
        print(drone.get_height())
    drone.stop()


    is_aimmed = False
    img_ratio = 0

    # 本番
    while True:
        fr += 1
        frame = drone.background_frame_read.frame
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (fr % r == 0):
            xy1, xy2, conf, labels, center, num = detect(TF.to_tensor(frame).to(device), model, opt, 640)

            mydick = {} # {'stellar': found_idx}

            for i in range(num):
                # print("class: {}, conf: {}, center: {}".format(classes[labels[i]], conf[i], center[i]))
                mydick[classes[labels[i]]] = i

            render(frame, xy1, xy2, conf,center, labels, classes)

            cv2.imshow("plot", frame)
            cv2.waitKey(10)

            if target in mydick:

                found_idx = mydick[target]
                sqrt_area = abs((xy2[found_idx][0] - xy1[found_idx][0]) * (xy2[found_idx][1] - xy1[found_idx][1])) ** 0.5
                center_x, center_y = center[found_idx]
                print(sqrt_area**2/720/960)

                if is_aimmed or 960*2/5 < center_x < 960*3/5 and 720*2/5 < center_y < 720*3/5:
                    if not is_aimmed:
                        errSum = np.zeros((4,))
                    print(sqrt_area)
                    is_aimmed = True
                    # PID forward&up
                    err = np.array([0.8*(center_x - 480),
                                    1.2*(360 - center_y),
                                    500 - sqrt_area, 0]) # (720*960*0.35)**0.5
                    result, errSum, prevErr = drone.PID(err, errSum, prevErr)
                    drone.send_rc_control(*map(int, [result[0], result[2], result[1], 0]))

                    if sqrt_area > img_ratio:
                        drone.stop()
                        if sqrt_area > 500 and 0<xy1[found_idx][0] and 0<xy1[found_idx][1] and xy2[found_idx][0] < 960 and xy2[found_idx][1] < 720:
                            drone.land()
                            exit()
                        cv2.imwrite(f'screenshots/{target}_0.jpg', frame)
                        img_ratio = sqrt_area*1.02

                elif not is_aimmed:
                    err = np.array([center_x - 480,
                                    360 - center_y,
                                    sqrt_area - 500, 0]) # (720*960*0.35)**0.5
                    result, errSum, prevErr = drone.PID(err, errSum, prevErr)
                    drone.send_rc_control(*map(int, [result[0], 0, result[1], 0]))
            else:
                if is_aimmed:
                    # target not found after aimmed => backward
                    drone.send_rc_control(0, -15, 0, 0)
                else:
                    drone.send_rc_control(0, 0, 0, 0)

    cv2.destroyAllWindows()
