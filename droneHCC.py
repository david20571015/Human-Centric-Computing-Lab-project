from djitellopy import Tello
from pynput import keyboard
# https://pynput.readthedocs.io/en/latest/keyboard.html
import time
import cv2
import numpy as np

""" HCC HCC HCC HCC HCC HCC HCC HCC HCC HCC"""

""" requirements
pip install djitellopy2
pip install pynput
pip install numpy
"""

# from droneHCC import Drone
""" Usage of this code:
This code is inherited from Tello in djitellopy
reference:
https://github.com/damiafuentes/DJITelloPy/blob/aad98c1e8d8fd916112e7e52eba1318398adb6ad/djitellopy/tello.py
basically, you only need
.move_forward(cm), .move_up(cm) ...
.rotate_clockwise(deg)
.land()
.takeoff()

I built some other useful extensions in this code
find example codes in the main function"""


class Drone(Tello):
    def __init__(self, kp, ki, kd, video=True) -> None:
        Tello.__init__(self)
        self.connect()  # command
        print(f'battery: {self.get_battery()}')

        self.kp = kp
        self.ki = ki
        self.kd = kd

        if video:
            self.streamon()
            self.get_frame_read()

        # keyboard control
        self.keyboardMonitor()
        self.joystick_move = False

        # wait for booting
        time.sleep(0.5)
        print('start...')

    """ keyboard related """

    def joystick(self, key):
        """ Used in keyboard Monitor. """
        # learn the control key by yourself
        speed = 65
        if key == 'o':
            if not self.is_flying:
                self.send_rc_control(0, 0, 0, 0)
                self.takeoff()
        elif key == 'p':
            if self.is_flying:
                self.send_rc_control(0, 0, 0, 0)
                self.land()

        elif key == 'b':
            print(self.get_battery())
        elif key == 'c':
            self.connect(False)
        elif key == 'n':
            self.stop()
        elif key == '0':
            self.emergency()
        elif key == 't':
            img = self.background_frame_read.frame
            cv2.imwrite('screenshots/hand.jpg', img)

        if self.is_flying == False:
            return

        # movement control
        self.joystick_move = True
        if key == 'i':
            self.send_rc_control(0, speed, 0, 0)
            # print("forward!!!!")
        elif key == 'k':
            self.send_rc_control(0, -1*speed, 0, 0)
            # print("backward!!!!")
        elif key == 'j':
            self.send_rc_control(-1*speed, 0, 0, 0)
            # print("left!!!!")
        elif key == 'l':
            self.send_rc_control(speed, 0, 0, 0)
            # print("right!!!!")
        elif key == 's':
            self.send_rc_control(0, 0, -1*speed, 0)
            # print("down!!!!")
        elif key == 'w':
            self.send_rc_control(0, 0, speed, 0)
            # print("up!!!!")
        elif key == 'a':
            self.send_rc_control(0, 0, 0, -2*speed)
            # print("counter rotate!!!!")
        elif key == 'd':
            self.send_rc_control(0, 0, 0, 2*speed)
            # print("rotate!!!!")
        else:
            self.joystick_move = False

    def on_press(self, key):
        try:
            # print(f'alphanumeric key {key.char} pressed')
            control_key = key.char
            self.joystick(control_key)
        except AttributeError:
            pass
            # print(f'special key {key} pressed')

    def on_release(self, key):
        # stop moving, hovering
        if self.joystick_move:
            self.send_rc_control(0, 0, 0, 0)
            self.joystick_move = False

        if key == keyboard.Key.esc:
            # Stop listener
            return False

    def keyboardMonitor(self):
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.setDaemon(True)
        listener.start()
        return listener

    """ tools """

    def stop(self):
        """  brake w.r.t. the speed of the drone """
        state = self.get_current_state()
        # forward, x<0; right, y<0; up, z>0
        theta = np.deg2rad(state['yaw'])
        rotate = [[np.cos(theta), -1*np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]
        vgx, vgy = state['vgx'], state['vgy']
        vec = np.dot([vgx, vgy], rotate)
        self.send_rc_control(int(50*vec[1]), int(50*vec[0]), 0, 0)
        speed = (vgx**2+vgy**2)**0.5
        time.sleep(speed/25)
        self.send_rc_control(0, 0, 0, 0)

    def state_monitor(self):
        state = self.get_current_state()
        print(f"pitch: {state['pitch']}, roll: {state['roll']}, yaw: {state['yaw']}")
        print(f"vgx: {state['vgx']}, vgy: {state['vgy']}, vgz: {state['vgz']}")
        theta = np.deg2rad(state['yaw'])
        rotate = [[np.cos(theta), -1*np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]
        vgx, vgy = state['vgx'], state['vgy']
        speed = (vgx**2+vgy**2)**0.5
        vec = np.dot([vgx, vgy], rotate)
        print(vec)
        print(speed)
        # print(f"agx: {state['agx']}, agy: {state['agy']}, agz: {state['agz']}")
        # print(f"height: {state['h']}, barometer: {state['baro']}, battery: {state['bat']}")
        print()

    """ PID control """
    @staticmethod
    def flight_control(x, y, z, yaw):
        # pure p control
        speed_limit = 50
        zspeed = (z - 455) * -0.00005  # 455 = (720*960*0.3)**0.5
        xspeed = (x - 480) * 0.15
        yspeed = (y - 360) * -0.15
        yawspeed = yaw * 30

        zspeed = max(-1*speed_limit, min(speed_limit, zspeed))
        xspeed = max(-1*speed_limit, min(speed_limit, xspeed))
        yspeed = max(-1*speed_limit, min(speed_limit, yspeed))
        yawspeed = max(-1*speed_limit, min(speed_limit, yawspeed))
        return xspeed, zspeed, yspeed, yawspeed

    def PID(self, err, errSum, prevErr, base=0):
        # err should be an array
        # [x:left/right, y:up/down, z:forward/backward, yaw]
        limit = 50  # speed limit
        errSum += err
        P = self.kp * err
        I = self.ki * errSum
        D = self.kd * (err - prevErr)
        result = P + I + D + base
        for i in range(4):
            result[i] = max(-1*limit, min(limit, result[i]))
        prevErr[:] = err
        return result, errSum, prevErr


def detect(frame):
    # dummy example
    labels, area, center_x, center_y = [], [], [], []
    return labels, area, center_x, center_y


# -------------------- sample code --------------------
if __name__ == '__main__':
    drone = Drone()
    # you should takeoff by keyboard or
    # drone.takeoff()

    # PID settings
    # [x:left/right, y:up/down, z:forward/backward, yaw]
    drone.kp = np.array([0.115, 0.115, 0.1, 25])
    drone.ki = np.array([0.001, 0.001, 0, 0])
    drone.kd = np.array([0.12, 0.12, 0, 0])

    # reset PID err params
    # you may want to reset errSum when some conditions met
    err = np.zeros((4,))
    errSum = np.zeros((4,))
    prevErr = np.zeros((4,))
    while True:
        frame = drone.background_frame_read.frame
        cv2.imshow('img', frame)
        cv2.waitKey(33)

    while True:
        frame = drone.background_frame_read.frame
        labels, area, center_x, center_y = detect(frame)
        distance = area**0.5
        # you may want to modify the (target) or (kp, ki, kd)
        # drone.kp = (distance - distance_target) * np.array([1.08, 1.2, 1, 25])
        x_target, y_target, distance_target = 480, 360, (720*960*0.4)**0.5

        if  'Virgo' in labels:
            err = np.array([center_x - x_target,
                            y_target - center_y,
                            distance - distance_target])
            result, errSum, prevErr = drone.PID(err, errSum, prevErr)
            drone.send_rc_control(*map(int, [result[0], result[2], result[1], 0]))
        else:
            drone.send_rc_control(0,0,0,0)
            # drone.stop()
        cv2.imshow('img', frame)
        cv2.waitKey(33)
