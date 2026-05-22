from drv_modbus import send
from drv_modbus import request
from landmark import aruco
from realsense import realsense
from pymodbus.client import ModbusTcpClient
import numpy as np
import cv2
import time

home = [386.077, -51.439, 680, -180, 0, -102.22800000000001]
d =  80;

def main():
    send.Go_Position(c,home[0], home[1], home[2], home[3], home[4], home[5])
    for theta in range(0, 360):
        send.Go_Position(c, home[0]-d*np.sin(np.radians(theta)), home[1]+d*np.cos(np.radians(theta)), home[2], home[3], home[4], home[5]+theta)

c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
c.connect()
