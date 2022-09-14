from botware import *
from multiprocessing import Process

motor = Motor()
arm = Arm()

servos = Process(target=arm.servo_ctrl).start()

while True:
    if motor.stopped == False:
        motor.move("fwd", 100)