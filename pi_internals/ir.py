

import hat
import RPi.GPIO as io
import RPi
from time import sleep
dir(RPi)

io.setmode(io.BCM)
io.setup(5, io.IN, pull_up_down = io.PUD_UP)
motor = hat.Motor()

while True:
    motor.move("fwd", 100)
    if not io.input(5):
        sleep(0.1)
        if not io.input(5):
            motor.move("left", 100)
            sleep(0.2)

