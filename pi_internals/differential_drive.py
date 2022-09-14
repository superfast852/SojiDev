from RPi.GPIO import setmode, setup, output, input, PWM, OUT, IN, HIGH, LOW, setwarnings, BCM, cleanup
import time

setwarnings(False)
setmode(BCM)


# Added: Motor Control, Servo Control, and Ultrasonic Control


class Motor:
    a1 = 27  # Motor A, Pin 1, BCM 27, Right
    a2 = 18  # Motor A, Pin 2, BCM 18, Right
    b1 = 26  # Motor B, Pin 1, BCM 26, Left
    b2 = 21  # Motor B, Pin 2, BCM 21, Left

    def __init__(self):
        setup(17, OUT)
        setup(4, OUT)
        setup(Motor.a1, OUT)
        setup(Motor.a2, OUT)
        setup(Motor.b1, OUT)
        setup(Motor.b2, OUT)

        try:
            self.apwm = PWM(17, 1000)
            self.apwm.start(0)
            self.bpwm = PWM(4, 1000)
            self.bpwm.start(0)
        except:
            print("PWM error")
            pass

    def left_motor(self, direction, speed):
        if direction == "fwd":
            output(Motor.b1, LOW)
            output(Motor.b2, HIGH)
        elif direction == "bwd":
            output(Motor.b1, HIGH)
            output(Motor.b2, LOW)
        self.bpwm.ChangeDutyCycle(speed)

    def right_motor(self, direction, speed):
        if direction == "fwd":
            output(Motor.a2, LOW)
            output(Motor.a1, HIGH)
        elif direction == "bwd":
            output(Motor.a2, HIGH)
            output(Motor.a1, LOW)
        self.apwm.ChangeDutyCycle(speed)

    def move(self, direction, speed):
        if direction == "fwd":
            self.left_motor("fwd", speed)
            self.right_motor("fwd", speed)
        elif direction == "bwd":
            self.left_motor("bwd", speed)
            self.right_motor("bwd", speed)
        elif direction == "left":
            self.left_motor("bwd", speed)
            self.right_motor("fwd", speed)
        elif direction == "right":
            self.left_motor("fwd", speed)
            self.right_motor("bwd", speed)

    def stop(self):
        output(Motor.a1, LOW)
        output(Motor.a2, LOW)
        output(Motor.b1, LOW)
        output(Motor.b2, LOW)

    def halt(self):
        output(Motor.a1, LOW)
        output(Motor.a2, LOW)
        output(Motor.b1, LOW)
        output(Motor.b2, LOW)
        output(17, LOW)
        output(4, LOW)
        cleanup()

if __name__ == "__main__":
    motor = Motor()
    for i in range(10):
        motor.move("fwd", 100)
        time.sleep(2)
        motor.move("bwd", 100)
        time.sleep(2)
        motor.move("left", 100)
        time.sleep(2)
        motor.move("right", 100)
        time.sleep(2)