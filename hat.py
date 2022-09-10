from RPi.GPIO import setmode, setup, output, input, PWM, OUT, IN, HIGH, LOW, setwarnings, BCM, cleanup
import socket
from adafruit_servokit import ServoKit
import pygame
from time import sleep

setwarnings(False)
setmode(BCM)
# Added: Motor Control, Servo Control, and Ultrasonic Control


class Motor:
    a1 = 27
    a2 = 18
    b1 = 26
    b2 = 21

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

    def move(self, direction, speed):
        if direction == "bwd":
            output(Motor.a1, LOW)
            output(Motor.a2, HIGH)
            output(Motor.b1, HIGH)
            output(Motor.b2, LOW)

        elif direction == "fwd":
            output(Motor.a1, HIGH)
            output(Motor.a2, LOW)
            output(Motor.b1, LOW)
            output(Motor.b2, HIGH)

        elif direction == "right":
            output(Motor.a1, LOW)
            output(Motor.a2, HIGH)
            output(Motor.b1, LOW)
            output(Motor.b2, HIGH)

        elif direction == "left":
            output(Motor.a1, HIGH)
            output(Motor.a2, LOW)
            output(Motor.b1, HIGH)
            output(Motor.b2, LOW)

        self.apwm.ChangeDutyCycle(speed)
        self.bpwm.ChangeDutyCycle(speed)
        
    def move_ctrl(self):
        pygame.init()
        while True:
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    done=True # Flag that we are done so we exit this loop
            #print("I'm a main loop")
            joystick = pygame.joystick.Joystick(0)
            joystick.init()

            x = round(joystick.get_axis(0), 4)
            y = round(joystick.get_axis(1), 4)

        
            if x > 0.1:
                self.apwm.ChangeDutyCycle(100-(x*100))

            elif x < -0.1:
                self.bpwm.ChangeDutyCycle(100)

            if y > 0.1:
                output(Motor.a1, LOW)
                output(Motor.a2, HIGH)
                output(Motor.b1, HIGH)
                output(Motor.b2, LOW)
                self.apwm.ChangeDutyCycle(y*100)
                self.bpwm.ChangeDutyCycle(y*100)

            elif y < -0.1:
                output(Motor.a1, HIGH)
                output(Motor.a2, LOW)
                output(Motor.b1, LOW)
                output(Motor.b2, HIGH)
                self.apwm.ChangeDutyCycle(-y*100)
                self.bpwm.ChangeDutyCycle(-y*100)
                
                    
            print(x, y)
            sleep(.02)
    
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


def servo_ctrl(address, port):
    servo_angle = [90, 90, 90]
    
    pca = ServoKit(channels=8)  # Servo controller
    x = pca.servo[0]
    y1 = pca.servo[1]
    y2 = pca.servo[2]
    x.angle, y1.angle, y2.angle = servo_angle
    print(servo_angle)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Init socket
    try:
        server_socket.bind((address, port))  # Open Socket in that address
    except:
        print("Using Backup Port")
        server_socket.bind((address, 9160))
# print(connect(8220, "tcp"))  # Open ngrok tunnel
    try:
        server_socket.listen(5)  # Listen for new clients
        print("Listening for client . . .")
    except:
        server_socket.close()

    while True:
        try:
            conn, addr = server_socket.accept()  # Start communication if a client is found
            print("Connected to client at ", addr)

            while conn is not None:
                output = conn.recv(16)  # Wait and receive a new packet
                
                if output.strip() == b"disconnect":  # If the packet is a disconnect.
                    print(str(address) + " Disconnected.")
                    conn.close()  # Close connection with client
                    conn = None  # Disable loop in next pass
                
                elif output:
                    a = output.decode("utf-8").split(" ")[:2]  # decode and split into a list the received adjustments
                    print(a)

                try:
                    for axis in range(2):  # Pretty self-explanatory imo
                        if a[axis][0] == '+':
                            servo_angle[axis] += 5
                        elif a[axis][0] == '-':
                            servo_angle[axis] -= 5

                    if a[1] != "=":
                        servo_angle[2] = 90 - (servo_angle[1] - 90)  # Broken, gotta fix

                    for i in range(3):  # Range Delimiters
                        if servo_angle[i] >= 180: servo_angle[i] = 180
                        elif servo_angle[i] <= 0: servo_angle[i] = 0

                    # Write angles to servos
                    x.angle, y1.angle, y2.angle = servo_angle
                    print(servo_angle)
                    if a[0][0] == a[1][0] == "=":
                        grab()
                except:
                    continue
        except OSError as E:
            print("Terminating...")

def grab():
    heading = 90-90  # x.angle


def motor_test():
    import time
    motor = Motor()
    motor.move("fwd", 100)
    time.sleep(1)
    motor.move("bwd", 100)
    time.sleep(1)
    motor.move("left", 100)
    time.sleep(1)
    motor.move("right", 100)
    time.sleep(1)
    motor.halt()
    print("Done")
    return "Done"

if __name__ == "__main__":
    import multiprocessing
    mtr = multiprocessing.Process(target=motor_test)
    srvo = multiprocessing.Process(target=servo_ctrl, args=('192.168.0.101', 42069))
    try:
        mtr.start()
        srvo.start()
        srvo.join()
        mtr.join()
    except KeyboardInterrupt:
        print("hi!")
        motor = Motor()
        srvo.terminate()
        mtr.terminate()
