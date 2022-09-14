from RPi.GPIO import setmode, setup, output, input, PWM, OUT, IN, HIGH, LOW, setwarnings, BCM, cleanup
import time
import multiprocessing


setwarnings(False)
setmode(BCM)


# Added: Motor Control, Servo Control, and Ultrasonic Control


class Motor:

    def __init__(self):
        self.stopped = False
        self.a1 = 27  # Motor A, Pin 1, BCM 27, Right
        self.a2 = 18  # Motor A, Pin 2, BCM 18, Right
        self.b1 = 26  # Motor B, Pin 1, BCM 26, Left
        self.b2 = 21  # Motor B, Pin 2, BCM 21, Left
        setup(17, OUT)
        setup(4, OUT)
        setup(self.a1, OUT)
        setup(self.a2, OUT)
        setup(self.b1, OUT)
        setup(self.b2, OUT)

        try:
            self.apwm = PWM(17, 1000)
            self.apwm.start(0)
            self.bpwm = PWM(4, 1000)
            self.bpwm.start(0)
        except Exception as e:
            print("PWM error: ", e)
            pass

    def left_motor(self, direction, speed):
        if direction == "fwd":
            output(self.b1, LOW)
            output(self.b2, HIGH)
        elif direction == "bwd":
            output(self.b1, HIGH)
            output(self.b2, LOW)
        self.bpwm.ChangeDutyCycle(speed)


    def right_motor(self, direction, speed):
        if direction == "fwd":
            output(self.a2, LOW)
            output(self.a1, HIGH)
        elif direction == "bwd":
            output(self.a2, HIGH)
            output(self.a1, LOW)
        self.apwm.ChangeDutyCycle(speed)

    def lm(self, speed):
        if speed > 0:
            output(self.b1, LOW)
            output(self.b2, HIGH)
        else:
            output(self.b1, HIGH)
            output(self.b2, LOW)
        self.bpwm.ChangeDutyCycle(abs(speed))

    def rm(self, speed):
        if speed > 0:
            output(self.a2, LOW)
            output(self.a1, HIGH)
        else:
            output(self.a2, HIGH)
            output(self.a1, LOW)
        self.apwm.ChangeDutyCycle(abs(speed))

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
        self.stopped = True
        output(self.a1, LOW)
        output(self.a2, LOW)
        output(self.b1, LOW)
        output(self.b2, LOW)

    def halt(self):
        output(self.a1, LOW)
        output(self.a2, LOW)
        output(self.b1, LOW)
        output(self.b2, LOW)
        output(17, LOW)
        output(4, LOW)
        cleanup()


class Sensors:
    def __init__(self, p1=OUT, p2=OUT, p3=OUT, pins=0, mpu=0):
        from py_qmc5883l import QMC5883L as qmc
        if mpu:
            from mpu6050 import mpu6050
            self.mpu = mpu6050(mpu)
        self.trig = 11
        self.echo = 8
        self.Port1 = 5
        self.Port2 = 6
        self.Port3 = 13

        setup(self.trig, OUT, initial=LOW)
        setup(self.echo, IN)
        setup(self.Port1, p1)
        setup(self.Port2, p2)
        setup(self.Port3, p3)
        if pins:
            for pin in pins:
                setup(pin[0], pin[1])

        try:
            self.mag = qmc()
            self.mag.set_calibration([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        except:
            print("Magnetometer not found.")
            qmc = None
            pass

    def readport(self, port):
        try:
            if port == 1:
                return input(self.Port1)
            elif port == 2:
                return input(self.Port2)
            elif port == 3:
                return input(self.Port3)
            else:
                return 0
        except:
            print(f'Could not read port {port}! Check if it is set to input.')
            return 0

    def writeport(self, port, state=LOW):
        try:
            if port == 1:
                output(self.Port1, state)
            elif port == 2:
                output(self.Port2, state)
            elif port == 3:
                output(self.Port3, state)
        except:
            print(f'Could not write to port {port}! Check if it is set to output.')

    def read_ultrasonic(self):
        output(self.trig, HIGH)
        time.sleep(0.000015)
        output(self.trig, LOW)
        while not input(self.echo):
            pass
        t1 = time.time()
        while input(self.echo):
            pass
        t2 = time.time()
        return (t2 - t1) * 340 / 2

    def read_magnetometer(self):
        if self.mag:
            return self.mag.get_magnet()
        else:
            return None

    def read_accelerometer(self):
        try:
            return self.mpu.get_accel_data()
        except ValueError:
            print("Accelerometer not found.")
            return None
        except Exception as e:
            print("Accelerometer error.")
            return None


class Arm:
    def __init__(self, mtr, address="", port=42069):
        import socket
        from adafruit_servokit import ServoKit
        if address == "":
            address = socket.gethostbyname("soji.local")
            address = address if address != "127.0.0.1" else "192.168.0.101"

        self.servo_angle = [90, 90, 90]
        pca = ServoKit(channels=8)  # Servo controller
        self.x = pca.servo[0]
        self.y1 = pca.servo[1]
        self.y2 = pca.servo[2]
        self.motor = mtr
        self.x.angle, self.y1.angle, self.y2.angle = self.servo_angle
        print(self.servo_angle)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Init socket
        try:
            print("Binding to {}:{}...".format(address, port))
            self.server_socket.bind((address, port))  # Open Socket in that address
        except Exception as e:
            print("Bind Failed! Using Backup Port 9160")
            self.server_socket.bind((address, 9160))
        print(f"Socket Opened at {address}:{port}")
        multiprocessing.Process(target=self.servo_ctrl, args=(self.motor,)).start()


    def servo_ctrl(self, motor):
        try:
            self.server_socket.listen(1)  # Listen for new clients
            print("Listening for client . . .")
        except:
            self.server_socket.close()

        while True:
            try:
                conn, addr = self.server_socket.accept()  # Start communication if a client is found
                print("Connected to client at ", addr)

                while conn is not None:
                    output = conn.recv(8)  # Wait and receive a new packet
                    conn.send(b"OK")  # Send OK to client

                    if output.strip() == b"disconnect":  # If the packet is a disconnect.
                        print(str(self.address) + " Disconnected.")
                        conn.close()  # Close connection with client
                        conn = None  # Disable loop in next pass
                        pass

                    elif output:
                        b = output.decode("utf-8").split(" ")[:2]  # decode and split into list the received adjustments
                        try:
                            a = [b[0][0], b[1][0]]
                        except IndexError:
                            try:
                                a = [b[0][0], ""]
                            except IndexError:
                                try:
                                    a = ["", b[1][0]]
                                except IndexError:
                                    a = ["", ""]

                    try:
                        for axis in range(2):  # Direction to Adjustment
                            if a[axis] == '+':
                                self.servo_angle[axis] += 5
                            elif a[axis] == '-':
                                self.servo_angle[axis] -= 5


                        if a[1] != "=":
                            self.servo_angle[2] = 90 - (self.servo_angle[1] - 90)  # Broken, gotta fix

                        for i in range(3):  # Range Delimiters
                            if self.servo_angle[i] >= 180:
                                self.servo_angle[i] = 180
                            elif self.servo_angle[i] <= 0:
                                self.servo_angle[i] = 0


                        # Write angles to servos
                        self.x.angle, self.y1.angle, self.y2.angle = self.servo_angle
                        if a[0] == a[1] == "=":
                            self.grab(motor)
                            conn.recv(8)
                            conn.send(b"OK")

                    except Exception as e:
                        print(e)
                        continue

            except Exception as e:
                print("Client Disconnected.")
                conn = None
                pass

    def grab(self):
        motors.stop()
        self.servo_angle = [90, 90, 90]
        self.x.angle, self.y1.angle, self.y2.angle = self.servo_angle
        print("Grabbing...")
        time.sleep(2)
        print("Grabbed!")
        motors.stopped = False
        return 0


if __name__ == "__main__":
    motor = Motor()
    arm = Arm(motor)
    while motor.stopped != True:
        motor.move("fwd", 50)
