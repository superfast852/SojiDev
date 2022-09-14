from adafruit_servokit import ServoKit
from botware import Sensors
from numpy import mean

pca = ServoKit(channels=8)
sense = Sensors()
angles = []

for i in range(181):
    pca.servo[0].angle = i
    cache = []
    for i in range(5):
        det = sense.read_ultrasonic()
        cache.append(det) if det<50 else cache.append(0.0)
    angles.append(cache)


for i in range(len(angles)):
    angles[i].sort()
    angles[i] = mean(angles[i])

max_det = max(angles)
max_det_location = angles.index(max_det)
max_det_left = []
max_det_right = []
for i in range(6, 1, -1):
    max_det_left.append(angles[max_det_location-i])
for i in range(1, 6):
    max_det_right.append(angles[max_det_location+i])
max_det_range = angles[max_det_location-5:max_det_location+6]
print(max_det)
print(max_det_left)
print(max_det_right)

print("Max Range:")
print(max_det_range)

locations = []
for i in max_det_range:
    locations.append(angles.index(i))

print("Data Locations: ")
print(locations)
