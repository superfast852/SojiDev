import Adafruit_ADS1x15
adc = Adafruit_ADS1x15.ADS1115()
print(adc.read_adc(4, gain=1,address = 0x48))
