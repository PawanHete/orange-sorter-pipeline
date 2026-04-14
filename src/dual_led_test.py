import RPi.GPIO as GPIO
import time

GREEN_LED = 27  # Physical Pin 13
RED_LED = 17    # Physical Pin 11

GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN_LED, GPIO.OUT)
GPIO.setup(RED_LED, GPIO.OUT)

print("Testing LEDs on GPIO17 and GPIO27...")

try:
    while True:
        # Green ON, Red OFF
        GPIO.output(GREEN_LED, GPIO.HIGH)
        GPIO.output(RED_LED, GPIO.LOW)
        print("GREEN ON, RED OFF")
        time.sleep(1)

        # Green OFF, Red ON
        GPIO.output(GREEN_LED, GPIO.LOW)
        GPIO.output(RED_LED, GPIO.HIGH)
        print("GREEN OFF, RED ON")
        time.sleep(1)

except KeyboardInterrupt:
    print("Cleaning up GPIO...")
    GPIO.cleanup()
