import pyautogui
import time

print("Moving in 3 seconds...")
time.sleep(3)

pyautogui.moveTo(100, 100, duration=1)

print("Done")
