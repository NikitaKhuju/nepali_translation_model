import pyautogui
import time

interval = 60  # seconds → 1 minute
duration_hours = 12
total_presses = duration_hours * 60  # 12 hours × 60 minutes = 720 presses

print("Auto Spacebar Presser will start in 10 seconds...")
time.sleep(10)

for i in range(total_presses):
    pyautogui.press("space")
    print(f"Pressed Space {i + 1}")
    time.sleep(interval)

print("✅ Finished pressing space every minute for 12 hours.")
