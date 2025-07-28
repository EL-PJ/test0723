# -*- coding: utf-8 -*-
"""
### MediaPipe- Hand Landmark detection by Albert Wang
"""

## numpy, mediapipe, opencv installation
## mediapipe version adjusted by Albert Wang

##pip install numpy==1.23.5

import cv2
import mediapipe as mp

##pip install --upgrade --force-reinstall mediapipe opencv-python


# 本地端選擇檔案（取代Colab上傳）
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # 不顯示主視窗
file_path = filedialog.askopenfilename(title="請選擇手部照片", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
if not file_path:
    raise Exception("未選擇檔案，程式結束。")

## MediaPipe for Hand Landmark detection by Albert Wang

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

image = cv2.imread(file_path)
if image is None:
    raise Exception(f"圖片讀取失敗：{file_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5
) as hands:
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    else:
        print("❌ 沒偵測到手部，請試試另一張圖片。")

# 用cv2.imshow顯示結果，並等待關閉
cv2.imshow("Hand Detection Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()