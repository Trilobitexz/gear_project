import cv2
import numpy as np
import mediapipe as mp

def main():
    # 初始化 MediaPipe Hands 模組
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # 自訂繪圖樣式：綠色關節點 (0, 255, 0)、藍色連接線 (255, 0, 0)
    landmark_style = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=5)
    connection_style = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)

    # 定義五根手指的指尖節點 ID (大拇指、食指、中指、無名指、小拇指)
    tip_ids = [4, 8, 12, 16, 20]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟相機")
        return

    # 設定手部偵測模型參數
    with mp_hands.Hands(
        min_detection_confidence=0.7,  # 偵測嚴格度 (調高可以減少誤判)
        min_tracking_confidence=0.5,   # 追蹤嚴格度
        max_num_hands=2                # 最多偵測兩隻手
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 水平翻轉影像（像照鏡子），並將 BGR 轉換為 RGB (MediaPipe 使用 RGB)
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 將影像傳入模型進行辨識
            results = hands.process(frame_rgb)

            total_fingers = 0
            message = "無手掌"

            # 如果畫面上偵測到手
            if results.multi_hand_landmarks:
                # 遍歷畫面中的每一隻手
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    
                    # 直接在原始彩色 frame 上畫出骨架與關節點 (套用自訂的綠點藍線)
                    mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        landmark_style,
                        connection_style
                    )

                    # 取得該手是左手還是右手 (MediaPipe 的 Label)
                    handedness = results.multi_handedness[hand_idx].classification[0].label

                    # 提取所有 21 個節點的座標
                    lm_list = []
                    h, w, c = frame.shape
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([id, cx, cy])

                    if len(lm_list) != 0:
                        fingers = []

                        # 1. 判斷大拇指 (Thumb)
                        # 因為畫面翻轉過，實際的左手在畫面左側 (標籤會顯示為 'Right')
                        if handedness == 'Right': 
                            # 比較大拇指指尖(4)與關節(3)的 X 座標
                            if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
                                fingers.append(1)
                            else:
                                fingers.append(0)
                        else: # 實際的右手
                            if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
                                fingers.append(1)
                            else:
                                fingers.append(0)

                        # 2. 判斷其他四根手指 (食指、中指、無名指、小拇指)
                        for id in range(1, 5):
                            # 若指尖的 Y 座標小於(高於)下面第二個關節的 Y 座標，代表手指伸直
                            if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                                fingers.append(1)
                            else:
                                fingers.append(0)

                        # 將這隻手伸直的手指數加到總數中
                        total_fingers += fingers.count(1)

            # 決定要在畫面上顯示的文字
            if results.multi_hand_landmarks:
                if total_fingers == 0:
                    message = "握拳 (0)"
                else:
                    message = f"手指總數: {total_fingers}"
            
            # 創建黑色背景，只畫手式線條
            display = np.zeros_like(frame)
            
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_draw.draw_landmarks(
                        display, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        landmark_style,
                        connection_style
                    )

            # 把文字寫在黑色背景上 (白色文字)
            cv2.putText(display, message, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # 顯示黑色背景視窗 (只有手式線條)
            cv2.imshow("Hand Gesture Lines Only", display)

            # 按下 ESC 鍵 (ASCII 27) 退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
