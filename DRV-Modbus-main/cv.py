import cv2  # OpenCV 庫，用於圖像處理
import numpy as np  # NumPy 庫，用於數組操作

# 讀取圖片檔案
image = cv2.imread(r"C:\DRV-Modbus-main\apple.jpg")

# 檢查圖片是否成功讀取
if image is None:
    print("Error: Image not found or path is incorrect.")
    exit()

# 將圖片從 BGR 色彩空間轉換為 HSV 色彩空間
# HSV 更適合顏色檢測，因為它分離了色相(H)、飽和度(S)和亮度(V)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定義紅色在 HSV 空間的範圍
# 由於紅色環跨越0度/180度，因此需要兩個區間
lower_hsv1 = np.array([0, 120, 70])
upper_hsv1 = np.array([10, 255, 255])
lower_hsv2 = np.array([170, 120, 70])
upper_hsv2 = np.array([180, 255, 255])

# 創建遮罩：只保留在指定 HSV 範圍內的像素（白色），其他為黑色
mask1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
mask2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
mask = mask1 + mask2

# 從遮罩中檢測輪廓
# cv2.RETR_LIST: 檢測所有輪廓，不建立層次結構
# cv2.CHAIN_APPROX_SIMPLE: 壓縮輪廓，只保存關鍵點
contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 複製原始圖片，用於繪製輪廓
output = image.copy()

# 遍歷所有檢測到的輪廓
for contour in contours:
    # 計算輪廓面積
    area = cv2.contourArea(contour)
    # 只處理面積大於 500 的輪廓，避免小噪點
    if area > 500:
        # 在輸出圖片上繪製輪廓
        # [contour]: 輪廓列表, -1: 繪製所有輪廓, (0,255,0): 綠色, 3: 線寬
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)

# 縮放比例，用於顯示
scale = 0.5

# 縮放所有圖片到 50% 大小，便於顯示
image_resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
mask_resized = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
output_resized = cv2.resize(output, (0, 0), fx=scale, fy=scale)

# 顯示結果視窗
cv2.imshow("Original Image", image_resized)  # 原始圖片
cv2.imshow("Mask", mask_resized)  # 遮罩（二值圖）
cv2.imshow("Apple Contours", output_resized)  # 輪廓檢測結果

# 等待按鍵，按任意鍵關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存結果圖片
cv2.imwrite("output_contours.jpg", output)