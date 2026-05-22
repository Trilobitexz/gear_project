from drv_modbus import send
from drv_modbus import request
from landmark import aruco
from realsense import realsense
from pymodbus.client import ModbusTcpClient
import numpy as np
import cv2
import time

# =====================================================================
# ⚙️ 系統與 ArUco 參數設定區
# =====================================================================
# 四個 ArUco 在世界座標中的 X/Y (請確認這四個點的 ID 分別是 1, 2, 3, 4，或是 0, 1, 2, 3)
ARUCO_WORLD_POINTS = {
    1: (374.864, -215.266), # ⚠️ 請確認你的真實貼紙 ID，依據情況修改這裡的 key (1~4)
    2: (374.864, 74.246),
    3: (622.865, -215.266),
    4: (622.865, 74.246),
}

# 物件世界座標 Z 固定值 (桌面高度)
WORLD_Z = 412

# 機器手臂初始高空位置 (Home)
HOME_POSITION = [386.077, -51.439, 680, -179.161, -0.32, -102.228]

# 🎯 專案專屬的 ArUco 與 RealSense 內參設定
aruco_5x5_100_id = aruco.Aruco(aruco.ARUCO_DICT().DICT_5X5_100, 1, 200)
aruco_length = 0.0525
K = realsense.Get_Color_K()
D = np.array([0.0,0.0,0.0,0.0,0.0,])

# =====================================================================
# 📸 相機與視覺處理函數
# =====================================================================
def take_photo():
    """
    使用 RealSense 相機，清除移動殘影後，拍攝並回傳一張清晰照片
    """
    try:
        print("🧹 清除相機緩衝區殘影...")
        # 連續抓取幾張並丟棄，確保拿到手臂停穩後的最新的畫面
        for _ in range(15):
            _ = realsense.Get_RGB_Frame()
            
        time.sleep(0.5) # 給予一點曝光適應時間
        
        print("📸 拍攝高空照片...")
        frame = realsense.Get_RGB_Frame()
        
        if frame is None or not isinstance(frame, np.ndarray):
            raise RuntimeError("拍照失敗：無法從 RealSense 取得畫面")
        
        # 將 RealSense 的 RGB 格式轉為 OpenCV 習慣的 BGR 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 顯示拍到的照片供你確認有沒有對準
        cv2.imshow("Captured Frame", frame)
        cv2.waitKey(1000) # 顯示 1 秒
        cv2.destroyWindow("Captured Frame")
        
        return frame
        
    except Exception as e:
        raise RuntimeError(f"相機運作發生錯誤: {e}")

def detect_red_squares(image):
    """
    檢測圖像中的紅色正方形
    """
    # 轉換到HSV色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定義紅色範圍
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # 創建遮罩
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # 形態學操作去除噪點
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 檢測輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    red_squares = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # 過濾小面積
            continue
        
        # 計算周長
        perimeter = cv2.arcLength(contour, True)
        
        # 多邊形逼近
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        # 檢查是否為四邊形
        if len(approx) == 4:
            # 檢查是否為正方形（寬高比接近1）
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.8 < aspect_ratio < 1.2:
                # 計算中心點
                center_x = x + w / 2
                center_y = y + h / 2
                red_squares.append((center_x, center_y, w, h))
    
    return red_squares

def _extract_marker_id_and_center(marker):
    if isinstance(marker, tuple) and len(marker) == 2:
        return marker[0], marker[1]
    return None, None

def build_homography(aruco_result):
    image_points = []
    world_points = []

    # 按 marker_id 排序以確保一致性
    sorted_aruco = sorted(aruco_result, key=lambda x: x[0])

    for marker in sorted_aruco:
        marker_id, center = _extract_marker_id_and_center(marker)
        if marker_id not in ARUCO_WORLD_POINTS:
            continue

        wx, wy = ARUCO_WORLD_POINTS[marker_id]
        if wx is None or wy is None:
            continue
        if center is None:
            continue

        u, v = center
        image_points.append([float(u), float(v)])
        world_points.append([float(wx), float(wy)])

    # 檢查是否蒐集到 4 個點來建立矩陣
    if len(image_points) < 4:
        raise RuntimeError(f"只偵測到 {len(image_points)} 個有效 ArUco 標記，需要 4 個才能計算！")

    image_points = np.array(image_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)
    H, _ = cv2.findHomography(image_points, world_points)
    if H is None:
        raise RuntimeError("無法建立影像到世界座標轉換矩陣")
    return H

def pixel_to_world(H, pixel_xy):
    pt = np.array([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float32)
    world_xy = cv2.perspectiveTransform(pt, H)[0][0]
    return float(world_xy[0]), float(world_xy[1]), WORLD_Z

# =====================================================================
# 🤖 主程式控制流程
# =====================================================================
def main():
    print("🔌 正在連線至 Modbus 機器手臂...")
    c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
    
    if not c.connect():
        print("❌ 手臂連線失敗，請檢查網路設定。")
        return
    print("🟢 手臂連線成功！")

    try:
        # 1️⃣ 控制手臂移動至高處初始位置 (Home)
        print("🏠 正在移動至高處初始位置...")
        send.Go_Position(c, HOME_POSITION[0], HOME_POSITION[1], HOME_POSITION[2], 
                            HOME_POSITION[3], HOME_POSITION[4], HOME_POSITION[5], 50)
        
        print("⏳ 等待手臂停穩...")
        time.sleep(3.0)

        # 2️⃣ 📸 使用 RealSense 鏡頭拍照！
        frame = take_photo()

        # 3️⃣ 檢測紅色正方形
        print("🔴 檢測紅色正方形...")
        red_squares = detect_red_squares(frame)
        print(f"🔺 找到紅色正方形數量: {len(red_squares)}")
        
        # 匯出找到的紅色正方形位置到文件
        if red_squares:
            with open("red_squares_positions.txt", "w") as f:
                f.write("Center_X, Center_Y, Width, Height\n")
                for cx, cy, w, h in red_squares:
                    f.write(f"{cx:.2f}, {cy:.2f}, {w:.2f}, {h:.2f}\n")
            print("📄 紅色正方形位置已匯出到 red_squares_positions.txt")
        else:
            print("⚠️ 未找到紅色正方形")

        # 4️⃣ 尋找 ArUco 並建立投影矩陣
        print("🔍 尋找 ArUco 標籤並建立投影矩陣...")
        aruco_result = detect_aruco_markers(frame)
        print(f"👀 找到標籤數量: {len(aruco_result)}")
        
        # 匯出找到的 ArUco 位置到文件
        if aruco_result:
            with open("aruco_positions.txt", "w") as f:
                f.write("ID, Pixel_X, Pixel_Y\n")
                for marker_id, (x, y) in aruco_result:
                    f.write(f"{marker_id}, {x:.2f}, {y:.2f}\n")
            print("📄 ArUco 位置已匯出到 aruco_positions.txt")
        
        H = build_homography(aruco_result)
        print("✅ 矩陣建立成功！")

        # 4️⃣ 座標轉換測試
        # 假設你想測試畫面上某個特定點 (例如像素 x=320, y=240) 的世界座標
        pixel_xy = (320, 240) 
        world_x, world_y, world_z = pixel_to_world(H, pixel_xy)

        print("\n🎯 轉換結果 (Object world position):")
        print(f"X = {world_x:.3f}")
        print(f"Y = {world_y:.3f}")
        print(f"Z = {world_z:.3f}")

    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        
    finally:
        c.close()
        print("\n🔌 系統已安全關閉。")

if __name__ == "__main__":
    main()