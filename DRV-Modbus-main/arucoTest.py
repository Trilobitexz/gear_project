from drv_modbus import send
from drv_modbus import request
from landmark import aruco
from realsense import realsense
from pymodbus.client import ModbusTcpClient
from ultralytics import YOLO 
import numpy as np
import cv2
import time

# =====================================================================
# 🚀 雙 YOLO 模型載入區 (雙專家大腦就緒！)
# =====================================================================
print("正在載入 YOLO 偵測模型 (用於高處定位)...")
# ✅ 負責在高處尋找齒輪位置的偵測模型
detect_model = YOLO(r"C:\我的\大學\大二下\自主專題_智慧機器人_機器人感測與周邊整合\gear.v1i.yolov11\runs\detect\yolov11_results\gear_experiment\weights\best.pt") 

print("正在載入 YOLO 分割模型 (用於近距離計算齒數)...")
# ✅ 負責在近距離取得齒輪輪廓的分割模型
seg_model = YOLO(r"C:\我的\大學\大二下\自主專題_智慧機器人_機器人感測與周邊整合\tooth-2.v1i.yolov11\runs\segment\gear_seg_v1\weights\best.pt") 
print("🎉 雙模型載入完成！")
# =====================================================================

# =====================================================================
# ⚙️ 影像處理、攤平與 FFT 頻譜分析輔助函數
# =====================================================================

# 🌟 影像前處理模組 (去除陰影、強化對比)
def enhance_contrast_and_denoise(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # 使用 CLAHE 限制對比度自適應直方圖均衡化，完美消除局部陰影
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)

def unwrap_gear(image: np.ndarray, center: tuple, max_radius: float) -> np.ndarray:
    safe_center = (float(center[0]), float(center[1]))
    circumference = int(2 * np.pi * max_radius)
    radial_res = int(max_radius)
    
    unwrapped = cv2.warpPolar(
        image, 
        (radial_res, circumference), 
        safe_center,  
        max_radius, 
        cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR
    )
    unwrapped = cv2.rotate(unwrapped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return unwrapped

def map_to_unwrapped_coords(point, center, max_radius):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    rho = np.hypot(dx, dy)
    
    theta = np.arctan2(dy, dx)
    if theta < 0:
        theta += 2 * np.pi

    circumference = 2 * np.pi * max_radius
    mapped_x = int(theta * (circumference / (2 * np.pi)))
    mapped_y = int(max_radius - rho) 
    return mapped_x, mapped_y

def fourier_tooth_analysis(unwrapped_mask: np.ndarray, w=960, h=700):
    signal_raw = np.sum(unwrapped_mask > 128, axis=0).astype(float)
    signal = signal_raw - np.mean(signal_raw)
    fft_mags = np.abs(np.fft.rfft(signal))
    fft_mags[:5] = 0 
    fft_teeth = int(np.argmax(fft_mags))
    
    mean_mag = np.mean(fft_mags[5:]) 
    max_mag = fft_mags[fft_teeth]
    snr = max_mag / mean_mag if mean_mag > 0 else 0
    
    # 重新配置圖表與面板寬度
    chart_w = 660
    panel_x = chart_w
    spectrum_img = np.ones((h, w, 3), dtype=np.uint8) * 20 
    
    # --- 區域 A：1D 齒輪拉平波形圖 ---
    wave_h = 240
    cv2.rectangle(spectrum_img, (0, 0), (chart_w, wave_h), (30, 30, 35), -1)
    
    sig_min, sig_max = np.min(signal_raw), np.max(signal_raw)
    if sig_max > sig_min:
        pts = []
        for x in range(chart_w):
            idx = int((x / chart_w) * len(signal_raw))
            idx = min(idx, len(signal_raw) - 1)
            norm_val = (signal_raw[idx] - sig_min) / (sig_max - sig_min)
            draw_y = wave_h - 20 - int(norm_val * (wave_h - 60))
            pts.append([x, draw_y])
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(spectrum_img, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
    cv2.line(spectrum_img, (0, wave_h), (chart_w, wave_h), (100, 100, 100), 2)

    # --- 區域 B：FFT 頻譜直方圖 ---
    for y in range(wave_h + 40, h, 40):
        cv2.line(spectrum_img, (0, y), (chart_w, y), (50, 50, 50), 1)
    for x in range(0, chart_w, 40):
        cv2.line(spectrum_img, (x, wave_h), (x, h), (50, 50, 50), 1)

    max_mag_disp = np.max(fft_mags) if np.max(fft_mags) > 0 else 1
    num_bins = min(100, len(fft_mags))
    bin_w = chart_w / num_bins
    
    avg_y = h - 30 - int((mean_mag / max_mag_disp) * (h - wave_h - 80))
    cv2.line(spectrum_img, (0, avg_y), (chart_w, avg_y), (0, 0, 200), 2)
    cv2.putText(spectrum_img, "Avg Noise", (chart_w - 90, avg_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    for i in range(num_bins):
        bar_h = int((fft_mags[i] / max_mag_disp) * (h - wave_h - 80))
        x1 = int(i * bin_w)
        x2 = max(x1 + 1, int((i + 1) * bin_w) - 2) 
        y1 = h - 30 - bar_h
        y2 = h - 30
        
        if i == fft_teeth:
            cv2.rectangle(spectrum_img, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.putText(spectrum_img, f"Peak:{i}", (x1 - 25, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.rectangle(spectrum_img, (x1, y1), (x2, y2), (255, 180, 0), -1) 
        
        if i % 10 == 0:
            cv2.putText(spectrum_img, str(i), (x1, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 圖表標題
    cv2.rectangle(spectrum_img, (0, 0), (chart_w, 40), (0, 0, 0), -1)
    cv2.putText(spectrum_img, "2. FFT FREQUENCY SPECTRUM", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- 區域 C：右側診斷面板 ---
    cv2.rectangle(spectrum_img, (panel_x, 0), (w, h), (15, 15, 15), -1) 
    cv2.line(spectrum_img, (panel_x, 0), (panel_x, h), (100, 100, 100), 2) 
    
    p_margin = panel_x + 20
    cv2.putText(spectrum_img, "FFT DIAGNOSTICS", (p_margin, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(spectrum_img, (p_margin, 60), (w - 20, 60), (100, 100, 100), 1)
    
    cv2.putText(spectrum_img, "Detected Cycles:", (p_margin, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(spectrum_img, f"{fft_teeth} Teeth", (p_margin, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    
    cv2.putText(spectrum_img, "Signal-to-Noise (SNR):", (p_margin, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    snr_color = (0, 255, 0) if snr > 4.0 else (0, 165, 255) 
    cv2.putText(spectrum_img, f"{snr:.1f}x", (p_margin, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.2, snr_color, 2)
    
    status = "PASS (High Confidence)" if snr > 4.0 else "WARNING (Noisy Signal)"
    cv2.putText(spectrum_img, f"Status: {status}", (p_margin, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, snr_color, 2)
    
    return spectrum_img, fft_teeth

# ✅ 進階幾何計算齒數 (已升級：橢圓擬合增強版)
def count_teeth_from_mask(mask_array: np.ndarray, annotated_img: np.ndarray):
    mask_uint8 = (mask_array * 255).astype(np.uint8)
    
    kernel = np.ones((5, 5), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, None, 0, None, 0.0, [], mask_uint8
        
    c = max(contours, key=cv2.contourArea)
    
    # ---------------------------------------------------------
    # ✨ 升級：使用「邊緣橢圓擬合」取代重心與外接矩形
    # ---------------------------------------------------------
    if len(c) >= 5:
        ellipse = cv2.fitEllipse(c)
        center_x, center_y = int(ellipse[0][0]), int(ellipse[0][1])
        gear_diameter = max(ellipse[1][0], ellipse[1][1]) 
    else:
        (x, y), radius = cv2.minEnclosingCircle(c)
        center_x, center_y = int(x), int(y)
        gear_diameter = radius * 2
        
    center = np.array([center_x, center_y])
    # ---------------------------------------------------------
    
    orange = (0, 165, 255)
    cv2.line(annotated_img, (center_x - 30, center_y), (center_x + 30, center_y), orange, 3)
    cv2.line(annotated_img, (center_x, center_y - 30), (center_x, center_y + 30), orange, 3)
    cv2.circle(annotated_img, (center_x, center_y), 5, orange, -1) 
    
    c_points = c.reshape(-1, 2)
    distances_to_center = np.linalg.norm(c_points - center, axis=1)
    peak_idx = int(np.argmax(distances_to_center))
    c = np.roll(c, -peak_idx, axis=0) 

    min_depth_threshold = gear_diameter * 0.010 
    min_width_threshold = gear_diameter * 0.015
    merge_dist_threshold = gear_diameter * 0.04 
    
    hull = cv2.convexHull(c, returnPoints=False)
    defects = cv2.convexityDefects(c, hull)
    
    if defects is None:
        return 0, center, gear_diameter, None, 0.0, [], mask_uint8

    raw_valleys = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        far_point = np.array(c[f][0]) 
        defect_width = np.linalg.norm(np.array(c[s][0]) - np.array(c[e][0]))
        if depth > min_depth_threshold and defect_width > min_width_threshold: 
            raw_valleys.append(far_point)

    def merge_points(points, threshold):
        merged = []
        for p in points:
            found_close = False
            for i, m in enumerate(merged):
                if np.linalg.norm(p - m) < threshold:
                    merged[i] = (m + p) / 2.0 
                    found_close = True
                    break
            if not found_close:
                merged.append(p)
        return [np.array(m, dtype=int) for m in merged]

    merged_valleys = merge_points(raw_valleys, merge_dist_threshold)

    final_valleys = []
    if len(merged_valleys) > 0:
        distances = [np.linalg.norm(v - center) for v in merged_valleys]
        median_radius = np.median(distances) 
        radius_tolerance = median_radius * 0.20 
        for v, dist in zip(merged_valleys, distances):
            if abs(dist - median_radius) < radius_tolerance:
                final_valleys.append(v)
            else:
                cv2.circle(annotated_img, tuple(v), 8, (128, 128, 128), -1) 
                
    valleys_with_angles = []
    for v in final_valleys:
        dx = v[0] - center_x
        dy = v[1] - center_y
        angle_rad = np.arctan2(-dy, dx)
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360.0 
        valleys_with_angles.append((v, angle_deg))
        
        cv2.line(annotated_img, (center_x, center_y), tuple(v), (255, 180, 0), 4) 
        cv2.circle(annotated_img, tuple(v), 12, (0, 0, 255), -1)                  

    first_tooth_pt = None
    first_angle_deg = 0.0
    if len(valleys_with_angles) >= 2:
        valleys_with_angles.sort(key=lambda x: x[1]) 
        v_min = valleys_with_angles[0][0]  
        v_max = valleys_with_angles[-1][0] 
        
        cv2.line(annotated_img, (center_x, center_y), tuple(v_min), (0, 255, 0), 10) 
        cv2.line(annotated_img, (center_x, center_y), tuple(v_max), (0, 255, 0), 10) 
        
        avg_x = int((v_min[0] + v_max[0]) / 2.0)
        avg_y = int((v_min[1] + v_max[1]) / 2.0)
        first_tooth_pt = (avg_x, avg_y)
        
        ft_dx = avg_x - center_x
        ft_dy = avg_y - center_y
        first_angle_rad = np.arctan2(-ft_dy, ft_dx)
        first_angle_deg = np.degrees(first_angle_rad)
        if first_angle_deg < 0:
            first_angle_deg += 360.0
            
        cv2.line(annotated_img, (center_x, center_y), first_tooth_pt, (0, 255, 255), 10)
        cv2.circle(annotated_img, first_tooth_pt, 16, (0, 255, 255), -1)

    return len(final_valleys), center, gear_diameter, first_tooth_pt, first_angle_deg, final_valleys, mask_uint8

# 影像視角校正轉換
def Warp(frame, c_center_list, width, height):
    p1 = np.float32(c_center_list)
    p2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    m = cv2.getPerspectiveTransform(p1,p2)
    output = cv2.warpPerspective(frame, m, (width, height))
    return output, m

# =====================================================================
# 🤖 機器手臂與核心視覺流程
# =====================================================================
# 📍 第一階段：辨識齒輪位置 (支援多齒輪，並排除重疊)
def Find_Gear_Object():
    frame = realsense.Get_RGB_Frame()
    
    if frame is None or not isinstance(frame, np.ndarray):
        print("⚠️ 警告：無法從相機取得畫面 (高處定位)")
        return [], []
        
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ret, T_cam_to_aruco_result, T_aruco_to_cam_result, id_result, corner_result = aruco.Detect_Aruco(
                                                                                                    frame, 
                                                                                                    K, 
                                                                                                    D, 
                                                                                                    aruco_length, 
                                                                                                    aruco_5x5_100_id.aruco_dict, 
                                                                                                    aruco_5x5_100_id.aruco_params)
    
    if len(id_result) < 4:
        cv2.imshow("Step 1: High View Lock", frame) 
        return [], []

    if len(id_result) == 4:
        c_center_list = [(0, 0), (0, 0), (0, 0), (0, 0)]
        for id, c in zip(id_result, corner_result):
            c_center = aruco.Corners_Center(c)
            c_center = (int(c_center[0]), int(c_center[1]))
            c_center_list[id - 1] = c_center
        
        output, m = Warp(frame, c_center_list, int(real_width), int(real_height))
        
        # 🚀 保持原圖輸入，依靠 gear best.pt 原生實力
        results = detect_model(output, conf=0.45, verbose=False)
        boxes = results[0].boxes
        
        real_points = []
        filtered_centers = [] 
        min_dist_threshold = 40 

        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # ==========================================
                # 🎯 幾何修正方案：利用「齒輪為正圓」特性抵抗陰影
                # ==========================================
                w = x2 - x1
                h = y2 - y1
                
                # 因為照片中陰影在右側，導致框的寬度被拉長 (w > h)
                # 左、上、下的邊緣是準確的。所以我們取高度 h 作為齒輪的「真實直徑」
                true_diameter = min(w, h)
                
                # 以左上角 (x1, y1) 為錨點，加上真實半徑來推算完美中心
                cx = int(x1 + true_diameter / 2)
                cy = int(y1 + true_diameter / 2)
                # ==========================================

                is_duplicate = False
                for (fcx, fcy) in filtered_centers:
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array([fcx, fcy]))
                    if dist < min_dist_threshold:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue 
                
                filtered_centers.append((cx, cy))

                offset_x = 0.0  
                offset_y = 0.0  
                
                # 畫出原始的 YOLO 綠色框 (讓你看看陰影在哪)
                cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 畫出「幾何修正後」的完美正方形框 (黃色虛線框，選用)
                # cv2.rectangle(output, (int(x1), int(y1)), (int(x1+true_diameter), int(y1+true_diameter)), (0, 255, 255), 1, cv2.LINE_AA)
                
                # 畫出修正後的中心點紅十字
                cv2.line(output, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
                cv2.line(output, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
                cv2.circle(output, (cx, cy), 4, (0, 255, 255), -1)            
                
                final_x = cy + o_point[1] + offset_x
                final_y = cx + o_point[0] + offset_y
                real_points.append((final_x, final_y))
                
                # 在影像上標示手臂座標
                text_coord = f"X:{final_x:.1f}, Y:{final_y:.1f}"
                cv2.putText(output, text_coord, (cx + 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                print(f"🎯 找到第 {len(filtered_centers)} 個齒輪！ 影像中心:({cx}, {cy}) -> 輸出座標: X={final_x:.2f}, Y={final_y:.2f}")
            
            print(f"👀 高處視角發現並過濾後，共確認了 {len(real_points)} 個有效齒輪！")

        # 畫 X 軸 (紅色) 往右
        cv2.arrowedLine(output, (30, 30), (150, 30), (0, 0, 255), 3, tipLength=0.1)
        cv2.putText(output, "X", (160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 畫 Y 軸 (綠色) 往下
        cv2.arrowedLine(output, (30, 30), (30, 150), (0, 255, 0), 3, tipLength=0.1)
        cv2.putText(output, "Y", (35, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Step 1: High View Lock", output)
        return output, real_points     
    return [], []

# 🚁 第二階段與第三階段：下降鏡頭並計算齒數，最後夾取 (支援多目標依序夾取)
def gripper_Behave(real_points):
    for idx, pt in enumerate(real_points):
        target_x = pt[0]
        target_y = pt[1]
        
        print(f"⬇️ 準備下降鏡頭處理第 {idx+1} 個齒輪...")
        
        # 手臂移動到檢查高度 (鏡頭下降)
        send.Go_Position(c, target_x, target_y-33.935, z_height_check_number_of_teeth, home[3], home[4], -11.45, 50)
        
        # 等待手臂穩定與相機曝光適應
        time.sleep(1.5) 
        
        # 清除相機舊的暫存幀
        for _ in range(10):
            _ = realsense.Get_RGB_Frame()

        check_frame = realsense.Get_RGB_Frame()
        
        if check_frame is None or not isinstance(check_frame, np.ndarray):
            print(f"⚠️ 錯誤：無法取得第 {idx+1} 個齒輪的相機畫面！跳過計算。")
            continue
            
        check_frame = cv2.cvtColor(check_frame, cv2.COLOR_RGB2BGR)
        
        print("📸 拍攝近景完成，正在進行 FFT 與幾何分析...")
        results = seg_model.predict(source=check_frame, conf=0.35, verbose=False)
        
        if len(results) > 0 and results[0].masks is not None:
            annotated_frame = results[0].plot()
            mask_data = results[0].masks.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_data, (check_frame.shape[1], check_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # --- 🌟 套用 predict_gear 進階演算邏輯 ---
            teeth, gear_center, gear_dia, first_tooth_pt, first_angle, valleys, clean_mask = count_teeth_from_mask(mask_resized, annotated_frame)
            
            fft_teeth = 0
            fft_img = None
            unwrapped_img = None
            
            if gear_center is not None and gear_dia > 0:
                max_radius = (gear_dia / 2) * 1.1 
                unwrapped_mask = unwrap_gear(clean_mask, tuple(gear_center), max_radius)
                fft_img, fft_teeth = fourier_tooth_analysis(unwrapped_mask, w=960, h=700)
                
                unwrapped_img = unwrap_gear(check_frame, tuple(gear_center), max_radius)
                for v in valleys:
                    mapped_x, mapped_y = map_to_unwrapped_coords(v, tuple(gear_center), max_radius)
                    if 0 <= mapped_x < unwrapped_img.shape[1] and 0 <= mapped_y < unwrapped_img.shape[0]:
                        cv2.line(unwrapped_img, (mapped_x, 0), (mapped_x, unwrapped_img.shape[0]), (0, 255, 0), 4)
                        cv2.circle(unwrapped_img, (mapped_x, mapped_y), 10, (0, 0, 255), -1)

            # 在原圖上畫上文字標籤
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (20, 20), (1050, 400), (0, 0, 0), -1) 
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            cv2.putText(annotated_frame, f"Geometry Teeth: {teeth}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)
            cv2.putText(annotated_frame, f"FFT Verification: {fft_teeth}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
            cv2.putText(annotated_frame, f"1st Tooth Angle: {first_angle:.1f} deg", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)

            print(f"⚙️ 第 {idx+1} 個齒輪 -> 幾何計算齒數: {teeth} | FFT 分析齒數: {fft_teeth}")

            # ==========================================
            # 🏆 組合「全方位戰情儀表板」顯示給操作員看
            # ==========================================
            dashboard = np.zeros((1000, 1600, 3), dtype=np.uint8)
            
            # [板塊 1] 上方：拉平圖
            if unwrapped_img is not None:
                cv2.rectangle(dashboard, (0, 0), (1600, 40), (25, 25, 25), -1)
                cv2.putText(dashboard, "1. UNWRAPPED GEAR SPATIAL VERIFICATION (Green lines hit valleys)", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                uw_resized = cv2.resize(unwrapped_img, (1600, 260))
                dashboard[40:300, 0:1600] = uw_resized

            # [板塊 2] 左下：FFT 圖
            if fft_img is not None:
                dashboard[300:1000, 0:960] = fft_img

            # [板塊 3] 右下：齒輪標示圖
            cv2.rectangle(dashboard, (960, 300), (1600, 340), (25, 25, 25), -1)
            cv2.putText(dashboard, "3. GEOMETRY DETECTION", (975, 328), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            annot_h, annot_w = annotated_frame.shape[:2]
            scale = min(640 / annot_w, 660 / annot_h)
            new_w, new_h = int(annot_w * scale), int(annot_h * scale)
            annot_resized = cv2.resize(annotated_frame, (new_w, new_h))
            
            y_offset = 340 + (660 - new_h) // 2
            x_offset = 960 + (640 - new_w) // 2
            dashboard[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = annot_resized
            
            cv2.line(dashboard, (0, 300), (1600, 300), (100, 100, 100), 2)
            cv2.line(dashboard, (960, 300), (960, 1000), (100, 100, 100), 2)

            window_name = f"Dashboard: Gear {idx+1}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, dashboard)
            # 儀表板停留 3 秒，讓操作人員過目，之後繼續執行夾取
            cv2.waitKey(3000) 
            
        else:
            print("⚠️ 未檢測到齒輪遮罩，無法進行進階計算")

        print(f"🦾 準備夾取第 {idx+1} 個目標...")
        send.Go_Position(c, target_x, target_y, z_height_check_number_of_teeth, home[3], home[4], -133.567, 50)
        send.Go_Position(c, target_x, target_y, z_height, home[3], home[4], -133.567, 50)
        send.gripper_ON(c)
        send.Go_Position(c, target_x, target_y, home[2], home[3], home[4], -133.567, 50)
        send.Go_Position(c, drop[0], drop[1], drop[2], home[3], home[4], home[5], 50)
        send.Go_Position(c, drop[0], drop[1], drop[2]-30, home[3], home[4], home[5], 50)
        send.gripper_OFF(c)
        send.Go_Position(c, drop[0], drop[1], drop[2], home[3], home[4], home[5], 50)
        send.Go_Position(c, home[0], home[1], home[2], home[3], home[4], home[5], 50)

def main():
    send.Go_Position(c, home[0], home[1], home[2], home[3], home[4], home[5], 50)
    send.gripper_OFF(c)
    for i in range(50):
        frame = realsense.Get_RGB_Frame()
    real_points = []
    
    print("🚀 開始執行任務：搜尋高處目標...")
    while len(real_points) == 0:
        output, real_points = Find_Gear_Object()
        cv2.waitKey(1)
        
    print(f"🎯 高處尋找完畢，共找到 {len(real_points)} 個目標座標！")
    cv2.waitKey(1000)
    
    gripper_Behave(real_points)
    print("✅ 單次搜尋清空完畢!!!")

# =====================================================================
# 🔧 常數與連線設定區
# =====================================================================
aruco_5x5_100_id = aruco.Aruco(aruco.ARUCO_DICT().DICT_5X5_100, 1, 200)
aruco_length = 0.0525

o_point = (-216.897,376.422)
e_point = (75.044, 616.16)
z_height = 412
z_height_check_number_of_teeth = z_height + 25

real_width = e_point[0] - o_point[0]
real_height = e_point[1] - o_point[1]

K = realsense.Get_Color_K()
D = np.array([0.0,0.0,0.0,0.0,0.0,])

# 手臂連線設定
print("🔌 正在連線至 Modbus 機器手臂...")
c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
c.connect()
print("🟢 手臂連線成功！")

gripper_rz = -133.567
home = [386.077, -51.439, 680,  -179.161, -0.32, -102.22800000000001]
drop = [532.1080000000001, -229.305, z_height+40, -179.161, -0.32, -133.608]

# =====================================================================
# 🏃 程式執行進入點
# =====================================================================
if __name__ == "__main__":
    try:
        for i in range(3): # 測試執行 3 次全局任務
            main()
            
        print("\n" + "="*50)
        print("🏁 所有任務已順利完成！")
        print("👀 輸出的視窗已保留在螢幕上供您檢視。")
        print("👉 檢視完畢後，請點擊任一影像視窗，並按下「鍵盤任意鍵」以關閉程式...")
        print("="*50 + "\n")
        
        cv2.waitKey(0) 
        
    finally:
        c.close()
        cv2.destroyAllWindows()
        print("🔌 系統已安全關閉。")
