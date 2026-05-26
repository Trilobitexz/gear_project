from drv_modbus import send
from drv_modbus import request
from landmark import aruco
from realsense import realsense
from pymodbus.client import ModbusTcpClient
from ultralytics import YOLO 
import numpy as np
import cv2
import time
import math  # 導入 math 模組以供面積、半徑等幾何運算使用

# =====================================================================
# 🚀 雙 YOLO 模型載入區 (雙專家大腦就緒！)
# =====================================================================
print("正在載入 YOLO 偵測模型 (用於高處定位)...")
detect_model = YOLO(r"C:\我的\大學\大二下\自主專題_智慧機器人_機器人感測與周邊整合\gear.v1i.yolov11\runs\detect\yolov11_results\gear_experiment\weights\best.pt") 

print("正在載入 YOLO 分割模型 (用於近距離計算齒數)...")
seg_model = YOLO(r"C:\我的\大學\大二下\自主專題_智慧機器人_機器人感測與周邊整合\tooth-2.v1i.yolov11\runs\segment\gear_seg_v1\weights\best.pt") 
print("🎉 雙模型載入完成！")
# =====================================================================

# =====================================================================
# 🖥️ 智慧視窗顯示引擎 (保證視窗不爆框)
# =====================================================================
def show_smart_window(title, image, max_width=1400, max_height=800):
    """
    智慧調整視窗大小：如果影像大於設定的螢幕最大範圍，就等比例縮小，
    確保在任何筆電或螢幕上都能完整顯示。
    """
    if image is None or image.size == 0:
        return
        
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    
    if scale < 1.0: 
        new_w, new_h = int(w * scale), int(h * scale)
        display_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        display_img = image.copy()
        
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, display_img)

# =====================================================================
# ⚙️ 影像處理、攤平與 FFT 頻譜分析輔助函數
# =====================================================================

def enhance_contrast_and_denoise(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
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
    
    chart_w = 660
    panel_x = chart_w
    spectrum_img = np.ones((h, w, 3), dtype=np.uint8) * 20 
    
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

    cv2.rectangle(spectrum_img, (0, 0), (chart_w, 40), (0, 0, 0), -1)
    cv2.putText(spectrum_img, "2. FFT FREQUENCY SPECTRUM", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

def count_teeth_from_mask(mask_array: np.ndarray, annotated_img: np.ndarray):
    mask_uint8 = (mask_array * 255).astype(np.uint8)
    
    kernel = np.ones((5, 5), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, None, 0, None, 0.0, [], mask_uint8
        
    c = max(contours, key=cv2.contourArea)
    
    if len(c) >= 5:
        ellipse = cv2.fitEllipse(c)
        center_x, center_y = int(ellipse[0][0]), int(ellipse[0][1])
        gear_diameter = max(ellipse[1][0], ellipse[1][1]) 
    else:
        (x, y), radius = cv2.minEnclosingCircle(c)
        center_x, center_y = int(x), int(y)
        gear_diameter = radius * 2
        
    center = np.array([center_x, center_y])
    
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

def Warp(frame, c_center_list, width, height):
    p1 = np.float32(c_center_list)
    p2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    m = cv2.getPerspectiveTransform(p1,p2)
    output = cv2.warpPerspective(frame, m, (width, height))
    return output, m

# =====================================================================
# 🤖 機器手臂與核心視覺流程
# =====================================================================

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
        # 即時搜尋中：套用智慧視窗
        show_smart_window("Step 1: High View Lock (Live)", frame) 
        return [], []

    if len(id_result) == 4:
        c_center_list = [(0, 0), (0, 0), (0, 0), (0, 0)]
        for id, c in zip(id_result, corner_result):
            c_center = aruco.Corners_Center(c)
            c_center = (int(c_center[0]), int(c_center[1]))
            c_center_list[id - 1] = c_center
        
        output, m = Warp(frame, c_center_list, int(real_width), int(real_height))
        
        results = detect_model(output, conf=0.45, verbose=False)
        boxes = results[0].boxes
        
        real_points = []
        filtered_centers = [] 
        min_dist_threshold = 40 

        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                approx_w, approx_h = x2 - x1, y2 - y1
                pad = int(min(approx_w, approx_h) * 0.2) 
                
                h_img, w_img = output.shape[:2]
                x1_roi = max(0, x1 - pad)
                y1_roi = max(0, y1 - pad)
                x2_roi = min(w_img, x2 + pad)
                y2_roi = min(h_img, y2 + pad)
                
                roi = output[y1_roi:y2_roi, x1_roi:x2_roi]
                if roi.size == 0: continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    
                    area = cv2.contourArea(c)
                    perimeter = cv2.arcLength(c, True)
                    
                    if perimeter == 0 or area < 100: 
                        continue
                        
                    circularity = 4 * math.pi * (area / (perimeter * perimeter))
                    if circularity < 0.5:
                        continue 
                    
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        c_x = int(M["m10"] / M["m00"])
                        c_y = int(M["m01"] / M["m00"])
                    else:
                        c_x, c_y = int(approx_w/2), int(approx_h/2)
                    
                    radius = math.sqrt(area / math.pi) * 1.05 
                    cx = x1_roi + c_x
                    cy = y1_roi + c_y
                else:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    radius = min(approx_w, approx_h) / 2

                is_duplicate = False
                for (fcx, fcy) in filtered_centers:
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array([fcx, fcy]))
                    if dist < min_dist_threshold:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue 
                
                filtered_centers.append((cx, cy))

                final_box_x1 = int(cx - radius)
                final_box_y1 = int(cy - radius)
                final_box_x2 = int(cx + radius)
                final_box_y2 = int(cy + radius)
                
                cv2.circle(output, (cx, cy), int(radius), (255, 105, 180), 2)
                cv2.rectangle(output, (final_box_x1, final_box_y1), (final_box_x2, final_box_y2), (0, 255, 0), 2)
                
                cv2.line(output, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
                cv2.line(output, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
                cv2.circle(output, (cx, cy), 4, (0, 255, 255), -1)            
                
                final_x = cy + o_point[1]
                final_y = cx + o_point[0]
                real_points.append((final_x, final_y))
                
                text_coord = f"X:{final_x:.1f}, Y:{final_y:.1f}"
                cv2.putText(output, text_coord, (cx + 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        cv2.arrowedLine(output, (30, 30), (150, 30), (0, 0, 255), 3, tipLength=0.1)
        cv2.putText(output, "X", (160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.arrowedLine(output, (30, 30), (30, 150), (0, 255, 0), 3, tipLength=0.1)
        cv2.putText(output, "Y", (35, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        show_smart_window("Step 1: High View Lock (Live)", output)
        return output, real_points     
    return [], []

def gripper_Behave(real_points, task_idx):
    for idx, pt in enumerate(real_points):
        target_x = pt[0]
        target_y = pt[1]
        
        print(f"⬇️ 準備下降鏡頭處理第 {idx+1} 個齒輪...")
        
        send.Go_Position(c, target_x, target_y-33.935, z_height_check_number_of_teeth, home[3], home[4], -11.45, 80)
        time.sleep(1.5) 
        
        for _ in range(10):
            try:
                _ = realsense.Get_RGB_Frame()
            except RuntimeError:
                pass 
        
        try:
            check_frame = realsense.Get_RGB_Frame()
        except RuntimeError:
            print(f"⚠️ 錯誤：相機讀取逾時！跳過第 {idx+1} 個齒輪。")
            continue
        
        if check_frame is None or not isinstance(check_frame, np.ndarray):
            print(f"⚠️ 錯誤：無法取得第 {idx+1} 個齒輪的畫面！跳過計算。")
            continue
            
        check_frame = cv2.cvtColor(check_frame, cv2.COLOR_RGB2BGR)
        print("📸 拍攝近景完成，正在進行 FFT 與幾何分析...")
        results = seg_model.predict(source=check_frame, conf=0.35, verbose=False)
        
        if len(results) > 0 and results[0].masks is not None:
            img_h, img_w = check_frame.shape[:2]
            img_cx, img_cy = img_w / 2, img_h / 2
            
            # ========================================================
            # ✨ 升級：鎖定中心點目標，拒絕旁邊亂入的齒輪
            # ========================================================
            boxes = results[0].boxes.xywh.cpu().numpy()  # [中心x, 中心y, 寬, 高]
            masks = results[0].masks.data.cpu().numpy()  # N個遮罩的陣列
            
            best_idx = 0
            min_dist = float('inf')
            
            # 遍歷畫面中被偵測到的每一個齒輪
            for i, box in enumerate(boxes):
                cx, cy = box[0], box[1]
                # 計算該齒輪中心與相機畫面中心的距離
                dist = math.hypot(cx - img_cx, cy - img_cy)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            # 只抽出離畫面中心點「最近」的那個齒輪遮罩
            mask_data = masks[best_idx]
            mask_resized = cv2.resize(mask_data, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            
            # 取得畫有所有 YOLO 目標的畫面
            annotated_frame = results[0].plot()
            
            # 畫出中心十字準心與鎖定紅線，讓操作員知道選對了
            cv2.circle(annotated_frame, (int(img_cx), int(img_cy)), 8, (255, 255, 255), -1)
            cv2.putText(annotated_frame, "FOV Center", (int(img_cx)+15, int(img_cy)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            target_cx, target_cy = int(boxes[best_idx][0]), int(boxes[best_idx][1])
            cv2.line(annotated_frame, (int(img_cx), int(img_cy)), (target_cx, target_cy), (0, 0, 255), 4)
            cv2.putText(annotated_frame, "LOCKED TARGET", (target_cx-50, target_cy-30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # ========================================================

            # 將「唯一過濾出來的 Mask」丟進去算齒數
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

            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (20, 20), (1050, 400), (0, 0, 0), -1) 
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            cv2.putText(annotated_frame, f"Geometry Teeth: {teeth}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)
            cv2.putText(annotated_frame, f"FFT Verification: {fft_teeth}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
            cv2.putText(annotated_frame, f"1st Tooth Angle: {first_angle:.1f} deg", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)

            print(f"⚙️ 鎖定目標 -> 幾何齒數: {teeth} | FFT 齒數: {fft_teeth}")

            dashboard = np.zeros((1000, 1600, 3), dtype=np.uint8)
            
            if unwrapped_img is not None:
                cv2.rectangle(dashboard, (0, 0), (1600, 40), (25, 25, 25), -1)
                cv2.putText(dashboard, "1. UNWRAPPED GEAR SPATIAL VERIFICATION (Green lines hit valleys)", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                uw_resized = cv2.resize(unwrapped_img, (1600, 260))
                dashboard[40:300, 0:1600] = uw_resized

            if fft_img is not None:
                dashboard[300:1000, 0:960] = fft_img

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

            # 套用智慧縮放視窗
            window_name = f"Task {task_idx} - Dashboard: Gear {idx+1}"
            show_smart_window(window_name, dashboard)
            cv2.waitKey(3000) 
            
        else:
            print("⚠️ 未檢測到齒輪遮罩，無法進行進階計算")

        print(f"🦾 準備夾取第 {idx+1} 個目標...")
        send.Go_Position(c, target_x, target_y, z_height_check_number_of_teeth, home[3], home[4], -133.567, 80)
        send.Go_Position(c, target_x, target_y, z_height, home[3], home[4], -133.567, 80)
        send.gripper_ON(c)
        send.Go_Position(c, target_x, target_y, home[2], home[3], home[4], -133.567, 80)
        send.Go_Position(c, drop[0], drop[1], drop[2],  drop[3], drop[4], drop[5], 80)
        send.Go_Position(c, drop[0], drop[1], drop[2]-35,  drop[3], drop[4], drop[5], 80)
        send.gripper_OFF(c)
        send.Go_Position(c, drop[0], drop[1], drop[2],  -179.145, -0.321, -133.959, 80)
        send.Go_Position(c, home[0], home[1], home[2], home[3], home[4], home[5], 80)

def main(task_idx):
    send.Go_Position(c, home[0], home[1], home[2], home[3], home[4], home[5], 80)
    send.gripper_OFF(c)
    
    print("📷 正在等待相機啟動與清除緩衝區...")
    success_frames = 0
    for i in range(50):
        try:
            frame = realsense.Get_RGB_Frame()
            success_frames += 1
        except RuntimeError as e:
            print(f"⚠️ 相機讀取超時 (暖機中...)，正在重試... ({i+1}/50)")
            time.sleep(0.1) 
            
    if success_frames == 0:
        print("❌ 錯誤：相機完全沒有畫面！請檢查 USB 連線並重新插拔相機。")
        return 
    
    real_points = []
    
    print(f"🚀 開始執行任務 {task_idx}：搜尋高處目標...")
    while len(real_points) == 0:
        output, real_points = Find_Gear_Object()
        cv2.waitKey(1)
        
    try:
        cv2.destroyWindow("Step 1: High View Lock (Live)")
    except Exception:
        pass
    
    # 輸出高處的最終結果並智慧調整大小
    show_smart_window(f"Task {task_idx} - High View Final Result", output)
        
    print(f"🎯 高處尋找完畢，共找到 {len(real_points)} 個目標座標！")
    cv2.waitKey(1000)
    
    gripper_Behave(real_points, task_idx)
    print(f"✅ 任務 {task_idx} 搜尋清空完畢!!!")

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

print("🔌 正在連線至 Modbus 機器手臂...")
c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
c.connect()
print("🟢 手臂連線成功！")

gripper_rz = -133.567
home = [386.077, -51.439, 680,  -179.161, -0.32, -102.22800000000001]
drop = [474.953, -228.878, z_height+50, -178.932, 0.478, -134.746]

if __name__ == "__main__":
    try:
        for i in range(3): 
            main(task_idx=i+1) 
            
        print("\n" + "="*50)
        print("🏁 所有任務已順利完成！")
        print("👀 輸出的視窗已全部保留在螢幕上供您檢視。")
        print("👉 檢視完畢後，請點擊任一影像視窗，並按下「鍵盤任意鍵」以關閉程式...")
        print("="*50 + "\n")
        
        cv2.waitKey(0) 
        
    finally:
        c.close()
        cv2.destroyAllWindows()
        print("🔌 系統已安全關閉。")
