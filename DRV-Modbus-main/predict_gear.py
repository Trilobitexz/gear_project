import argparse
import os
import random
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 0. 影像前處理模組
# ==========================================
def enhance_contrast_and_denoise(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)

# ==========================================
# 1. 攤平圓形齒輪 (極座標轉換)
# ==========================================
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

# ==========================================
# 2. 精準座標映射
# ==========================================
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

# ==========================================
# 🌟 3. FFT 頻譜分析 (比例優化版 960x700)
# ==========================================
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
    cv2.putText(spectrum_img, "2. FFT FREQUENCY SPECTRUM (Left: Waveform, Right: FFT)", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

# ==========================================
# 4. 幾何核心演算法
# ==========================================
def count_teeth_from_mask(mask_array: np.ndarray, annotated_img: np.ndarray):
    mask_uint8 = (mask_array * 255).astype(np.uint8)
    
    kernel = np.ones((5, 5), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, None, 0, None, 0.0, [], mask_uint8
        
    c = max(contours, key=cv2.contourArea)
    
    M = cv2.moments(c)
    if M["m00"] == 0:
        return 0, None, 0, None, 0.0, [], mask_uint8
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    center = np.array([center_x, center_y])
    
    orange = (0, 165, 255)
    cv2.line(annotated_img, (center_x - 30, center_y), (center_x + 30, center_y), orange, 3)
    cv2.line(annotated_img, (center_x, center_y - 30), (center_x, center_y + 30), orange, 3)
    cv2.circle(annotated_img, (center_x, center_y), 5, orange, -1) 
    
    _, _, w, h = cv2.boundingRect(c)
    gear_diameter = max(w, h)
    
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

# ==========================================
# 5. 主程式入口 (合成一體化儀表板)
# ==========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gear Whole Object Segmentation")
    parser.add_argument("image", nargs="?", default=None, help="Input image path")
    parser.add_argument("--weights", default=r"C:\我的\大學\大二下\自主專題_智慧機器人_機器人感測與周邊整合\tooth-2.v1i.yolov11\runs\segment\gear_seg_v1\weights\best.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.85, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IOU threshold")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    if args.image is None:
        training_data_dir = r"C:\我的\大學\大二下\自主專題_智慧機器人_機器人感測與周邊整合\訓練資料2"
        images = [f for f in os.listdir(training_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
             print(f"⚠️ 訓練資料夾 {training_data_dir} 是空的或沒有圖片。")
             return
        args.image = os.path.join(training_data_dir, random.choice(images))
        print(f"📸 隨機選取圖片: {args.image}")
    
    original = cv2.imread(os.path.normpath(args.image))
    if original is None:
        print(f"⚠️ 無法讀取圖片: {args.image}")
        return

    enhanced = enhance_contrast_and_denoise(original)
    model = YOLO(args.weights)
    results = model.predict(source=enhanced, conf=args.conf, iou=args.iou, augment=False, imgsz=640, verbose=False, device=0)

    if len(results) == 0 or len(results[0].boxes) == 0:
        print("⚠️ 未檢測到齒輪。")
        return

    annotated = results[0].plot()
    mask_data = results[0].masks.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask_data, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    teeth, gear_center, gear_dia, first_tooth_pt, first_angle, valleys, clean_mask = count_teeth_from_mask(mask_resized, annotated)
    
    fft_teeth = 0
    fft_img = None
    unwrapped_img = None
    
    if gear_center is not None and gear_dia > 0:
        max_radius = (gear_dia / 2) * 1.1 
        unwrapped_mask = unwrap_gear(clean_mask, tuple(gear_center), max_radius)
        # 取得縮小優化比例後的 FFT 圖像 (960 x 700)
        fft_img, fft_teeth = fourier_tooth_analysis(unwrapped_mask, w=960, h=700)
        
        # 取得拉平原圖
        unwrapped_img = unwrap_gear(original, tuple(gear_center), max_radius)
        for v in valleys:
            mapped_x, mapped_y = map_to_unwrapped_coords(v, tuple(gear_center), max_radius)
            if 0 <= mapped_x < unwrapped_img.shape[1] and 0 <= mapped_y < unwrapped_img.shape[0]:
                cv2.line(unwrapped_img, (mapped_x, 0), (mapped_x, unwrapped_img.shape[0]), (0, 255, 0), 4)
                cv2.circle(unwrapped_img, (mapped_x, mapped_y), 10, (0, 0, 255), -1)

    # 主畫面文字框
    overlay = annotated.copy()
    cv2.rectangle(overlay, (20, 20), (1050, 400), (0, 0, 0), -1) 
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    cv2.putText(annotated, f"Geometry Teeth: {teeth}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)
    cv2.putText(annotated, f"FFT Verification: {fft_teeth}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
    cv2.putText(annotated, f"1st Tooth Angle: {first_angle:.1f} deg", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)

    # ==========================================
    # 🏆 組合「全方位戰情儀表板」 (總尺寸 1600 x 1000)
    # ==========================================
    dashboard = np.zeros((1000, 1600, 3), dtype=np.uint8)
    
    # [板塊 1] 上方：拉平圖 (1600 x 300)
    if unwrapped_img is not None:
        cv2.rectangle(dashboard, (0, 0), (1600, 40), (25, 25, 25), -1)
        cv2.putText(dashboard, "1. UNWRAPPED GEAR SPATIAL VERIFICATION (Green lines hit valleys)", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        uw_resized = cv2.resize(unwrapped_img, (1600, 260))
        dashboard[40:300, 0:1600] = uw_resized

    # [板塊 2] 左下：FFT 圖 (960 x 700)
    if fft_img is not None:
        dashboard[300:1000, 0:960] = fft_img

    # [板塊 3] 右下：齒輪標示圖 (填滿剩餘的 640 x 700 空間)
    cv2.rectangle(dashboard, (960, 300), (1600, 340), (25, 25, 25), -1)
    cv2.putText(dashboard, "3. GEOMETRY DETECTION", (975, 328), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    annot_h, annot_w = annotated.shape[:2]
    # 計算等比例縮放，確保放得進 640 x 660 的區塊
    scale = min(640 / annot_w, 660 / annot_h)
    new_w, new_h = int(annot_w * scale), int(annot_h * scale)
    annot_resized = cv2.resize(annotated, (new_w, new_h))
    
    # 將齒輪圖置中貼上
    y_offset = 340 + (660 - new_h) // 2
    x_offset = 960 + (640 - new_w) // 2
    dashboard[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = annot_resized
    
    # 畫分割線
    cv2.line(dashboard, (0, 300), (1600, 300), (100, 100, 100), 2)
    cv2.line(dashboard, (960, 300), (960, 1000), (100, 100, 100), 2)

    # 顯示最終儀表板 (允許使用者自由縮放視窗大小)
    cv2.namedWindow("Industrial AOI Dashboard", cv2.WINDOW_NORMAL)
    cv2.imshow("Industrial AOI Dashboard", dashboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()