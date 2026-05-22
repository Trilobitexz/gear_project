import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# 1. 設定序列埠 
COM_PORT = "COM5" 
BAUD_RATE = 115200

try:
    # 稍微縮短 timeout，避免極端情況下卡死 UI
    arduino_nano = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2) 
    
    # 💡 新增：清空一開始連線時累積的雜訊與舊資料
    arduino_nano.reset_input_buffer() 
    
    print(f"成功連線至 Arduino Nano！串口狀態: {arduino_nano.is_open}")
    print(f"串口資訊: {arduino_nano}")
except Exception as e:
    print(f"連線失敗，請檢查線路或 COM Port: {e}")
    exit()

# 2. 設定資料儲存容器
MAX_POINTS = 50
y_pressure = deque(maxlen=MAX_POINTS)

# 初始化一些測試數據
for i in range(10):
    y_pressure.append(512)

# 3. 初始化 Matplotlib 視窗與子圖
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
fig.canvas.manager.set_window_title('壓力感測器即時監控')

line_press, = ax.plot([], [], lw=2, color='#d62728', label='Pressure (Analog)')
ax.set_xlim(0, MAX_POINTS)
ax.set_ylim(0, 1024) 
ax.set_ylabel("Pressure Value")
ax.set_xlabel("Data Frames")
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.6)

# 4. 定義動畫更新函數
def update(frame):
    # 💡 新增：先檢查有沒有資料，沒有就直接 return 繼續畫上一幀，不要卡死等待
    if arduino_nano.in_waiting == 0:
        return line_press,

    try:
        data_raw = arduino_nano.readline()
        
        if data_raw:
            data_str = data_raw.decode('utf-8').strip()
            # print(f"Raw: {data_str}") # 若畫面正常了，可以把這行註解掉避免終端機洗版
            
            sensor_values = [int(val) for val in data_str.split(',')]
            
            if len(sensor_values) >= 5:
                press = sensor_values[4]
                
                y_pressure.append(press)
                
                line_press.set_data(range(len(y_pressure)), y_pressure)
                
                # print(f"壓力: {press:4d}")
                
    except ValueError:
        # 偶爾會有雜訊導致無法轉成整數，略過即可
        pass
    except UnicodeDecodeError:
        # 處理偶爾出現的亂碼無法解碼錯誤
        pass

    return line_press,

# 5. 啟動即時動畫
ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show() 

arduino_nano.close()
print("序列埠已安全關閉，程式結束。")