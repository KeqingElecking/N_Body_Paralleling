import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Tên file output từ chương trình C++
FILENAME = "particles.bin"

if not os.path.exists(FILENAME):
    print(f"Error: {FILENAME} not found. Run the C++ simulation first!")
    sys.exit(1)

print(f"Loading {FILENAME}...")

try:
    with open(FILENAME, "rb") as f:
        # Đọc Header
        N = np.fromfile(f, dtype=np.int32, count=1)[0]
        steps = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        print(f"Simulation Info: N={N}, Steps={steps}")
        
        # Đọc toàn bộ dữ liệu vị trí
        # Mỗi frame có N hạt, mỗi hạt là struct float4 (4 floats: x, y, z, mass)
        # Tổng số float cần đọc = Steps * N * 4
        raw_data = np.fromfile(f, dtype=np.float32)
        
        # Reshape thành mảng 3 chiều: (Số frame, Số hạt, 4 giá trị)
        expected_size = steps * N * 4
        if raw_data.size != expected_size:
            print(f"Warning: Expected {expected_size} floats but got {raw_data.size}. Simulation might have crashed or stopped early.")
            # Cắt bớt để khớp
            frames_count = raw_data.size // (N * 4)
            raw_data = raw_data[:frames_count * N * 4]
            steps = frames_count
            
        frames = raw_data.reshape(steps, N, 4)

except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

print("Data loaded. Preparing animation...")

# Setup Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor("black")
# Tắt lưới và trục cho đẹp
ax.grid(False)
ax.set_axis_off()

# Tính giới hạn khung hình dựa trên toàn bộ dữ liệu để không bị giật
all_x = frames[:, :, 0].flatten()
all_y = frames[:, :, 1].flatten()
all_z = frames[:, :, 2].flatten()
max_range = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)), np.max(np.abs(all_z)))
limit = max_range * 0.8

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)

# Tạo đối tượng scatter ban đầu
# Lấy frame đầu tiên
start_data = frames[0]
# Màu sắc dựa trên khối lượng hoặc vị trí
colors = plt.cm.magma(np.linspace(0.5, 1, N)) 
scatter = ax.scatter(start_data[:, 0], start_data[:, 1], start_data[:, 2], 
                     s=2, c=colors, alpha=0.8, edgecolors='none')

title = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color="white")

def update(frame_idx):
    current_data = frames[frame_idx]
    
    # Cập nhật vị trí
    scatter._offsets3d = (current_data[:, 0], current_data[:, 1], current_data[:, 2])
    
    title.set_text(f"Step: {frame_idx}/{steps}")
    return scatter, title

# Tạo animation
# interval=20ms -> 50fps
ani = animation.FuncAnimation(fig, update, frames=steps, interval=20, blit=False)

plt.show()
