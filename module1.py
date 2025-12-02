import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Tên file dữ liệu cần đọc
BINARY_FILE = "particles.bin"

# Lấy đường dẫn tuyệt đối của file bin (cùng thư mục với script này)
script_dir = os.path.dirname(os.path.abspath(__file__))
bin_path = os.path.join(script_dir, BINARY_FILE)

def main():
    # --- BƯỚC 1: KIỂM TRA FILE DỮ LIỆU ---
    if not os.path.exists(bin_path):
        print(f"LỖI: Không tìm thấy file '{BINARY_FILE}'.")
        print("Vui lòng chạy chương trình C++ trước để sinh ra file dữ liệu này!")
        sys.exit(1)

    print(f"Đang đọc dữ liệu từ: {BINARY_FILE} ...")

    # --- BƯỚC 2: ĐỌC FILE BINARY ---
    try:
        with open(bin_path, "rb") as f:
            # Đọc Header: [N] [STEPS] (2 số nguyên int32)
            header = np.fromfile(f, dtype=np.int32, count=2)
            if header.size < 2:
                print("Lỗi: Header file không hợp lệ.")
                sys.exit(1)
                
            N, steps = header[0], header[1]
            
            # Đọc dữ liệu (float4: x, y, z, mass)
            raw_data = np.fromfile(f, dtype=np.float32)
            
            # Kiểm tra khớp dữ liệu
            expected_floats = steps * N * 4
            if raw_data.size != expected_floats:
                # Nếu C++ đang chạy dở hoặc bị tắt ngang, dữ liệu có thể thiếu
                real_steps = raw_data.size // (N * 4)
                if real_steps == 0:
                    print("Lỗi: File dữ liệu chưa có đủ một frame hoàn chỉnh.")
                    sys.exit(1)
                
                raw_data = raw_data[:real_steps * N * 4]
                steps = real_steps
                print(f"Cảnh báo: Dữ liệu không khớp header. Chỉ hiển thị {steps} bước.")

            # Reshape thành (Time, Particles, Properties)
            frames = raw_data.reshape(steps, N, 4)
            print(f"Load thành công: {N} hạt, {steps} bước.")

    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        sys.exit(1)

    # --- BƯỚC 3: VẼ ĐỒ HỌA ---
    print("Đang render animation...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("black")
    ax.grid(False)
    ax.set_axis_off()

    # Auto-scale
    all_xyz = frames[:, :, :3]
    if all_xyz.size > 0:
        bound = np.max(np.abs(all_xyz)) * 0.9
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_zlim(-bound, bound)

    # Init Scatter
    start_frame = frames[0]
    colors = plt.cm.spring(np.linspace(0, 1, N)) 
    scatter = ax.scatter(start_frame[:, 0], start_frame[:, 1], start_frame[:, 2], 
                         s=2, c=colors, alpha=0.8, edgecolors='none')
    
    txt = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color="white")

    def update(frame_idx):
        data = frames[frame_idx]
        scatter._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
        txt.set_text(f"Step: {frame_idx}/{steps}")
        return scatter, txt

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=20, blit=False)
    plt.show()

if __name__ == "__main__":
    main()