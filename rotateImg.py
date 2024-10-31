import cv2
import numpy as np
import os

# Đường dẫn tới ảnh gốc (hướng 0 độ)
input_image_path = 'hand_0.png'  # Thay đổi đường dẫn tới ảnh gốc
output_dir = 'hand'  # Thư mục lưu ảnh xoay

# Tạo thư mục lưu ảnh nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Đọc ảnh gốc
image = cv2.imread(input_image_path)

# Kích thước ảnh gốc
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# Tính toán kích thước khung hình mở rộng
# Đường chéo là kích thước lớn nhất cần thiết để xoay mà không mất phần ảnh nào
diag_len = int(np.sqrt(h**2 + w**2))
pad_h = (diag_len - h) // 2
pad_w = (diag_len - w) // 2

# Thêm padding cho ảnh để mở rộng khung hình
padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

# Tâm của ảnh sau khi thêm padding
padded_center = (diag_len // 2, diag_len // 2)

# Xoay ảnh theo từng góc 5 độ và lưu lại
for angle in range(0, 360, 5):
    # Tạo ma trận xoay và xoay ảnh
    rotation_matrix = cv2.getRotationMatrix2D(padded_center, -angle, 1.0)
    rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (diag_len, diag_len))
    
    # Lưu ảnh với tên chứa thông tin góc
    output_path = os.path.join(output_dir, f'hand2_{angle}.png')
    cv2.imwrite(output_path, rotated_image)
    print(f"Saved rotated image at angle {angle}° as {output_path}")
