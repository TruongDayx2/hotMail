import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import os



# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('hand_rotation_model.h5')

# Đặt kích thước target_size (bạn nên thay bằng kích thước thực tế đã xác định)
target_size = (224, 224)

# Hàm dự đoán góc từ ảnh
def predict_angle(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, target_size) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_angle = predicted_class * 10  # Mỗi lớp đại diện cho 10 độ
    return predicted_angle


def setCircle():
    # Load the image

  image_path = 'hand.PNG'

  image = Image.open(image_path)



  # Determine the center and radius

  width, height = image.size

  center_x, center_y = width // 2, height // 2

  radius = center_x  # Radius from the center to the left edge



  # Create a circular mask

  mask = Image.new("L", (width, height), 0)

  draw = ImageDraw.Draw(mask)

  draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=255)



  # Apply the mask to the image

  circular_image = Image.new("RGBA", image.size)

  circular_image.paste(image, (0, 0), mask=mask)


  circular_image.save(f"circleTest.png")

# Ví dụ cách sử dụng hàm
if __name__ == "__main__":
    setCircle()
    new_image_path = 'circleTest.png'  # Thay đường dẫn ảnh
    predicted_angle = predict_angle(new_image_path)
    print(f"Góc dự đoán: {predicted_angle}°")
