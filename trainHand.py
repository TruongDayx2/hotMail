import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = 'circle'

# Danh sách lưu ảnh và nhãn
images = []
labels = []

# Xác định kích thước target_size dựa trên bán kính của hình tròn
def get_target_size(sample_image_path):
    img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(largest_contour)
        return (int(radius * 2), int(radius * 2))
    return (224, 224)  # Kích thước mặc định nếu không tìm thấy hình tròn

sample_image_path = os.path.join(data_dir, os.listdir(data_dir)[0])
target_size = get_target_size(sample_image_path)

# Đọc và gán nhãn cho các ảnh
for filename in os.listdir(data_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Đọc ảnh
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)
        
        # Đảm bảo kích thước phù hợp với mô hình
        img_resized = cv2.resize(img, target_size)
        
        # Lấy góc từ tên tệp
        angle = int(filename.split('_')[-1].split('.')[0])
        label = angle // 10  # Mỗi lớp ứng với góc cách nhau 10 độ
        
        images.append(img_resized)
        labels.append(label)

# Chuyển đổi sang numpy array và chuẩn hóa
images = np.array(images) / 255.0
labels = np.array(labels)

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Chuyển đổi nhãn sang one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=36)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=36)

# Tăng cường dữ liệu với ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
val_generator = val_datagen.flow(x_val, y_val, batch_size=32)

# Xây dựng mô hình dựa trên ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*target_size, 3))

# Đóng băng các lớp của ResNet50 để không huấn luyện lại
for layer in base_model.layers:
    layer.trainable = False

# Thêm các lớp mới cho bài toán phân loại góc quay
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(36, activation='softmax')  # 36 lớp tương ứng với 36 góc
])

# Biên dịch mô hình
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

# Đánh giá mô hình trên tập kiểm tra
loss, accuracy = model.evaluate(val_generator)
print(f'Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%')

# Lưu mô hình đã huấn luyện
model.save('hand_rotation_model.h5')
