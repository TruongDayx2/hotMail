import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Đường dẫn tới thư mục chứa dataset
data_dir = 'hand'  # Thay đổi đường dẫn tới thư mục chứa ảnh xoay

# Cấu hình ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # 80% train, 20% validation
)

# Tạo generator cho tập huấn luyện
train_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(128, 128),  # Kích thước ảnh đưa vào mô hình
    batch_size=32,
    class_mode='categorical',  # Phân loại nhiều lớp
    subset='training',
    shuffle=True
)

# Tạo generator cho tập validation
validation_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Số lượng lớp đầu ra (tương ứng với các góc độ)
num_classes = len(train_generator.class_indices)  # Số lớp phân loại (ví dụ: 72 cho các góc từ 0-355 với bước 5)

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Lớp đầu ra với softmax cho phân loại
])

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20
)
