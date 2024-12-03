import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. 數據加載與預處理
base_dir = r"N:\Project\model\data"
object_names = ["CJ10", "DF21", "DF31", "DF41", "YJ08", "YJ12",
                "YJ18", "YJ21", "YJ62", "YJ82", "YJ83", "YJ91"]

def load_data(base_dir, object_names):
    data = []
    labels_object = []
    
    for obj_name in object_names:
        obj_dir = os.path.join(base_dir, obj_name)
        for filename in os.listdir(obj_dir):
            if filename.endswith(".npz"):
                filepath = os.path.join(obj_dir, filename)
                
                # 加載所有 100 個 frames 並組合成 (41, 41, 100) 的結構
                with np.load(filepath) as npz_file:
                    frames = [npz_file[f'frame_{i}'] for i in range(1, 101)]
                    wave_data = np.stack(frames, axis=-1)  # 合併為 (41, 41, 100)
                
                data.append(wave_data)
                labels_object.append(obj_name)
    
    data = np.array(data)
    labels_object = np.array(labels_object)
    
    return data, labels_object

# 加載數據
X, y_object = load_data(base_dir, object_names)

# 標準化數據
scaler = StandardScaler()

# 逐樣本逐時間點進行標準化
X_reshaped = X.reshape(-1, 100)  # 先將每個樣本展平為 (樣本數 * 41 * 41, 100)
X_normalized = scaler.fit_transform(X_reshaped).reshape(X.shape)  # 再將標準化後的數據重塑回 (樣本數, 41, 41, 100)

# 更新數據
X = X_normalized

# 物體名稱標籤化
label_encoder = LabelEncoder()
y_object_encoded = label_encoder.fit_transform(y_object)
y_object_categorical = to_categorical(y_object_encoded, num_classes=len(object_names))

# 2. 建立CNN模型
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

input_shape = (41, 41, 100)
num_classes = len(object_names)
cnn_model = create_cnn_model(input_shape, num_classes)
cnn_model.summary()

# 3. 訓練模型
history = cnn_model.fit(X, y_object_categorical, epochs=20, batch_size=32, validation_split=0.05)

# 4. 保存模型
# 指定保存的路徑
save_dir = r"N:\Project\model"
save_path = os.path.join(save_dir, "model.keras")

# 確保目錄存在
os.makedirs(save_dir, exist_ok=True)

# 保存模型
cnn_model.save(save_path)
print(f"模型已保存到: {save_path}")