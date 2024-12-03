import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# 1. 載入保存的模型
save_path = r"N:\Project\model\saved_model\model.keras"
# 修改為模型位置
model = load_model(save_path)

# 2. 測試數據加載與預處理
def preprocess_data(filepath):
    # 假設測試數據與訓練數據結構一致
    with np.load(filepath) as npz_file:
        frames = [npz_file[f'frame_{i}'] for i in range(1, 101)]
        wave_data = np.stack(frames, axis=-1)  # (41, 41, 100)
    
    # 標準化處理
    scaler = StandardScaler()
    wave_data_reshaped = wave_data.reshape(-1, 100)
    wave_data_normalized = scaler.fit_transform(wave_data_reshaped).reshape(wave_data.shape)
    return wave_data_normalized

# 3. 載入測試數據
test_file = r"N:\Project\model\test\1.npz"
# 修改為測試數據位置
test_data = preprocess_data(test_file)
test_data = np.expand_dims(test_data, axis=0)  # 增加批次維度，形狀變為 (1, 41, 41, 100)

# 4. 模型預測
predictions = model.predict(test_data)

# 5. 取得辨識結果與正確機率
predicted_class = np.argmax(predictions, axis=-1)[0]  # 取出最高機率類別的索引
confidence = predictions[0][predicted_class]  # 對應的正確機率

# 6. 對應物體名稱
object_names = ["CJ10", "DF21", "DF31", "DF41", "YJ08", "YJ12",
                "YJ18", "YJ21", "YJ62", "YJ82", "YJ83", "YJ91"]
predicted_object = object_names[predicted_class]

# 7. 輸出結果
print(f"預測結果: {predicted_object}")
print(f"正確機率: {confidence:.2%}")