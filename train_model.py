import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def build_model(input_shape):
    # 使用 Keras 的 Sequential 模型按顺序堆叠网络层
    model = models.Sequential([
        # 第一层卷积：16个滤波器，卷积核大小为5，使用ReLU激活函数
        # input_shape 应为 (360, 1)，表示1秒长度的单通道信号
        layers.Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape),
        # 第一个最大池化层：缩小特征图尺寸，降低计算量并提取显著特征
        layers.MaxPooling1D(pool_size=2),
        # 第二层卷积：增加到32个滤波器，提取更高阶的抽象特征
        layers.Conv1D(32, kernel_size=5, activation='relu'),
        # 第二个最大池化层
        layers.MaxPooling1D(pool_size=2),
        # 展平层：将多维特征向量拉直，以便连接到全连接层
        layers.Flatten(),
        # 全连接层：32个神经元，进一步进行特征组合
        layers.Dense(32, activation='relu'),
        # 输出层：3个神经元对应3个类别（ECG、正弦、噪声）
        # 使用 softmax 激活函数将输出转换为概率分布
        layers.Dense(3, activation='softmax')
    ])
    return model

def train_model():
    # 1. 加载预处理好的数据（由 prepare_data.py 生成）
    # 使用 .astype 确保数据类型符合 TensorFlow 的输入要求
    X_train = np.load('X_train.npy').astype(np.float32) # 训练集特征
    y_train = np.load('y_train.npy').astype(np.int64)   # 训练集标签
    X_val = np.load('X_val.npy').astype(np.float32)     # 验证集特征
    y_val = np.load('y_val.npy').astype(np.int64)       # 验证集标签

    # 2. 确定输入形状并构建模型
    # 输入形状为 (信号长度, 通道数)，例如 (360, 1)
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape)

    # 3. 编译模型
    # 使用 Adam 优化器，自动调整学习率
    # 损失函数使用稀疏类别交叉熵，适用于多分类任务且标签为整数
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 打印模型结构，查看各层参数量和输出尺寸
    model.summary()

    # 4. 开始训练
    # 训练10个轮次 (epochs)，每批处理32条数据 (batch_size)
    # 每个轮次结束后使用验证集评估性能，防止过拟合
    history = model.fit(X_train, y_train, 
                        epochs=10, 
                        batch_size=32,
                        validation_data=(X_val, y_val))

    # 5. 保存训练好的模型为 HDF5 格式
    model.save('ecg_classifier_model.h5')
    print("Model saved to ecg_classifier_model.h5")

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig('training_results_2.png')
    print("Training history plot saved to training_results.png")

if __name__ == "__main__":
    train_model()
