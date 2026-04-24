import sys
import numpy as np
import os
import serial
import serial.tools.list_ports
import re
import time

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLineEdit, QLabel, QMessageBox,
                             QComboBox, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 类别映射
class_names = {0: "ECG (心电)", 1: "Sine (正弦)", 2: "Noise (噪声)"}

# 串口读取线程
class SerialReader(QThread):
    data_received = pyqtSignal(str)
    
    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.running = True

    def run(self):
        while self.running:
            try:
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('ascii', errors='ignore').strip()
                    if line:
                        self.data_received.emit(line)
            except Exception as e:
                print(f"Serial read error: {e}")
                break
            time.sleep(0.01) # Avoid busy-waiting

    def stop(self):
        self.running = False
        self.wait()

class ECGSTM32VizApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG/Sine/Noise STM32 串口验证工具")
        self.setGeometry(100, 100, 1000, 750)

        # 1. 加载数据
        self.load_data()

        # 2. 初始化当前索引
        self.current_index = 0

        # 3. 串口对象
        self.ser = None
        self.serial_reader_thread = None

        # 4. 设置 UI
        self.init_ui()

        # 5. 显示初始样本
        self.update_sample()

    def load_data(self):
        # 检查文件是否存在
        if not (os.path.exists('X_test.npy') and os.path.exists('y_test.npy')):
            QMessageBox.critical(self, "错误", "未找到测试集文件 (X_test.npy, y_test.npy)！")
            sys.exit()
        
        # 加载测试集
        self.X_test = np.load('X_test.npy').astype(np.float32)
        self.y_test = np.load('y_test.npy').astype(np.int64)

    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- 串口配置区域 ---
        serial_config_layout = QHBoxLayout()
        serial_config_layout.addWidget(QLabel("串口:"))
        self.combo_ports = QComboBox()
        serial_config_layout.addWidget(self.combo_ports)
        
        serial_config_layout.addWidget(QLabel("波特率:"))
        self.combo_baud = QComboBox()
        self.combo_baud.addItems(["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"])
        self.combo_baud.setCurrentText("115200")
        serial_config_layout.addWidget(self.combo_baud)

        self.btn_open_serial = QPushButton("打开串口")
        self.btn_open_serial.clicked.connect(self.toggle_serial_port)
        serial_config_layout.addWidget(self.btn_open_serial)
        
        self.btn_refresh_ports = QPushButton("刷新串口")
        self.btn_refresh_ports.clicked.connect(self.refresh_ports)
        serial_config_layout.addWidget(self.btn_refresh_ports)

        serial_config_layout.addStretch(1)
        main_layout.addLayout(serial_config_layout)
        
        # 在创建完所有串口相关控件后再调用 refresh_ports
        self.refresh_ports()

        # --- 顶部显示区域 (Matplotlib) ---
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        main_layout.addWidget(self.canvas)

        # --- 中间结果显示区域 ---
        result_layout = QHBoxLayout()
        self.lbl_true = QLabel("真实类别: -")
        self.lbl_pred_mcu = QLabel("MCU 预测类别: -")
        self.lbl_conf_mcu = QLabel("MCU 置信度: -")
        
        # 设置字体样式
        font = self.lbl_true.font()
        font.setPointSize(12)
        font.setBold(True)
        for lbl in [self.lbl_true, self.lbl_pred_mcu, self.lbl_conf_mcu]:
            lbl.setFont(font)
            result_layout.addWidget(lbl)
        
        main_layout.addLayout(result_layout)

        # --- 底部控制区域 ---
        control_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("上一个样本")
        self.btn_prev.clicked.connect(self.prev_sample)
        
        self.edit_index = QLineEdit()
        self.edit_index.setFixedWidth(80)
        self.edit_index.setPlaceholderText("0-{}".format(len(self.X_test)-1))
        self.edit_index.returnPressed.connect(self.jump_to_index)
        
        self.btn_jump = QPushButton("跳转")
        self.btn_jump.clicked.connect(self.jump_to_index)
        
        self.btn_next = QPushButton("下一个样本")
        self.btn_next.clicked.connect(self.next_sample)

        self.btn_send_to_mcu = QPushButton("发送到 STM32")
        self.btn_send_to_mcu.setStyleSheet("background-color: #aaffaa; font-weight: bold;")
        self.btn_send_to_mcu.clicked.connect(self.send_sample_to_mcu)
        self.btn_send_to_mcu.setEnabled(False) # 默认禁用，串口打开后启用

        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(QLabel("索引:"))
        control_layout.addWidget(self.edit_index)
        control_layout.addWidget(self.btn_jump)
        control_layout.addWidget(self.btn_next)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.btn_send_to_mcu)
        
        main_layout.addLayout(control_layout)

    def refresh_ports(self):
        self.combo_ports.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.combo_ports.addItem(port.device)
        if not ports:
            self.combo_ports.addItem("无可用串口")
            self.btn_open_serial.setEnabled(False)
        else:
            self.btn_open_serial.setEnabled(True)

    def toggle_serial_port(self):
        if self.ser and self.ser.is_open:
            self.stop_serial_reader()
            self.ser.close()
            self.btn_open_serial.setText("打开串口")
            self.btn_open_serial.setStyleSheet("")
            self.btn_send_to_mcu.setEnabled(False)
            self.combo_ports.setEnabled(True)
            self.combo_baud.setEnabled(True)
            self.btn_refresh_ports.setEnabled(True)
        else:
            port_name = self.combo_ports.currentText()
            baud_rate = int(self.combo_baud.currentText())
            if port_name == "无可用串口":
                QMessageBox.warning(self, "警告", "没有选择有效的串口！")
                return
            try:
                self.ser = serial.Serial(port_name, baud_rate, timeout=1)
                self.start_serial_reader()
                self.btn_open_serial.setText("关闭串口")
                self.btn_open_serial.setStyleSheet("background-color: #ffaaaa;")
                self.btn_send_to_mcu.setEnabled(True)
                self.combo_ports.setEnabled(False)
                self.combo_baud.setEnabled(False)
                self.btn_refresh_ports.setEnabled(False)
                QMessageBox.information(self, "成功", f"串口 {port_name} 已打开！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法打开串口 {port_name}: {e}")

    def start_serial_reader(self):
        if self.serial_reader_thread:
            self.serial_reader_thread.stop()
            self.serial_reader_thread.wait()
        self.serial_reader_thread = SerialReader(self.ser)
        self.serial_reader_thread.data_received.connect(self.handle_mcu_response)
        self.serial_reader_thread.start()

    def stop_serial_reader(self):
        if self.serial_reader_thread:
            self.serial_reader_thread.stop()
            self.serial_reader_thread.wait()
            self.serial_reader_thread = None

    def handle_mcu_response(self, response_str):
        print(f"Received from MCU: {response_str}")
        # 解析 MCU 返回的字符串，例如 "P=[0.996094,0.003174,0.000732], pred=0"
        match = re.match(r"P=\[(.+),(.+),(.+)\], pred=(\d+)", response_str)
        if match:
            probs = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
            pred_label = int(match.group(4))
            confidence = probs[pred_label]

            self.lbl_pred_mcu.setText(f"MCU 预测类别: {class_names[pred_label]}")
            self.lbl_conf_mcu.setText(f"MCU 置信度: {confidence:.3%} (P=[{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}])")
            
            true_label = self.y_test[self.current_index]
            if pred_label == true_label:
                self.lbl_pred_mcu.setStyleSheet("color: green;")
            else:
                self.lbl_pred_mcu.setStyleSheet("color: red;")
        else:
            self.lbl_pred_mcu.setText("MCU 预测类别: 解析失败")
            self.lbl_conf_mcu.setText("MCU 置信度: -")
            self.lbl_pred_mcu.setStyleSheet("color: orange;") # 橙色表示解析错误

    def update_sample(self):
        # 1. 获取当前数据
        sample_data = self.X_test[self.current_index : self.current_index + 1]
        true_label = self.y_test[self.current_index]

        # 2. 更新 UI 文本 (MCU 结果清空，等待新数据)
        self.lbl_true.setText(f"真实类别: {class_names[true_label]}")
        self.lbl_pred_mcu.setText("MCU 预测类别: -")
        self.lbl_conf_mcu.setText("MCU 置信度: -")
        self.lbl_pred_mcu.setStyleSheet("") # 清除颜色
        self.edit_index.setText(str(self.current_index))

        # 3. 更新绘图
        self.ax.clear()
        self.ax.plot(sample_data[0, :, 0], color='#1f77b4', linewidth=1.5)
        self.ax.set_title(f"Sample index: {self.current_index} (1s @ 360Hz)", fontsize=14)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.canvas.draw()

    def send_sample_to_mcu(self):
        if not (self.ser and self.ser.is_open):
            QMessageBox.warning(self, "警告", "串口未打开！")
            return

        sample_data = self.X_test[self.current_index, :, 0]
        
        # 准备发送的数据帧 (使用 ADC 整数形式，方便 STM32 处理)
        # 将归一化浮点数转换回 0-4095 的整数，模拟 ADC 原始数据
        adc_data = np.clip((sample_data * 4095.0).round().astype(int), 0, 4095)
        payload = ','.join(map(str, adc_data.tolist()))
        frame = f'$E,{payload}\r\n' # 使用 CRLF 作为行结束，匹配常见 STM32 USART 终止符

        try:
            self.ser.write(frame.encode('ascii'))
            self.ser.flush()
            print(f"Sent to MCU: {frame.strip()}")
            # 清空 MCU 结果显示，等待新的响应
            self.lbl_pred_mcu.setText("MCU 预测类别: 等待响应...")
            self.lbl_conf_mcu.setText("MCU 置信度: 等待响应...")
            self.lbl_pred_mcu.setStyleSheet("color: gray;")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"发送数据失败: {e}")

    def next_sample(self):
        if self.current_index < len(self.X_test) - 1:
            self.current_index += 1
            self.update_sample()

    def prev_sample(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_sample()

    def jump_to_index(self):
        try:
            idx = int(self.edit_index.text())
            if 0 <= idx < len(self.X_test):
                self.current_index = idx
                self.update_sample()
            else:
                QMessageBox.warning(self, "警告", "索引超出范围！")
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数字索引！")

    def closeEvent(self, event):
        if self.ser and self.ser.is_open:
            self.stop_serial_reader()
            self.ser.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 支持中文字体显示
    font = QApplication.font()
    font.setFamily("SimHei") # 尝试使用黑体
    app.setFont(font)

    window = ECGSTM32VizApp()
    window.show()
    sys.exit(app.exec_())
