# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog
from Ui_second_window import Ui_MainWindow2
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# 模型导入
from models_re import model_BiLSTM, model_LSTM, model_MLP, model_Transformer

# 讯飞星火大模型接口

import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from websocket import WebSocketApp
from websocket import enableTrace

answer = ""
isFirstcontent = False
text = []

class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url

    def create_url(self):
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        url = self.Spark_url + '?' + urlencode(v)
        return url

def on_error(ws, error):
    print("### error:", error)

def on_close(ws,one,two):
    print(" ")

def on_open(ws):
    thread.start_new_thread(run, (ws,))

def run(ws, *args):
    data = json.dumps(gen_params(appid=ws.appid, domain= ws.domain,question=ws.question))
    ws.send(data)

def on_message(ws, message):
    global answer, isFirstcontent
    data = json.loads(message)
    code = data['header']['code']
    content =''
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        text = choices['text'][0]
        if ( 'reasoning_content' in text and '' != text['reasoning_content']):
            reasoning_content = text["reasoning_content"]
            print(reasoning_content, end="")
            isFirstcontent = True
        if('content' in text and '' != text['content']):
            content = text["content"]
            if(True == isFirstcontent):
                print("\n*******************思维链结束，模型回复如下********************\n")
            print(content, end="")
            isFirstcontent = False
        answer += content
        if status == 2:
            ws.close()

def gen_params(appid, domain,question):
    data = {
        "header": {
            "app_id": appid,
            "uid": "1234",
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "temperature": 0.7,
                "max_tokens": 32768
            }
        },
        "payload": {
            "message": {
                "text": question
            }
        }
    }
    return data

def main(appid, api_key, api_secret, Spark_url,domain, question):
    global answer
    answer = ""
    wsParam = Ws_Param(appid, api_key, api_secret, Spark_url)
    enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid
    ws.question = question
    ws.domain = domain
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

def getText(role, content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length

def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text

# 全局配置
APPID = "caf60f2e"
API_SECRET = "YTNlNmI0YzY5N2YyNGU2OWZkMzE3ZTQ3"
API_KEY = "27b3fb25ba19f3bd4797960484f20361"
DOMAIN = "pro-128k"
SPARK_URL = "wss://spark-api.xf-yun.com/chat/pro-128k"

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SEQ_LEN = 100
PRED_LEN = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 4
HIDDEN_DIM = 64

def H2T(V, H):
    a = -0.4134 * 4.1804
    b = -0.4134 * 0.2684 + 4.1804 * 1007.6
    c = 1007.6 * 0.2684 - H / V
    roots = np.roots([a, b, c])
    for r in roots:
        if np.isreal(r) and 0 < np.real(r) < 100:
            return float(np.real(r))
    return 43.4

def model(y, u):
    hcw = 100.6
    rolcw = 997.1
    hhw = 0
    rolhw = 0
    def hout(t): return 4.1804 * t + 0.2684
    def rolout(t): return -0.4134 * t + 1007.6
    V, H = y
    fcw, fhw, Wst = u
    x = V / 16e-4
    T = H2T(V, H)
    fout = 1e-4 * (0.1013 * np.sqrt(55 + x) + 0.0237)
    dV = fcw + fhw - fout
    dH = Wst + hcw * rolcw * fcw + hhw * rolhw * fhw - hout(T) * rolout(T) * fout
    return np.array([dV, dH])

def csth_simulation(Wst_input, fcw_input, fhw_input, T_input, fault_type):
    ts = 0.1
    t_total = 500
    Nsim = int(t_total / ts) + 1
    fault_step = 1000
    x_nom = 20.48
    T_nom = T_input
    V0 = 16 * x_nom * 1e-6
    H0 = (4.1804 * T_nom + 0.2684) * (-0.4134 * T_nom + 1007.6) * V0
    y0 = np.array([V0, H0])
    u_current = np.array([fcw_input, fhw_input, Wst_input])
    y_state = np.zeros((2, Nsim))
    y_state[:, 0] = y0
    T_history = np.zeros(Nsim)
    Wst_history = np.zeros(Nsim)
    V_history = np.zeros(Nsim)
    T_history[0] = T_nom
    Wst_history[0] = Wst_input

    r = T_nom
    Kp = 0.05
    Ti = 1
    Ki = Kp * ts / Ti
    e_prev = 0
    u_prev = Wst_input
    eff = 1.0
    pulse_step = 0

    for k in range(1, Nsim):
        y_prev = y_state[:, k-1]
        T_actual = H2T(y_prev[0], y_prev[1])
        T_measured = T_actual
        if k >= fault_step:
            if fault_type == 1:
                T_measured += 4 + 0.05 * np.random.randn()
            elif fault_type == 3:
                eff = 0.6
            elif fault_type == 4:
                T_measured += 2 * np.sin(2 * np.pi * pulse_step * ts / 2) + 0.1 * np.random.randn()
                pulse_step += 1
        e = r - T_measured
        delta_u = Kp * (e - e_prev) + Ki * e
        Wst = u_prev + delta_u
        Wst = np.clip(Wst, 0, 15.04)
        if k >= fault_step and fault_type == 2:
            Wst = 4.0
        if k >= fault_step and fault_type == 4:
            if pulse_step < 50:
                Wst = 7.1844 + (12 - 7.1844) * (pulse_step / 50)
            elif pulse_step < 100:
                Wst = 12
            elif pulse_step < 150:
                Wst = 12 - (12 - 7.1844) * ((pulse_step - 100) / 50)
            Wst = np.clip(Wst, 0, 15.04)
        u_current[2] = Wst * eff
        e_prev = e
        u_prev = Wst
        h = ts
        k1 = model(y_prev, u_current)
        k2 = model(y_prev + h * k1/2, u_current)
        k3 = model(y_prev + h * k2/2, u_current)
        k4 = model(y_prev + h * k3, u_current)
        y_now = y_prev + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        y_state[:, k] = y_now
        T_history[k] = H2T(y_now[0], y_now[1])
        Wst_history[k] = Wst
        V_history[k] = V0
    return np.arange(0, t_total+ts, ts), T_history, Wst_history, V_history

# 预报线程
class PredictThread(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, file_path, file_idx, model_choice, start_step):
        super().__init__()
        self.file_path = file_path
        self.file_idx = file_idx
        self.model_choice = model_choice
        self.start_step = start_step

    def run(self):
        try:
            self.log_signal.emit("✅ 后台生成预测动图...")
            gif_path = self.generate_pred_gif()
            self.done_signal.emit(gif_path)
        except Exception as e:
            self.error_signal.emit(str(e))

    def generate_pred_gif(self):
        def create_multistep_dataset(file_path, seq_len, pred_len):
            df = pd.read_excel(file_path)
            raw_data = df.values[:, :3]
            temp_diff = np.diff(raw_data[:, 0], axis=0).reshape(-1, 1)
            temp_diff = np.vstack(([0], temp_diff))
            data_extended = np.hstack((raw_data, temp_diff))
            scaler = MinMaxScaler()
            data_norm = scaler.fit_transform(data_extended)
            X, y = [], []
            for i in range(len(data_norm) - seq_len - pred_len + 1):
                X.append(data_norm[i:i+seq_len, :])
                y.append(data_norm[i+seq_len : i+seq_len+pred_len, 0])
            return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)), scaler, data_norm

        def load_trained_model(model_cls, file_idx):
            model_name = model_cls.__name__
            save_path = f"pth/{model_name}_data_{file_idx+1}_epoch1000_lr0.001.pth"
            if model_cls == model_BiLSTM:
                model = model_BiLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN).to(DEVICE)
            elif model_cls == model_LSTM:
                model = model_LSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN).to(DEVICE)
            elif model_cls == model_Transformer:
                model = model_Transformer(4, 64, 4, 2, SEQ_LEN, PRED_LEN).to(DEVICE)
            elif model_cls == model_MLP:
                model = model_MLP(input_dim=INPUT_DIM, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN).to(DEVICE)
            model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
            model.eval()
            return model

        X, y, scaler, data_norm = create_multistep_dataset(self.file_path, SEQ_LEN, PRED_LEN)
        model = load_trained_model(self.model_choice, self.file_idx)

        t_min = scaler.data_min_[0]
        t_range = scaler.data_max_[0] - t_min
        HIST_SHOW = SEQ_LEN
        PRED_SHOW = PRED_LEN

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.set_title("CSTH 实时故障预报（10s历史 → 5s预测）")
        ax.set_xlabel("时间步")
        ax.set_ylabel("温度 (℃)")
        ax.grid(alpha=0.3)

        line_hist, = ax.plot([], [], 'b-', linewidth=2, label='历史温度')
        line_pred, = ax.plot([], [], 'r-', linewidth=2, label='预测温度')
        line_now, = ax.plot([], [], 'g--', linewidth=1.5, label='当前时刻')
        ax.legend()

        max_frames = 180
        current_step = self.start_step
        frame_count = 0

        def update(frame):
            nonlocal current_step, frame_count
            if frame_count >= max_frames:
                return line_hist, line_pred, line_now

            x_hist = np.arange(current_step - HIST_SHOW, current_step)
            x_pred = np.arange(current_step, current_step + PRED_SHOW)
            hist_seq = data_norm[current_step - HIST_SHOW : current_step, 0]
            hist_real = hist_seq * t_range + t_min

            with torch.no_grad():
                input_tensor = X[current_step - SEQ_LEN].unsqueeze(0).to(DEVICE)
                pred_norm = model(input_tensor).cpu().numpy().flatten()
                pred_real = pred_norm * t_range + t_min
                pred_real = pred_real + (hist_real[-1] - pred_real[0])

            line_hist.set_data(x_hist, hist_real)
            line_pred.set_data(x_pred, pred_real)
            line_now.set_data([current_step, current_step], [hist_real.min()-3, hist_real.max()+3])
            ax.set_xlim(current_step - 120, current_step + 70)
            ax.set_ylim(hist_real.min()-3, hist_real.max()+3)

            current_step += 1
            if current_step >= len(data_norm)-PRED_LEN:
                current_step = self.start_step
            frame_count +=1
            return line_hist, line_pred, line_now

        ani = animation.FuncAnimation(
            fig, update,
            frames=max_frames,
            interval=80,
            blit=True,
            cache_frame_data=False
        )

        gif_path = "prediction.gif"
        writer = animation.PillowWriter(fps=12, codec="png")
        ani.save(gif_path, writer=writer, dpi=100)
        plt.close()
        return gif_path

# 讯飞大模型报告线程
class XunFeiReportThread(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, diag_data):
        super().__init__()
        self.diag_data = diag_data

    def run(self):
        global text, answer
        try:
            self.log_signal.emit("📡 正在连接讯飞大模型...")
            
            df = self.diag_data.head(30)
            data_info = df.to_string(max_rows=10)
            
            prompt = f"""你是专业的故障诊断专家，请根据以下数据生成一份专业、简洁的故障诊断报告：
数据概况：
{data_info}

请分析：
1. 故障类型
2. 可能原因
3. 风险等级
4. 处理建议
"""

            text.clear()
            question = checklen(getText("user", prompt))
            main(APPID, API_KEY, API_SECRET, SPARK_URL, DOMAIN, question)
            report = answer
            self.done_signal.emit(report)

        except Exception as e:
            self.error_signal.emit(str(e))


# 主界面
class MainWindow2(QMainWindow, Ui_MainWindow2):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.textEdit.setPlainText("你好！欢迎使用本软件！")
        self.last_t = None
        self.last_T = None
        self.last_Wst = None
        self.last_V = None
        self.diag_data = None
        self.file_path = ""
        self.label_14.mousePressEvent = self.choose_file
        self.predict_thread = None
        self.xunfei_thread = None

    def log_info(self, message):
        current_text = self.textEdit.toPlainText()
        self.textEdit.setPlainText(current_text + "\n" + message)
        current_text2 = self.textBrowser_2.toPlainText()
        self.textBrowser_2.setPlainText(current_text2 + "\n" + message)

    def choose_file(self, event):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "Excel (*.xlsx)")
        if path:
            self.lineEdit.setText(path)
            self.file_path = path
            self.log_info(f"📂 已选择：{path}")

    @pyqtSlot()
    def on_pushButton_6_clicked(self):
        path = self.lineEdit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "错误", "文件不存在！")
            return
        try:
            self.diag_data = pd.read_excel(path)
            self.log_info("✅ 数据加载成功！")
            self.log_info(f"数据长度：{len(self.diag_data)} 行")
        except Exception as e:
            self.log_info(f"❌ 加载失败：{str(e)}")

    def get_selected_model(self):
        model_name = self.comboBox.currentText()
        if "BiLSTM" in model_name:
            return model_BiLSTM
        elif "LSTM" in model_name:
            return model_LSTM
        elif "Transformer" in model_name:
            return model_Transformer
        elif "MLP" in model_name:
            return model_MLP
        else:
            return model_LSTM

    @pyqtSlot()
    def on_pushButton_8_clicked(self):
        if self.diag_data is None:
            QMessageBox.warning(self, "提示", "请先加载数据！")
            return
        self.log_info("🔍 正在后台进行故障预报...")
        try:
            model_choice = self.get_selected_model()
            self.log_info(f"📌 使用模型：{self.comboBox.currentText()}")
            current_step = int(self.current_step.text())
            label = int(self.diag_data.iloc[:, -1].mode()[0])
            file_idx = label
            self.log_info(f"📌 数据标签：{label} → 加载权重 data_{file_idx+1}")
            self.predict_thread = PredictThread(self.file_path, file_idx, model_choice, current_step)
            self.predict_thread.log_signal.connect(self.log_info)
            self.predict_thread.done_signal.connect(self.on_predict_done)
            self.predict_thread.error_signal.connect(self.on_predict_error)
            self.predict_thread.start()
        except Exception as e:
            self.log_info(f"❌ 预报失败：{str(e)}")

    def on_predict_done(self, gif_path):
        self.textBrowser.clear()
        self.textBrowser.setHtml(f'<img src="{gif_path}?t={np.random.rand()}" width="680">')
        self.log_info("✅ 故障预报完成！动图已显示")

    def on_predict_error(self, err):
        self.log_info(f"❌ 预报失败：{err}")
        QMessageBox.warning(self, "失败", err)

    # 生成故障报告
    @pyqtSlot()
    def on_pushButton_9_clicked(self):
        if self.diag_data is None:
            QMessageBox.warning(self, "提示", "请先加载数据！")
            return
        self.log_info("📝 开始生成故障诊断报告...")
        self.xunfei_thread = XunFeiReportThread(self.diag_data)
        self.xunfei_thread.log_signal.connect(self.log_info)
        self.xunfei_thread.done_signal.connect(self.on_report_done)
        self.xunfei_thread.error_signal.connect(self.on_report_error)
        self.xunfei_thread.start()

    def on_report_done(self, report):
        self.log_info("✅ 故障报告生成完成！")
        self.textBrowser_2.setPlainText(report)  # 报告不影响动图

    def on_report_error(self, err):
        self.log_info(f"❌ 报告生成失败：{err}")

    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        try:
            wst = float(self.wst.text())
            fcw = float(self.fcw.text())
            fhw = float(self.fhw.text())
            T_input = float(self.T_input.text())
            fault_type = 0
            if self.radioButton.isChecked():
                fault_type = 1
            elif self.radioButton_2.isChecked():
                fault_type = 2
            elif self.radioButton_3.isChecked():
                fault_type = 3
            elif self.radioButton_4.isChecked():
                fault_type = 4
            self.fault_type = fault_type
            self.log_info("【仿真开始】")
            t, T, Wst, V = csth_simulation(wst, fcw, fhw, T_input, fault_type)
            self.last_t = t
            self.last_T = T
            self.last_Wst = Wst
            self.last_V = V
            plt.figure(figsize=(6,4))
            plt.plot(t, T, 'b-', linewidth=1.5)
            plt.axhline(T_input, color='r', linestyle='--')
            plt.axvline(100, color='gray', linestyle=':')
            plt.xlabel("时间(s)")
            plt.ylabel("温度(℃)")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig("sim_result.png", dpi=120)
            plt.close()
            self.textEdit_2.setHtml('<img src="sim_result.png" width="550">')
            self.log_info(f"✅ 仿真完成！最终温度：{T[-1]:.2f}℃")
        except Exception as e:
            self.log_info(f"❌ 错误：{str(e)}")

    @pyqtSlot()
    def on_pushButton_clicked(self):
        self.textEdit_2.clear()
        self.textBrowser.clear()
        self.textBrowser_2.clear()
        self.log_info("已清空")
    @pyqtSlot()
    def on_pushButton_10_clicked(self):
        self.textEdit_2.clear()
        self.textBrowser.clear()
        self.textBrowser_2.clear()
        self.log_info("已清空")

    @pyqtSlot()
    def on_pushButton_4_clicked(self):
        if not os.path.exists("sim_result.png"):
            QMessageBox.warning(self, "提示", "请先仿真")
            return
        p, _ = QFileDialog.getSaveFileName(self, "保存图片", "仿真图.png", "*.png")
        if p:
            with open("sim_result.png","rb") as f:
                open(p,"wb").write(f.read())
            self.log_info("🖼️ 图片已保存")

    @pyqtSlot()
    def on_pushButton_5_clicked(self):
        if self.last_t is None:
            QMessageBox.warning(self, "提示", "请先仿真")
            return
        p, _ = QFileDialog.getSaveFileName(self, "保存Excel", "数据.xlsx", "*.xlsx")
        if p:
            pd.DataFrame({
                "温度":self.last_T,
                "加热量":self.last_Wst,"体积":self.last_V,"标签":self.fault_type
            }).to_excel(p, index=False)
            self.log_info("📊 Excel已保存")

    def on_wst_textChanged(self, p0): pass
    def on_fcw_textChanged(self, p0): pass
    def on_fhw_textChanged(self, p0): pass
    def on_fout_textChanged(self, p0): pass
    def on_T_input_textEdited(self, p0): pass

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = MainWindow2()
    ui.show()
    sys.exit(app.exec())