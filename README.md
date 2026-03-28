# 🚀 Faster-Whisper Real-Time STT

ဒီ Project က `GPU` ရှိတဲ့ Laptop မှာ Server ထောင်ပြီး၊ `CPU` ပဲရှိတဲ့ Laptop တွေကနေ အသံဖမ်းပြီး Transcribe လုပ်နိုင်အောင် ပြုလုပ်ထားတာဖြစ်ပါတယ်။ 


---

## 💻 Project Structure

```text
faster-whisper/
├── src/
│   ├── core/           # Configuration and Logging
│   ├── server/         # Transcription Engine and WebSocket Handler
│   └── client/         # Audio Recording and Communication
├── models/             # Downloaded Whisper models (automatically created)
├── run_server.py       # Main entry point (GPU Laptop မှာ run ရမယ်)
├── run_client.py       # Main entry point (CPU Laptop မှာ run ရမယ်)
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

---

## 🛠️ Setup (GPU + CPU Setup)

### ၁။ GPU ရှိသော Laptop (Server)
အသံတွေကို အချိန်နဲ့တပြေးညီ (Real-time) ပြောင်းလဲပေးဖို့ Server ကို GPU ရှိတဲ့ Laptop မှာ run ရပါမယ်။

1. **Environment ဆောက်ပါ**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. **Dependencies သွင်းပါ**:
   ```bash
   pip install -r requirements.txt
   ```
3. **GPU Library များ သတ်မှတ်ပါ**:
   ```bash
   export LD_LIBRARY_PATH=$(python3 -c 'import nvidia.cublas as cb; import nvidia.cudnn as cn; print(f"{cb.__path__[0]}/lib:{cn.__path__[0]}/lib")'):$LD_LIBRARY_PATH
   ```
4. **Server ကို Start လုပ်ပါ**:
   ```bash
   python run_server.py
   ```
   *(မှတ်ချက်: `0.0.0.0` နဲ့ run နေမှာဖြစ်လို့ တစ်ခြား Laptop တွေကနေ လှမ်းချိတ်လို့ရပါတယ်)*

### ၂။ CPU ပဲရှိသော Laptop (Client)
ဒီ Laptop က မိုက်ကရိုဖုန်းကနေ အသံဖမ်းပြီး Server ဆီ ပို့ပေးမှာဖြစ်ပါတယ်။ GPU ရှိစရာ မလိုပါ။

1. **Environment ဆောက်ပါ**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. **လိုအပ်သော Library များသွင်းပါ** (PyAudio နဲ့ WebSockets ပဲ လိုအပ်ပါတယ်):
   ```bash
   pip install pyaudio websockets
   ```
3. **Server ဆီ လှမ်းချိတ်ပါ**:
   Server ရဲ့ IP Address ကို အရင်ရှာပါ။ (Server Laptop မှာ `hostname -I` လို့ ရိုက်ကြည့်ပါ)။ ဥပမာ `192.168.1.10` ဖြစ်တယ်ဆိုရင်:
   ```bash
   python run_client.py --host 192.168.1.10 --language en
   ```

---

## ⚙️ Configuration
`src/core/config.py` ထဲမှာ default settings တွေကို ပြင်နိုင်ပါတယ်။

- **Server-side**: GPU မရှိရင် `WHISPER_DEVICE=cpu` လို့ ပေးပြီး run နိုင်ပါတယ်။ CPU မှာ ပိုမြန်အောင် `int8` compute type ကို auto-select လုပ်ပေးမှာဖြစ်ပါတယ်။
- **Client-side**: `--host` နဲ့ `--language` flag တွေကို သုံးပြီး စိတ်ကြိုက် ပြောင်းလဲနိုင်ပါတယ်။

---

## 💡 Quick Start Tips
- **Burmese Transcription**: `--language my` ကို သုံးပါ။
- **Performance**: GPU Laptop မှာ `distil-large-v3` က အရမ်းမြန်ပါတယ်။
- **Network**: Laptop နှစ်လုံးက WiFi တူတူ ချိတ်ထားဖို့ လိုပါတယ်။