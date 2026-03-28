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

## 🛠️ Setup (Conda Environment)

### ၁။ GPU ရှိသော Laptop (Server)

1. **Environment ဆောက်ပါ**:
   ```bash
   conda env create -f environment.yml
   conda activate faster-whisper
   ```
2. **GPU Library များ သတ်မှတ်ပါ**:
   - **Linux**: 
     ```bash
     export LD_LIBRARY_PATH=$(python -c 'import nvidia.cublas as cb; import nvidia.cudnn as cn; print(f"{cb.__path__[0]}/lib:{cn.__path__[0]}/lib")'):$LD_LIBRARY_PATH
     ```
   - **Windows**: `run_server.py` က auto-detect လုပ်ပေးမှာဖြစ်လို့ ဘာမှလုပ်စရာမလိုပါ။ (`nvidia-*` packages တွေ သွင်းထားဖို့ပဲလိုပါတယ်)
3. **Server ကို Start လုပ်ပါ**:
   ```bash
   python run_server.py
   ```

### ၂။ CPU ပဲရှိသော Laptop (Client)

1. **Environment ဆောက်ပါ**:
   ```bash
   # Conda နဲ့ environment သစ်ဆောက်မယ်
   conda create -n whisper-client python=3.10
   conda activate whisper-client
   ```
2. **လိုအပ်သော Library များသွင်းပါ**:
   ```bash
   # Linux only: sudo apt install libasound2-dev portaudio19-dev
   pip install pyaudio websockets
   ```
3. **Server ဆီ လှမ်းချိတ်ပါ**:
   ```bash
   python run_client.py --host <SERVER_IP> --language en
   ```

---

## ⚙️ Operating System Notes

#### **Windows အသုံးပြုသူများအတွက်**:
- **PyAudio**: `pip install pyaudio` က error တက်ခဲ့ရင် [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) ဒါမှမဟုတ် `pip install pipwin && pipwin install pyaudio` ကို သုံးနိုင်ပါတယ်။
- **CUDA**: `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` packages တွေကို သွင်းထားရင် `run_server.py` က DLL တွေကို အလိုအလျောက် ရှာပေးမှာဖြစ်ပါတယ်။

#### **Linux အသုံးပြုသူများအတွက်**:
- **Audio Dependencies**: PyAudio မသွင်းခင် `sudo apt install libasound2-dev portaudio19-dev` ကို အရင်သွင်းပေးရပါမယ်။
- **LD_LIBRARY_PATH**: GPU သုံးမယ်ဆိုရင် `export` command ကို Terminal တိုင်းမှာ ဒါမှမဟုတ် `.bashrc` ထဲမှာ ထည့်ထားပေးရပါမယ်။

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