# Faster-Whisper Server Setup လမ်းညွှန်ချက်

ဒါက faster-whisper ကို သုံးပြီး real-time audio transcription server ကို setup လုပ်ဖို့ လမ်းညွှန်ချက်ဖြစ်ပါတယ်။ Faster-whisper က OpenAI Whisper ကို ပိုမြန်အောင် optimized လုပ်ထားတာဖြစ်ပြီး၊ GPU ကိုသုံးပြီး real-time transcription လုပ်နိုင်ပါတယ်။

## လိုအပ်ချက်တွေ

- Python 3.11
- NVIDIA GPU (CUDA support)
- Microphone
- Internet connection (model download အတွက်)

## အဆင့် ၁: Environment Setup

```bash
# Folder အသစ်ဆောက်ပြီး ထဲဝင်မယ်
mkdir faster-whisper && cd faster_whisper
```

Python 3.11 ရှိမရှိ စစ်ပါ။ (မရှိရင် `sudo apt install python3.11-venv` နဲ့ အရင်သွင်းရပါမယ်)။ ရှိရင် အောက်ပါအတိုင်း run ပါ။
```bash
# Python 3.11 environment ဆောက်မယ်
python3.11 -m venv venv
source venv/bin/activate
```

## အဆင့် ၂: Packages သွင်းခြင်း

NVIDIA GPU Libraries နဲ့ အသံဖမ်းဖို့ PyAudio ကို သွင်းပါ:
```bash
pip install --upgrade pip
pip install fastapi uvicorn websockets numpy faster-whisper
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 pyaudio
```

LD_LIBRARY_PATH ကို set လုပ်ပါ:
```bash
export LD_LIBRARY_PATH=$(python3 -c 'import nvidia.cublas as cb; import nvidia.cudnn as cn; print(f"{cb.__path__[0]}/lib:{cn.__path__[0]}/lib")'):$LD_LIBRARY_PATH
```

## အဆင့် ၄: Run လုပ်ခြင်း

အရင်ဆုံး server ကို GPU ရှိတဲ့ laptop မှာ run လုပ်ပါ (terminal အသစ်မှာ):
```bash
python server.py
```

ပြီးရင် client ကို run လုပ်ပါ (terminal အသစ်မှာ):
```bash
python client.py
```

ဒါဆိုရင် microphone ကနေ အသံဖမ်းပြီး real-time transcription လုပ်ပြီး ပြပေးမှာဖြစ်ပါတယ်။ Ctrl+C နဲ့ ရပ်နိုင်ပါတယ်။

## မှတ်ချက်တွေ

- Model ကို ပထမဆုံး run တုန်းက download လုပ်မှာဖြစ်ပြီး အချိန်ယူပါလိမ့်မယ်။
- GPU မရှိရင် DEVICE = "cpu" ပြောင်းနိုင်ပါတယ်၊ ဒါပေမယ့် နှေးပါလိမ့်မယ်။
- Language ကို ပြောင်းချင်ရင် language="my" လို့ ပြောင်းနိုင်ပါတယ်။