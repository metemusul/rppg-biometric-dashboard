# rPPG Biometric Dashboard

> Gerçek zamanlı, temassız çok-biyometrik analiz sistemi.
> **Python · Flask · OpenCV · MediaPipe · SciPy**

---

## 🚀 Kurulum & Çalıştırma

```bash
# 1. Sanal ortam oluştur ve aktive et
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Sunucuyu başlat
python app.py
```

Tarayıcıdan `http://127.0.0.1:5000` adresini açın.

---

## 📐 Proje Mimarisi

```
rppg_biometric_dashboard/
├── app.py                   # Flask sunucusu + API rotaları
├── face_tracker.py          # MediaPipe Face Mesh → alın ROI
├── signal_processor.py      # FIFO tampon (30 sn döner pencere)
├── biometric_calculator.py  # DSP pipeline (BPM, HRV, RR, Stres)
├── requirements.txt
├── templates/
│   └── index.html           # Ana dashboard HTML
└── static/
    ├── css/
    │   └── dashboard.css    # Dark mode glassmorphism tasarım
    └── js/
        └── dashboard.js     # DOM güncellemeleri + polling
```

---

## 🔬 Algoritmalar

| Metrik         | Yöntem                                              |
|----------------|-----------------------------------------------------|
| **BPM**        | Alın bölgesi yeşil kanal → Butterworth BP (0.8–3Hz) → FFT → dominant frekans × 60 |
| **HRV**        | BP filtrelenmiş sinyal → `scipy.find_peaks` → IBI farkları → RMSSD (ms) |
| **Solunum (RR)**| Butterworth BP (0.1–0.5 Hz) → FFT → dominant frekans × 60 |
| **Stres**      | RMSSD > 50ms → Düşük, 25–50ms → Normal, < 25ms → Yüksek |

---

## 🎨 Tasarım Özellikleri

- **Dark Mode**: `#0d1117` arka plan, kurumsal mavi/yeşil vurgular
- **Glassmorphism**: `backdrop-filter: blur` + yarı saydam kart arka planları
- **Animasyonlar**: BPM'e senkronize kalp atış halkası, tarama çizgisi, stres renk geçişleri
- **Tipografi**: Roboto Mono (değerler) + Inter (arayüz metni)
- **Responsive**: 1100px ve 760px kırılma noktaları

---

## 🛑 Önemli Notlar

- İlk **10 saniye** kalibrasyon süresidir → metriklerde `"Kalibre Ediliyor..."` görünür.
- **İyi aydınlatma** şarttır. Direkt güneş ışığı ya da ampul ışığı altında çalışın.
- Kafanızı sabit tutun ve alın bölgesini kameraya açık bırakın.
- `debug=False` ve `use_reloader=False` kullanılmıştır; Flask'ın çift kamera başlatma sorununu önler.
