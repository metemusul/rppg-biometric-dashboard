"""
app.py
------
Flask web sunucusu ve API router.
- /           : Ana dashboard sayfası (index.html)
- /video_feed : MJPEG kamera akışı (OpenCV + EKG overlay)
- /metrics    : JSON biyometrik metrik API uç noktası
"""

import cv2
import numpy as np
import threading
import time
import logging

from flask import Flask, Response, render_template, jsonify

from face_tracker import FaceTracker
from signal_processor import SignalProcessor
from biometric_calculator import BiometricCalculator

# ── Loglama Ayarı ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rPPG-App")

# ── Flask Uygulaması ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0   # Statik dosyalar cache'lenmesin

# ── Global Durum (Thread-safe) ─────────────────────────────────────────────────
_lock = threading.Lock()

# Bileşen örnekleri
face_tracker   = FaceTracker(min_detection_confidence=0.5, min_tracking_confidence=0.5)
signal_proc    = SignalProcessor()
bio_calculator = BiometricCalculator()

# Paylaşılan metrik durumu
_current_metrics: dict = {
    "bpm":    None,
    "hrv":    None,
    "rr":     None,
    "stress": "Kalibre Ediliyor...",
    "is_ready":  False,
    "face_detected": False,
    "buffer_progress": 0.0   # 0.0 – 1.0 (kalibrasyon ilerleme oranı)
}

# Kamera ve kare
_camera: cv2.VideoCapture = None
_latest_frame: np.ndarray = None
_frame_available = threading.Event()

# EKG sinyal tamponu (overlay için)
_ekg_signal: np.ndarray = np.zeros(200)

# ── Kamera Başlatma ────────────────────────────────────────────────────────────

def _init_camera():
    """Kamerayı başlatır; başarılı olana dek farklı index'leri dener."""
    global _camera
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # DirectShow (Windows)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            _camera = cap
            logger.info(f"Kamera açıldı: index={idx}")
            return True
    logger.error("Kamera bulunamadı!")
    return False

# ── Kamera İşleme Thread'i ────────────────────────────────────────────────────

def _camera_worker():
    """
    Arka planda kamerayi sürekli okur:
    1. FaceTracker ile yüz tespiti yapar.
    2. Yeşil kanal değerini SignalProcessor'a ekler.
    3. Tampon hazırsa BiometricCalculator ile metrikleri hesaplar.
    4. EKG overlay'i ile işlenmiş kareyi _latest_frame'e yazar.
    """
    global _latest_frame, _ekg_signal, _current_metrics

    METRIC_UPDATE_INTERVAL = 1.0   # Saniyede bir metrik hesapla
    last_metric_time = 0.0

    frame_counter = 0

    while True:
        if _camera is None or not _camera.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = _camera.read()
        if not ret:
            time.sleep(0.033)
            continue

        frame_counter += 1
        now = time.time()

        # ── Yüz Tespiti & Yeşil Kanal Çıkarımı ───────────────────────────
        annotated, green_val, face_ok = face_tracker.process_frame(frame)

        if face_ok and green_val > 0:
            signal_proc.add_sample(green_val, now)

        # ── Kalibrasyon İlerleme Oranı ─────────────────────────────────────
        buffer_dur   = signal_proc.buffer_duration
        min_buf      = signal_proc.MIN_BUFFER_SECONDS
        buf_progress = min(1.0, buffer_dur / min_buf)

        # ── Metrik Hesaplama (saniyede bir) ────────────────────────────────
        if now - last_metric_time >= METRIC_UPDATE_INTERVAL:
            last_metric_time = now

            if signal_proc.is_ready:
                sig, ts = signal_proc.get_signal()

                metrics = bio_calculator.calculate_all(sig, ts)
                ekg_sig = bio_calculator.get_filtered_signal_for_display(sig, ts)

                with _lock:
                    _current_metrics.update({
                        "bpm":            metrics["bpm"],
                        "hrv":            metrics["hrv"],
                        "rr":             metrics["rr"],
                        "stress":         metrics["stress"],
                        "is_ready":       True,
                        "face_detected":  face_ok,
                        "buffer_progress": buf_progress
                    })
                    _ekg_signal = ekg_sig
            else:
                with _lock:
                    _current_metrics.update({
                        "is_ready":       False,
                        "face_detected":  face_ok,
                        "buffer_progress": buf_progress
                    })

        # ── EKG Overlay Çizimi ─────────────────────────────────────────────
        with _lock:
            ekg_to_draw = _ekg_signal.copy()

        _draw_ekg_overlay(annotated, ekg_to_draw)

        # ── Durum Bilgi Çubuğu ─────────────────────────────────────────────
        _draw_status_bar(annotated, face_ok, buf_progress)

        # ── Kareyi Paylaş ──────────────────────────────────────────────────
        with _lock:
            _latest_frame = annotated.copy()
        _frame_available.set()


def _draw_ekg_overlay(frame: np.ndarray, ekg_signal: np.ndarray):
    """
    Kare üzerine yeşil EKG poligonu çizer (sol alt köşeye).

    Args:
        frame:      BGR OpenCV karesi.
        ekg_signal: 0-1 normalize edilmiş EKG sinyal dizisi.
    """
    h, w = frame.shape[:2]

    # EKG kutu boyutu ve konumu
    box_w = int(w * 0.42)
    box_h = 70
    box_x = 10
    box_y = h - box_h - 10

    # Yarı saydam arka plan
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h),
                  (10, 20, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Kenarlık
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                  (0, 100, 50), 1)

    # Etiket
    cv2.putText(frame, "rPPG SIGNAL", (box_x + 5, box_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 100), 1, cv2.LINE_AA)

    if ekg_signal.size < 2:
        return

    # EKG nokta koordinatları
    n = len(ekg_signal)
    padding = 8
    plot_w = box_w - 2 * padding
    plot_h = box_h - 28  # Etiket için alan bırak

    pts = []
    for i, val in enumerate(ekg_signal):
        x = box_x + padding + int(i * plot_w / max(n - 1, 1))
        y = (box_y + 24 + plot_h) - int(val * plot_h)
        y = max(box_y + 24, min(box_y + 24 + plot_h, y))
        pts.append([x, y])

    pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_array], isClosed=False,
                  color=(0, 255, 120), thickness=1, lineType=cv2.LINE_AA)


def _draw_status_bar(frame: np.ndarray, face_ok: bool, progress: float):
    """
    Kare üzerine durum çubuğu çizer (sağ alt köşe).
    """
    h, w = frame.shape[:2]
    bar_w = 160
    bar_h = 36
    bx = w - bar_w - 10
    by = h - bar_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bar_w, by + bar_h),
                  (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Yüz durumu
    face_color = (0, 220, 100) if face_ok else (0, 80, 220)
    face_label = "FACE: OK" if face_ok else "FACE: --"
    cv2.putText(frame, face_label, (bx + 8, by + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, face_color, 1, cv2.LINE_AA)

    # Kalibrasyon ilerleme çubuğu
    prog_w = bar_w - 16
    cv2.rectangle(frame, (bx + 8, by + 22), (bx + 8 + prog_w, by + 30),
                  (30, 40, 30), -1)
    filled = int(prog_w * progress)
    if filled > 0:
        pct_color = (0, 255, 120) if progress >= 1.0 else (0, 180, 255)
        cv2.rectangle(frame, (bx + 8, by + 22),
                      (bx + 8 + filled, by + 30), pct_color, -1)

    pct_text = f"CAL: {int(progress * 100)}%"
    cv2.putText(frame, pct_text, (bx + 8 + prog_w + 2, by + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA)


# ── MJPEG Frame Generator ─────────────────────────────────────────────────────

def _generate_mjpeg():
    """
    MJPEG akışı için frame üretici (generator).
    Her kareyi JPEG'e encode ederek multipart response olarak gönderir.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 82]

    while True:
        _frame_available.wait(timeout=0.5)
        _frame_available.clear()

        with _lock:
            frame = _latest_frame

        if frame is None:
            # Kamera hazır değilse bekleme karesi gönder
            placeholder = _make_placeholder_frame()
            ret, buf = cv2.imencode(".jpg", placeholder, encode_params)
        else:
            ret, buf = cv2.imencode(".jpg", frame, encode_params)

        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() +
               b"\r\n")


def _make_placeholder_frame() -> np.ndarray:
    """Kamera başlamadan önce gösterilecek siyah placeholder kare."""
    ph = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(ph, "Kamera Baslatiliyor...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 1, cv2.LINE_AA)
    return ph


# ── Flask Rotaları ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Ana dashboard sayfası."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """
    MJPEG video akışı.
    Kamera karelerini, yüz overlay ve EKG sinyali ile birlikte stream eder.
    """
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/metrics")
def metrics():
    """
    Anlık biyometrik metrik JSON API uç noktası.

    Response:
        {
            "bpm":            float | "Kalibre Ediliyor...",
            "hrv":            float | "Kalibre Ediliyor...",
            "rr":             float | "Kalibre Ediliyor...",
            "stress":         "Düşük" | "Normal" | "Yüksek" | "Kalibre Ediliyor...",
            "is_ready":       bool,
            "face_detected":  bool,
            "buffer_progress": float (0.0 – 1.0)
        }
    """
    CALIBRATING = "Kalibre Ediliyor..."

    with _lock:
        m = _current_metrics.copy()

    if not m["is_ready"]:
        response = {
            "bpm":    CALIBRATING,
            "hrv":    CALIBRATING,
            "rr":     CALIBRATING,
            "stress": CALIBRATING,
            "is_ready":       False,
            "face_detected":  m.get("face_detected", False),
            "buffer_progress": m.get("buffer_progress", 0.0)
        }
    else:
        response = {
            "bpm":    m["bpm"]  if m["bpm"]  is not None else CALIBRATING,
            "hrv":    m["hrv"]  if m["hrv"]  is not None else CALIBRATING,
            "rr":     m["rr"]   if m["rr"]   is not None else CALIBRATING,
            "stress": m["stress"],
            "is_ready":       True,
            "face_detected":  m.get("face_detected", False),
            "buffer_progress": m.get("buffer_progress", 1.0)
        }

    return jsonify(response)


# ── Uygulama Başlatma ──────────────────────────────────────────────────────────

def _startup():
    """Kamerayı başlatır ve worker thread'ini çalıştırır."""
    if _init_camera():
        worker = threading.Thread(target=_camera_worker, daemon=True, name="CameraWorker")
        worker.start()
        logger.info("Kamera worker thread başlatıldı.")
    else:
        logger.warning("Kamera başlatılamadı, placeholder modda çalışılıyor.")


if __name__ == "__main__":
    _startup()
    logger.info("Flask sunucusu başlatılıyor → http://127.0.0.1:5000")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,       # Debug=True ile kamera çift başlar (reloader sorunu)
        threaded=True,
        use_reloader=False
    )
