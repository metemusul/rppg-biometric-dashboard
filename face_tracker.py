"""
face_tracker.py
---------------
Yüz ve alın (forehead) bölgesini tespit eden sınıf.

Bu implementasyon OpenCV'nin yerleşik Haar Cascade dedektörünü kullanır.
MediaPipe binary uyumsuzluklarına karşı dayanıklı, saf OpenCV tabanlı
çözüm.

ROI stratejisi:
  - Yüz bbox tespit edildiğinde alın bölgesi yaklaşık olarak
    yüzün üst 1/5'i + orta bölümünden alınır.
  - Bu bölgeden yeşil kanal ortalaması elde edilir.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class FaceTracker:
    """
    OpenCV Haar Cascade tabanlı yüz tespiti ve alın ROI çıkarımı.

    MediaPipe yerine OpenCV kullandığı için kurulum bağımlılığı
    minimumdur; opencv-python paketi yeterlidir.
    """

    # Haar Cascade yolu — proje köküne kopyalanmış versiyon kullanılır.
    # (Türkçe karakterli dizin adları OpenCV C++ backend'ini etkileyebileceğinden
    #  cv2.data.haarcascades yerine yerel kopya tercih edilir.)
    CASCADE_PATH: str = "haarcascade_frontalface_default.xml"

    # Alın bölgesi oranları (yüz bbox içinde)
    # y ekseni: yüzün tepesinden %10 - %40 arası
    # x ekseni: yüzün %20 - %80 arası (kenarları kes)
    FOREHEAD_Y_TOP_RATIO:    float = 0.10
    FOREHEAD_Y_BOTTOM_RATIO: float = 0.38
    FOREHEAD_X_LEFT_RATIO:   float = 0.20
    FOREHEAD_X_RIGHT_RATIO:  float = 0.80

    def __init__(self, max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):

        self._cascade = cv2.CascadeClassifier(self.CASCADE_PATH)
        if self._cascade.empty():
            raise RuntimeError(
                f"Haar Cascade dosyası yüklenemedi: {self.CASCADE_PATH}\n"
                "opencv-python kurulumunu kontrol edin."
            )

        self._max_faces      = max_num_faces
        self._scale_factor   = 1.15
        self._min_neighbors  = 4
        self._min_size       = (80, 80)

        # Son tespit edilen kare verileri
        self.face_detected   = False
        self.forehead_roi: Optional[Tuple[int,int,int,int]] = None
        self.face_bbox:    Optional[Tuple[int,int,int,int]] = None

        # Stabilizasyon için geçmiş bbox'ı tut
        self._prev_bbox: Optional[Tuple[int,int,int,int]] = None
        self._no_face_count: int = 0
        self._SMOOTHING_ALPHA: float = 0.35   # EMA katsayısı

    def process_frame(self, frame: np.ndarray):
        """
        Kareyi işler, yüz tespiti yapar ve alın ROI'sini çıkarır.

        Args:
            frame: BGR formatında OpenCV karesi

        Returns:
            tuple: (annotated_frame, mean_green_value, face_detected)
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # Kontrast normalleştirme

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        annotated   = frame.copy()
        mean_green  = 0.0
        self.face_detected = False

        if len(faces) > 0:
            self.face_detected = True
            self._no_face_count = 0

            # En büyük yüzü al
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces_sorted[0]

            # EMA stabilizasyon
            if self._prev_bbox is not None:
                a = self._SMOOTHING_ALPHA
                px, py, pw, ph = self._prev_bbox
                fx = int(a * fx + (1 - a) * px)
                fy = int(a * fy + (1 - a) * py)
                fw = int(a * fw + (1 - a) * pw)
                fh = int(a * fh + (1 - a) * ph)

            self._prev_bbox = (fx, fy, fw, fh)
            self.face_bbox  = (fx, fy, fx + fw, fy + fh)

            # ── Alın ROI hesaplama ─────────────────────────────────────
            rx1 = int(fx + fw * self.FOREHEAD_X_LEFT_RATIO)
            rx2 = int(fx + fw * self.FOREHEAD_X_RIGHT_RATIO)
            ry1 = int(fy + fh * self.FOREHEAD_Y_TOP_RATIO)
            ry2 = int(fy + fh * self.FOREHEAD_Y_BOTTOM_RATIO)

            # Sınır kontrolü
            rx1 = max(0, rx1); ry1 = max(0, ry1)
            rx2 = min(w, rx2); ry2 = min(h, ry2)

            self.forehead_roi = (rx1, ry1, rx2, ry2)

            # Yeşil kanal ortalaması
            if rx2 > rx1 and ry2 > ry1:
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size > 0:
                    mean_green = float(np.mean(roi[:, :, 1]))

            # Overlay çizimi
            self._draw_overlay(annotated, fx, fy, fw, fh, rx1, ry1, rx2, ry2)

        else:
            self._no_face_count += 1
            # 10 kare tespit edilemezse sıfırla
            if self._no_face_count > 10:
                self.forehead_roi = None
                self.face_bbox    = None
                self._prev_bbox   = None

        return annotated, mean_green, self.face_detected

    def _draw_overlay(self, frame: np.ndarray,
                      fx: int, fy: int, fw: int, fh: int,
                      rx1: int, ry1: int, rx2: int, ry2: int):
        """
        Kurumsal çizgi stilinde yüz ve alın ROI çerçevesini çizer.
        """
        # ── Alın ROI (neon yeşil köşe bracket) ──────────────────────────
        color_roi  = (0, 255, 150)   # Neon yeşil
        thickness  = 2
        corner_len = 16

        pts = [(rx1, ry1), (rx2, ry1), (rx1, ry2), (rx2, ry2)]
        dirs = [(1,1), (-1,1), (1,-1), (-1,-1)]

        for (cx, cy), (dx, dy) in zip(pts, dirs):
            cv2.line(frame, (cx, cy), (cx + dx*corner_len, cy), color_roi, thickness)
            cv2.line(frame, (cx, cy), (cx, cy + dy*corner_len), color_roi, thickness)

        # Hafif dolgu overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 180, 80), -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        cv2.putText(frame, "FOREHEAD ROI", (rx1, ry1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_roi, 1, cv2.LINE_AA)

        # ── Yüz bbox (mavi köşe bracket) ─────────────────────────────────
        face_color = (80, 180, 255)   # Açık mavi
        cl = 20

        cv2.line(frame, (fx, fy), (fx + cl, fy), face_color, 1)
        cv2.line(frame, (fx, fy), (fx, fy + cl), face_color, 1)
        cv2.line(frame, (fx+fw, fy), (fx+fw-cl, fy), face_color, 1)
        cv2.line(frame, (fx+fw, fy), (fx+fw, fy+cl), face_color, 1)
        cv2.line(frame, (fx, fy+fh), (fx+cl, fy+fh), face_color, 1)
        cv2.line(frame, (fx, fy+fh), (fx, fy+fh-cl), face_color, 1)
        cv2.line(frame, (fx+fw, fy+fh), (fx+fw-cl, fy+fh), face_color, 1)
        cv2.line(frame, (fx+fw, fy+fh), (fx+fw, fy+fh-cl), face_color, 1)

        cv2.putText(frame, "FACE DETECTED", (fx, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, face_color, 1, cv2.LINE_AA)

    def release(self):
        """Kaynakları serbest bırakır (Haar için ekstra işlem gerekmez)."""
        pass
