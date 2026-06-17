"""
signal_processor.py
-------------------
Yeşil renk kanalı verilerini ve zaman damgalarını depolayan,
en az 10 saniyelik veriyi tutan FIFO tampon sınıfı.
"""

import time
import numpy as np
from collections import deque
from typing import Optional, Tuple


class SignalProcessor:
    """
    rPPG sinyal tampon (buffer) yöneticisi.

    - Gelen her karedeki ortalama yeşil kanal değerini ve
      zaman damgasını depolar.
    - MAX_BUFFER_SECONDS: Tutulacak maksimum veri süresi (saniye).
    - MIN_BUFFER_SECONDS: Hesaplama yapılabilmesi için gereken minimum süre.
    - Veri FIFO (First-In-First-Out) mantığı ile döner pencerede saklanır.
    """

    MAX_BUFFER_SECONDS: float = 30.0   # 30 saniyelik döner pencere
    MIN_BUFFER_SECONDS: float = 10.0   # Kalibrasyon için gereken minimum süre
    EXPECTED_FPS: float = 30.0         # Beklenen kare hızı (fps tahmini için)

    def __init__(self):
        # Ham yeşil kanal değerleri
        self._green_buffer: deque = deque()
        # Her değere karşılık gelen Unix zaman damgası (saniye)
        self._timestamp_buffer: deque = deque()

        self._start_time: Optional[float] = None
        self._frame_count: int = 0

        # Son hesaplanan FPS
        self._fps: float = self.EXPECTED_FPS

    # ─────────────────────────────────────────────────────────────────────
    # Genel Özellikler
    # ─────────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """
        Tamponun hesaplama yapılabilecek minimum süreye ulaşıp ulaşmadığını
        döndürür.
        """
        if len(self._timestamp_buffer) < 2:
            return False
        duration = self._timestamp_buffer[-1] - self._timestamp_buffer[0]
        return duration >= self.MIN_BUFFER_SECONDS

    @property
    def buffer_duration(self) -> float:
        """Tamponda tutulan toplam veri süresini (saniye) döndürür."""
        if len(self._timestamp_buffer) < 2:
            return 0.0
        return self._timestamp_buffer[-1] - self._timestamp_buffer[0]

    @property
    def buffer_size(self) -> int:
        """Tamponda kaç örnek (sample) bulunduğunu döndürür."""
        return len(self._green_buffer)

    @property
    def estimated_fps(self) -> float:
        """Anlık FPS tahminini döndürür."""
        return self._fps

    # ─────────────────────────────────────────────────────────────────────
    # Veri Ekleme
    # ─────────────────────────────────────────────────────────────────────

    def add_sample(self, green_value: float, timestamp: Optional[float] = None):
        """
        Tampona yeni bir yeşil kanal örneği ekler.

        Args:
            green_value: Ortalama yeşil kanal değeri (0-255 arası float).
            timestamp:   Unix zaman damgası. None ise time.time() kullanılır.
        """
        if timestamp is None:
            timestamp = time.time()

        # İlk örnek için başlangıç zamanını kaydet
        if self._start_time is None:
            self._start_time = timestamp

        self._green_buffer.append(green_value)
        self._timestamp_buffer.append(timestamp)
        self._frame_count += 1

        # Eski verileri temizle (döner pencere)
        self._prune_old_samples(timestamp)

        # FPS güncelle (son 60 örnek üzerinden)
        self._update_fps()

    def _prune_old_samples(self, current_time: float):
        """MAX_BUFFER_SECONDS süresi dışındaki eski örnekleri kaldırır."""
        cutoff = current_time - self.MAX_BUFFER_SECONDS
        while (self._timestamp_buffer and
               self._timestamp_buffer[0] < cutoff):
            self._timestamp_buffer.popleft()
            self._green_buffer.popleft()

    def _update_fps(self):
        """Son örnekler üzerinden FPS tahminini günceller."""
        n = len(self._timestamp_buffer)
        if n >= 10:
            dt = self._timestamp_buffer[-1] - self._timestamp_buffer[-10]
            if dt > 0:
                self._fps = 9.0 / dt  # 10 örnek / dt saniye

    # ─────────────────────────────────────────────────────────────────────
    # Veri Okuma
    # ─────────────────────────────────────────────────────────────────────

    def get_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tamponda mevcut tüm yeşil kanal değerlerini ve zaman damgalarını
        numpy dizileri olarak döndürür.

        Returns:
            (green_array, timestamps_array): Her ikisi de float64 numpy dizisi.
        """
        if not self._green_buffer:
            return np.array([]), np.array([])

        return (np.array(self._green_buffer, dtype=np.float64),
                np.array(self._timestamp_buffer, dtype=np.float64))

    def get_normalized_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Yeşil kanal sinyalini ortalama çıkarılmış (mean-centered) ve
        standart sapmaya bölünmüş (z-score normalize) olarak döndürür.

        Returns:
            (normalized_signal, timestamps): Float64 numpy dizileri.
        """
        signal, timestamps = self.get_signal()
        if signal.size < 2:
            return signal, timestamps

        mean = np.mean(signal)
        std = np.std(signal)

        if std < 1e-9:
            return signal - mean, timestamps

        return (signal - mean) / std, timestamps

    def get_recent_signal(self, seconds: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Son `seconds` saniyedeki sinyal verilerini döndürür.

        Args:
            seconds: Kaç saniyelik veri alınacağı.

        Returns:
            (signal, timestamps): Float64 numpy dizileri.
        """
        signal, timestamps = self.get_signal()
        if timestamps.size == 0:
            return signal, timestamps

        cutoff = timestamps[-1] - seconds
        mask = timestamps >= cutoff
        return signal[mask], timestamps[mask]

    # ─────────────────────────────────────────────────────────────────────
    # Sıfırlama
    # ─────────────────────────────────────────────────────────────────────

    def reset(self):
        """Tamponu tamamen sıfırlar."""
        self._green_buffer.clear()
        self._timestamp_buffer.clear()
        self._start_time = None
        self._frame_count = 0
        self._fps = self.EXPECTED_FPS

    # ─────────────────────────────────────────────────────────────────────
    # Debug / Durum
    # ─────────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Tampon durum bilgilerini sözlük olarak döndürür."""
        return {
            "buffer_size": self.buffer_size,
            "duration_seconds": round(self.buffer_duration, 2),
            "is_ready": self.is_ready,
            "estimated_fps": round(self._fps, 2),
            "frame_count_total": self._frame_count
        }
