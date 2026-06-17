"""
biometric_calculator.py
-----------------------
BPM, HRV ve Solunum Hızı algoritmalarını içeren sinyal işleme sınıfı.
SciPy ve NumPy tabanlı DSP pipeline.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.signal import find_peaks, butter, filtfilt
from typing import Dict, Optional, Tuple


class BiometricCalculator:
    """
    rPPG sinyali üzerinden biyometrik değerleri hesaplar.

    Metotlar:
        - calculate_bpm()       : Butterworth + FFT ile nabız (BPM)
        - calculate_hrv()       : RMSSD formülü ile HRV (ms)
        - calculate_respiration(): Düşük bant filtresi + FFT ile nefes hızı
        - calculate_stress()    : HRV tabanlı kategorik stres endeksi
        - calculate_all()       : Tüm metrikleri tek seferde hesaplar
    """

    # ── Bandpass filtre parametreleri ─────────────────────────────────────
    BPM_LOW_HZ:  float = 0.8    # ~48 BPM alt sınır
    BPM_HIGH_HZ: float = 3.0    # ~180 BPM üst sınır
    RR_LOW_HZ:   float = 0.1    # ~6 nefes/dakika alt sınır
    RR_HIGH_HZ:  float = 0.5    # ~30 nefes/dakika üst sınır
    FILTER_ORDER: int  = 4      # Butterworth filtre derecesi

    # ── Tepe tespiti (peak detection) parametreleri ───────────────────────
    PEAK_MIN_DISTANCE_SEC: float = 0.3   # İki tepe arası minimum süre (saniye)
    PEAK_PROMINENCE:       float = 0.3   # Tepe belirginliği (normalize sinyal)

    # ── Stres eşik değerleri (RMSSD ms cinsinden) ─────────────────────────
    STRESS_LOW_THRESHOLD:    float = 50.0   # RMSSD > 50 → Düşük stres
    STRESS_NORMAL_THRESHOLD: float = 25.0   # RMSSD 25-50 → Normal stres
    # RMSSD < 25 → Yüksek stres

    def __init__(self):
        self._last_bpm: Optional[float] = None
        self._last_hrv: Optional[float] = None
        self._last_rr: Optional[float] = None
        self._last_stress: Optional[str] = None

    # ─────────────────────────────────────────────────────────────────────
    # Yardımcı: Butterworth Bandpass Filtresi
    # ─────────────────────────────────────────────────────────────────────

    def _bandpass_filter(self, data: np.ndarray, low_hz: float,
                         high_hz: float, fs: float) -> np.ndarray:
        """
        Butterworth bandpass filtresi uygular.

        Args:
            data:    Ham sinyal dizisi.
            low_hz:  Alt kesme frekansı (Hz).
            high_hz: Üst kesme frekansı (Hz).
            fs:      Örnekleme frekansı (Hz / fps).

        Returns:
            Filtrelenmiş sinyal dizisi.
        """
        nyquist = fs / 2.0

        # Nyquist sınırı kontrolü
        low_norm  = max(0.001, min(low_hz  / nyquist, 0.999))
        high_norm = max(0.001, min(high_hz / nyquist, 0.999))

        if low_norm >= high_norm:
            return data

        try:
            b, a = butter(self.FILTER_ORDER, [low_norm, high_norm],
                          btype='bandpass', analog=False)
            # filtfilt: sıfır-faz (zero-phase) filtreleme
            filtered = filtfilt(b, a, data)
            return filtered
        except Exception:
            return data

    # ─────────────────────────────────────────────────────────────────────
    # Yardımcı: Dominant Frekans Bulma (FFT)
    # ─────────────────────────────────────────────────────────────────────

    def _dominant_frequency(self, filtered_signal: np.ndarray,
                             fs: float, low_hz: float,
                             high_hz: float) -> Optional[float]:
        """
        FFT ile dominant frekansı bulur.

        Args:
            filtered_signal: Bandpass filtreden geçmiş sinyal.
            fs:              Örnekleme frekansı.
            low_hz / high_hz: Arama bandı sınırları.

        Returns:
            Dominant frekans (Hz) veya None.
        """
        n = len(filtered_signal)
        if n < 8:
            return None

        # Pencere fonksiyonu (Hanning) — sızıntıyı (spectral leakage) azaltır
        window = np.hanning(n)
        windowed = filtered_signal * window

        # FFT
        fft_vals = np.abs(np.fft.rfft(windowed))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        # Band içindeki frekansları filtrele
        mask = (fft_freqs >= low_hz) & (fft_freqs <= high_hz)
        if not mask.any():
            return None

        band_freqs = fft_freqs[mask]
        band_vals  = fft_vals[mask]

        dominant_idx = np.argmax(band_vals)
        return float(band_freqs[dominant_idx])

    # ─────────────────────────────────────────────────────────────────────
    # BPM Hesaplama
    # ─────────────────────────────────────────────────────────────────────

    def calculate_bpm(self, signal: np.ndarray,
                      timestamps: np.ndarray) -> Optional[float]:
        """
        Butterworth bandpass + FFT pipeline ile BPM hesaplar.

        Args:
            signal:     Ham yeşil kanal sinyali.
            timestamps: Karşılık gelen zaman damgaları.

        Returns:
            Hesaplanan BPM değeri veya None.
        """
        if signal.size < 30:
            return None

        # Örnekleme frekansını timestamp'ten hesapla
        duration = timestamps[-1] - timestamps[0]
        if duration < 1.0:
            return None

        fs = (len(timestamps) - 1) / duration

        # Ortalama çıkar (DC bileşenini kaldır)
        detrended = signal - np.mean(signal)

        # Bandpass filtre (0.8 – 3.0 Hz)
        filtered = self._bandpass_filter(detrended, self.BPM_LOW_HZ,
                                         self.BPM_HIGH_HZ, fs)

        # Dominant frekans → BPM dönüşümü
        dominant_hz = self._dominant_frequency(filtered, fs,
                                               self.BPM_LOW_HZ, self.BPM_HIGH_HZ)
        if dominant_hz is None:
            return None

        bpm = dominant_hz * 60.0
        # Fizyolojik sınır kontrolü
        bpm = max(40.0, min(200.0, bpm))

        self._last_bpm = round(bpm, 1)
        return self._last_bpm

    # ─────────────────────────────────────────────────────────────────────
    # HRV (RMSSD) Hesaplama
    # ─────────────────────────────────────────────────────────────────────

    def calculate_hrv(self, signal: np.ndarray,
                      timestamps: np.ndarray) -> Optional[float]:
        """
        Filtrelenmiş rPPG sinyalindeki tepelerden IBI (Inter-Beat Interval)
        farklılıklarının RMSSD formülü ile HRV (ms) hesaplar.

        RMSSD = sqrt( mean( (IBI[n+1] - IBI[n])^2 ) )

        Args:
            signal:     Ham yeşil kanal sinyali.
            timestamps: Karşılık gelen zaman damgaları.

        Returns:
            RMSSD değeri (ms) veya None.
        """
        if signal.size < 30:
            return None

        duration = timestamps[-1] - timestamps[0]
        if duration < 1.0:
            return None

        fs = (len(timestamps) - 1) / duration

        # BPM bandı filtresi uygula
        detrended = signal - np.mean(signal)
        filtered  = self._bandpass_filter(detrended, self.BPM_LOW_HZ,
                                           self.BPM_HIGH_HZ, fs)

        # Normalize et (find_peaks için tutarlı threshold)
        if np.std(filtered) > 1e-9:
            normalized = (filtered - np.mean(filtered)) / np.std(filtered)
        else:
            return None

        # Minimum mesafe (örneklerde)
        min_dist_samples = max(1, int(self.PEAK_MIN_DISTANCE_SEC * fs))

        # Tepe tespiti
        peaks, properties = find_peaks(
            normalized,
            distance=min_dist_samples,
            prominence=self.PEAK_PROMINENCE
        )

        if len(peaks) < 3:
            return None

        # Tepe zaman damgaları → IBI (ms)
        peak_times_ms = timestamps[peaks] * 1000.0
        ibi_ms = np.diff(peak_times_ms)

        # Fizyolojik olarak anlamsız IBI değerlerini filtrele
        # Normal IBI: 333ms (180 BPM) ile 1500ms (40 BPM) arası
        valid_ibi = ibi_ms[(ibi_ms >= 333) & (ibi_ms <= 1500)]

        if len(valid_ibi) < 2:
            return None

        # RMSSD formülü
        successive_diff = np.diff(valid_ibi)
        rmssd = float(np.sqrt(np.mean(successive_diff ** 2)))

        # Fizyolojik sınır kontrolü (5ms – 200ms)
        rmssd = max(5.0, min(200.0, rmssd))

        self._last_hrv = round(rmssd, 1)
        return self._last_hrv

    # ─────────────────────────────────────────────────────────────────────
    # Solunum Hızı Hesaplama
    # ─────────────────────────────────────────────────────────────────────

    def calculate_respiration(self, signal: np.ndarray,
                               timestamps: np.ndarray) -> Optional[float]:
        """
        Aynı rPPG sinyalini 0.1–0.5 Hz bandında filtreleyerek
        dakikadaki nefes sayısını (RR, breaths/min) hesaplar.

        Args:
            signal:     Ham yeşil kanal sinyali.
            timestamps: Karşılık gelen zaman damgaları.

        Returns:
            Solunum hızı (nefes/dakika) veya None.
        """
        if signal.size < 30:
            return None

        duration = timestamps[-1] - timestamps[0]
        if duration < 1.0:
            return None

        fs = (len(timestamps) - 1) / duration

        # Ortalama çıkar
        detrended = signal - np.mean(signal)

        # Solunum bandı filtresi (0.1 – 0.5 Hz)
        filtered = self._bandpass_filter(detrended, self.RR_LOW_HZ,
                                         self.RR_HIGH_HZ, fs)

        # Dominant frekans
        dominant_hz = self._dominant_frequency(filtered, fs,
                                               self.RR_LOW_HZ, self.RR_HIGH_HZ)
        if dominant_hz is None:
            return None

        rr = dominant_hz * 60.0
        # Fizyolojik sınır (6–30 nefes/dakika)
        rr = max(6.0, min(30.0, rr))

        self._last_rr = round(rr, 1)
        return self._last_rr

    # ─────────────────────────────────────────────────────────────────────
    # Stres Endeksi Hesaplama
    # ─────────────────────────────────────────────────────────────────────

    def calculate_stress(self, hrv_rmssd: Optional[float]) -> str:
        """
        HRV (RMSSD) değeri üzerinden kategorik stres durumu türetir.

        HRV ↑  →  Otonom sinir sistemi sağlıklı  →  Stres ↓
        HRV ↓  →  Sempatik aktivasyon artmış      →  Stres ↑

        Args:
            hrv_rmssd: RMSSD değeri (ms). None ise "Kalibre Ediliyor..." döner.

        Returns:
            "Düşük" | "Normal" | "Yüksek" | "Kalibre Ediliyor..."
        """
        if hrv_rmssd is None:
            return "Kalibre Ediliyor..."

        if hrv_rmssd >= self.STRESS_LOW_THRESHOLD:
            stress = "Düşük"
        elif hrv_rmssd >= self.STRESS_NORMAL_THRESHOLD:
            stress = "Normal"
        else:
            stress = "Yüksek"

        self._last_stress = stress
        return stress

    # ─────────────────────────────────────────────────────────────────────
    # Tüm Metrikleri Hesapla
    # ─────────────────────────────────────────────────────────────────────

    def calculate_all(self, signal: np.ndarray,
                      timestamps: np.ndarray) -> Dict:
        """
        Tüm biyometrik metrikleri tek seferde hesaplar.

        Args:
            signal:     Ham yeşil kanal sinyali.
            timestamps: Karşılık gelen zaman damgaları.

        Returns:
            dict: {
                "bpm":    float | None,
                "hrv":    float | None,
                "rr":     float | None,
                "stress": str
            }
        """
        bpm = self.calculate_bpm(signal, timestamps)
        hrv = self.calculate_hrv(signal, timestamps)
        rr  = self.calculate_respiration(signal, timestamps)
        stress = self.calculate_stress(hrv)

        return {
            "bpm":    bpm,
            "hrv":    hrv,
            "rr":     rr,
            "stress": stress
        }

    # ─────────────────────────────────────────────────────────────────────
    # Filtreli Sinyal (EKG Görünümü için)
    # ─────────────────────────────────────────────────────────────────────

    def get_filtered_signal_for_display(self, signal: np.ndarray,
                                         timestamps: np.ndarray,
                                         n_points: int = 200) -> np.ndarray:
        """
        Görüntüleme (EKG poligonu) için ölçeklendirilmiş filtrelenmiş sinyal.

        Args:
            signal:     Ham sinyal.
            timestamps: Zaman damgaları.
            n_points:   Döndürülecek nokta sayısı.

        Returns:
            0-1 arasında normalize edilmiş numpy dizisi.
        """
        if signal.size < 10:
            return np.zeros(n_points)

        duration = timestamps[-1] - timestamps[0]
        if duration < 0.5:
            return np.zeros(n_points)

        fs = (len(timestamps) - 1) / duration
        detrended = signal - np.mean(signal)
        filtered  = self._bandpass_filter(detrended, self.BPM_LOW_HZ,
                                           self.BPM_HIGH_HZ, fs)

        # n_points'e yeniden örnekle (resample)
        if len(filtered) != n_points:
            filtered = sp_signal.resample(filtered, n_points)

        # 0-1 arasına normalize et
        sig_min = filtered.min()
        sig_max = filtered.max()
        if (sig_max - sig_min) < 1e-9:
            return np.full(n_points, 0.5)

        normalized = (filtered - sig_min) / (sig_max - sig_min)
        return normalized
