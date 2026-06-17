/**
 * dashboard.js  —  rPPG Biometric Dashboard
 * Gerçek zamanlı DOM güncellemeleri, metrik animasyonları ve
 * /metrics API endpoint polling.
 */

"use strict";

/* ══════════════════════════════════════════════════════════════════════
   1. YAPILANDIRMA
══════════════════════════════════════════════════════════════════════ */

const CONFIG = {
  /** /metrics endpoint polling aralığı (ms) */
  POLL_INTERVAL_MS: 1000,

  /** Footer saat güncelleme aralığı (ms) */
  CLOCK_INTERVAL_MS: 1000,

  /** BPM'den kalp atış animasyon BPM hesaplama */
  HEARTBEAT_DEFAULT_BPM: 75,

  /** Mini bar için aralık tanımları */
  RANGES: {
    bpm: { min: 40,  max: 200, normal_low: 60,  normal_high: 100 },
    hrv: { min: 5,   max: 100, normal_low: 20,  normal_high: 80  },
    rr:  { min: 6,   max: 30,  normal_low: 12,  normal_high: 20  },
  },

  CALIBRATING_TEXT: "Kalibre Ediliyor...",
};

/* ══════════════════════════════════════════════════════════════════════
   2. DOM ELEMENTLERİ
══════════════════════════════════════════════════════════════════════ */

const DOM = {
  // Kamera & Kalibrasyon
  calProgressFill:  document.getElementById("calProgressFill"),
  calProgressBar:   document.getElementById("calProgressBar"),
  calProgressLabel: document.getElementById("calProgressLabel"),
  calStatusText:    document.getElementById("calStatusText"),

  // Header Durum
  statusFace:       document.getElementById("statusFace"),
  dotFace:          document.getElementById("dotFace"),
  statusFaceText:   document.getElementById("statusFaceText"),
  statusCal:        document.getElementById("statusCal"),
  dotCal:           document.getElementById("dotCal"),
  statusCalText:    document.getElementById("statusCalText"),

  // BPM
  bpmValue:         document.getElementById("bpmValue"),
  bpmBar:           document.getElementById("bpmBar"),
  bpmTrend:         document.getElementById("bpmTrend"),
  bpmPulseRing:     document.getElementById("bpmPulseRing"),
  cardBpm:          document.getElementById("cardBpm"),

  // HRV
  hrvValue:         document.getElementById("hrvValue"),
  hrvBar:           document.getElementById("hrvBar"),
  cardHrv:          document.getElementById("cardHrv"),

  // Solunum Hızı
  rrValue:          document.getElementById("rrValue"),
  rrBar:            document.getElementById("rrBar"),
  cardRr:           document.getElementById("cardRr"),

  // Stres
  stressValue:      document.getElementById("stressValue"),
  cardStress:       document.getElementById("cardStress"),
  stressGlow:       document.getElementById("stressGlow"),
  stressIconWrap:   document.getElementById("stressIconWrap"),
  siLow:            document.getElementById("si-low"),
  siNormal:         document.getElementById("si-normal"),
  siHigh:           document.getElementById("si-high"),

  // Footer
  footerTime:       document.getElementById("footerTime"),
  updateTimer:      document.getElementById("updateTimer"),
};

/* ══════════════════════════════════════════════════════════════════════
   3. DURUM
══════════════════════════════════════════════════════════════════════ */

const state = {
  lastBpm:     null,
  prevBpm:     null,
  heartbeatInterval: null,
  lastUpdateTime:    null,
  errorCount:        0,
};

/* ══════════════════════════════════════════════════════════════════════
   4. YARDIMCI FONKSİYONLAR
══════════════════════════════════════════════════════════════════════ */

/**
 * Sayısal değeri belirli bir aralıkta yüzdeye dönüştürür.
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number} 0–100 arası
 */
function toPercent(value, min, max) {
  if (value === null || value === undefined || isNaN(value)) return 0;
  return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
}

/**
 * Anlık BPM değerinden kalp atış animasyon süresini hesaplar.
 * @param {number} bpm
 * @returns {number} ms cinsinden interval
 */
function bpmToInterval(bpm) {
  if (!bpm || bpm < 40 || bpm > 220) return 800;
  return Math.round(60000 / bpm);
}

/**
 * DOM element üzerinde "value-updated" pop animasyonu tetikler.
 * @param {HTMLElement} el
 */
function triggerValuePop(el) {
  if (!el) return;
  el.classList.remove("value-updated");
  // Reflow zorla
  void el.offsetWidth;
  el.classList.add("value-updated");
}

/**
 * Trendi hesaplar ve emoji/ok döndürür.
 * @param {number|null} current
 * @param {number|null} prev
 * @returns {string}
 */
function getTrendIcon(current, prev) {
  if (current === null || prev === null) return "";
  const diff = current - prev;
  if (Math.abs(diff) < 2)  return "→";
  if (diff > 0)             return "↑";
  return "↓";
}

/* ══════════════════════════════════════════════════════════════════════
   5. KALIBRASYON İLERLEME ÇUBUĞU
══════════════════════════════════════════════════════════════════════ */

/**
 * Kalibrasyon progress bar'ını günceller.
 * @param {number} progress  0.0 – 1.0
 * @param {boolean} isReady  Kalibrasyon tamamlandı mı?
 */
function updateCalibration(progress, isReady) {
  const pct = Math.round(progress * 100);

  // Progress bar genişliği
  DOM.calProgressFill.style.width = `${pct}%`;
  DOM.calProgressBar.setAttribute("aria-valuenow", pct);
  DOM.calProgressLabel.textContent = `${pct}%`;

  if (isReady) {
    DOM.calStatusText.textContent = "✓ Kalibrasyon Tamamlandı";
    DOM.calProgressFill.style.background =
      "linear-gradient(90deg, #39d353 0%, #7fff9a 100%)";

    // Header kalibrasyon pill
    DOM.statusCal.classList.add("active");
    DOM.statusCalText.textContent = "Aktif";
  } else {
    const remaining = Math.round((1 - progress) * 10);
    DOM.calStatusText.textContent =
      progress < 0.05
        ? "Yüz kameranıza doğru bakın..."
        : `Kalibre ediliyor... ~${remaining}s`;
    DOM.calProgressFill.style.background =
      "linear-gradient(90deg, #58a6ff 0%, #20d4c8 100%)";

    DOM.statusCal.classList.remove("active");
    DOM.statusCalText.textContent = "Kalibre Ediliyor";
  }
}

/* ══════════════════════════════════════════════════════════════════════
   6. YÜZÜ DURUM GÜNCELLEMESİ
══════════════════════════════════════════════════════════════════════ */

/**
 * @param {boolean} detected
 */
function updateFaceStatus(detected) {
  if (detected) {
    DOM.statusFace.classList.add("active");
    DOM.statusFaceText.textContent = "Yüz Tespit Edildi";
  } else {
    DOM.statusFace.classList.remove("active");
    DOM.statusFaceText.textContent = "Yüz Bekleniyor";
  }
}

/* ══════════════════════════════════════════════════════════════════════
   7. BPM GÜNCELLEMESİ
══════════════════════════════════════════════════════════════════════ */

/**
 * @param {number|string} bpm
 */
function updateBPM(bpm) {
  const isCalibrating = (bpm === CONFIG.CALIBRATING_TEXT || bpm === null);

  if (isCalibrating) {
    DOM.bpmValue.textContent = "—";
    DOM.bpmValue.classList.add("calibrating");
    DOM.bpmBar.style.width = "0%";
    DOM.bpmTrend.textContent = "";
    DOM.cardBpm.classList.remove("active");

    // Kalp atış animasyonunu durdur
    DOM.bpmPulseRing.classList.remove("active");
    if (state.heartbeatInterval) {
      clearInterval(state.heartbeatInterval);
      state.heartbeatInterval = null;
    }
    return;
  }

  const numBpm = parseFloat(bpm);
  DOM.bpmValue.classList.remove("calibrating");
  DOM.cardBpm.classList.add("active");

  // Değer güncelle (pop animasyonu ile)
  if (DOM.bpmValue.textContent !== String(Math.round(numBpm))) {
    DOM.bpmValue.textContent = Math.round(numBpm);
    triggerValuePop(DOM.bpmValue);
  }

  // Trend
  state.prevBpm = state.lastBpm;
  state.lastBpm = numBpm;
  DOM.bpmTrend.textContent = getTrendIcon(state.lastBpm, state.prevBpm);

  // Mini bar
  const { min, max } = CONFIG.RANGES.bpm;
  DOM.bpmBar.style.width = `${toPercent(numBpm, min, max)}%`;

  // Kalp atış animasyonu – BPM'e göre hız
  const interval = bpmToInterval(numBpm);
  DOM.bpmPulseRing.classList.add("active");

  if (state.heartbeatInterval) clearInterval(state.heartbeatInterval);
  state.heartbeatInterval = setInterval(() => {
    DOM.bpmPulseRing.style.animationDuration = `${interval}ms`;
  }, interval);
}

/* ══════════════════════════════════════════════════════════════════════
   8. HRV GÜNCELLEMESİ
══════════════════════════════════════════════════════════════════════ */

/**
 * @param {number|string} hrv
 */
function updateHRV(hrv) {
  const isCalibrating = (hrv === CONFIG.CALIBRATING_TEXT || hrv === null);

  if (isCalibrating) {
    DOM.hrvValue.textContent = "—";
    DOM.hrvValue.classList.add("calibrating");
    DOM.hrvBar.style.width = "0%";
    DOM.cardHrv.classList.remove("active");
    return;
  }

  const numHrv = parseFloat(hrv);
  DOM.hrvValue.classList.remove("calibrating");
  DOM.cardHrv.classList.add("active");

  if (DOM.hrvValue.textContent !== String(numHrv.toFixed(1))) {
    DOM.hrvValue.textContent = numHrv.toFixed(1);
    triggerValuePop(DOM.hrvValue);
  }

  const { min, max } = CONFIG.RANGES.hrv;
  DOM.hrvBar.style.width = `${toPercent(numHrv, min, max)}%`;
}

/* ══════════════════════════════════════════════════════════════════════
   9. SOLUNUM HIZI GÜNCELLEMESİ
══════════════════════════════════════════════════════════════════════ */

/**
 * @param {number|string} rr
 */
function updateRR(rr) {
  const isCalibrating = (rr === CONFIG.CALIBRATING_TEXT || rr === null);

  if (isCalibrating) {
    DOM.rrValue.textContent = "—";
    DOM.rrValue.classList.add("calibrating");
    DOM.rrBar.style.width = "0%";
    DOM.cardRr.classList.remove("active");
    return;
  }

  const numRr = parseFloat(rr);
  DOM.rrValue.classList.remove("calibrating");
  DOM.cardRr.classList.add("active");

  if (DOM.rrValue.textContent !== String(numRr.toFixed(1))) {
    DOM.rrValue.textContent = numRr.toFixed(1);
    triggerValuePop(DOM.rrValue);
  }

  const { min, max } = CONFIG.RANGES.rr;
  DOM.rrBar.style.width = `${toPercent(numRr, min, max)}%`;
}

/* ══════════════════════════════════════════════════════════════════════
   10. STRES ENDEKSİ GÜNCELLEMESİ
══════════════════════════════════════════════════════════════════════ */

/** Stres düzeyine göre renk paleti */
const STRESS_PALETTE = {
  "Düşük":  {
    color:  "#39d353",
    shadow: "0 0 24px rgba(57, 211, 83, 0.50)",
    glow:   "radial-gradient(ellipse at 50% 0%, rgba(57,211,83,0.35) 0%, transparent 70%)",
    icon:   "rgba(57, 211, 83, 0.12)",
    iconBorder: "rgba(57, 211, 83, 0.30)",
    activeClass: "active-low",
    activeId: "si-low",
  },
  "Normal": {
    color:  "#ffa657",
    shadow: "0 0 24px rgba(255, 166, 87, 0.45)",
    glow:   "radial-gradient(ellipse at 50% 0%, rgba(255,166,87,0.35) 0%, transparent 70%)",
    icon:   "rgba(255, 166, 87, 0.12)",
    iconBorder: "rgba(255, 166, 87, 0.30)",
    activeClass: "active-normal",
    activeId: "si-normal",
  },
  "Yüksek": {
    color:  "#ff6b6b",
    shadow: "0 0 28px rgba(255, 107, 107, 0.55)",
    glow:   "radial-gradient(ellipse at 50% 0%, rgba(255,107,107,0.40) 0%, transparent 70%)",
    icon:   "rgba(255, 107, 107, 0.12)",
    iconBorder: "rgba(255, 107, 107, 0.35)",
    activeClass: "active-high",
    activeId: "si-high",
  },
};

/**
 * @param {string} stress  "Düşük" | "Normal" | "Yüksek" | calibrating
 */
function updateStress(stress) {
  const isCalibrating = (
    stress === CONFIG.CALIBRATING_TEXT ||
    stress === null ||
    !(stress in STRESS_PALETTE)
  );

  // Tüm göstergelerden aktif sınıfı kaldır
  [DOM.siLow, DOM.siNormal, DOM.siHigh].forEach(el => {
    el.classList.remove("active-low", "active-normal", "active-high");
  });

  if (isCalibrating) {
    DOM.stressValue.textContent = "—";
    DOM.stressValue.classList.add("calibrating");
    DOM.cardStress.classList.remove("active");
    DOM.stressGlow.style.background = "";
    return;
  }

  const palette = STRESS_PALETTE[stress];

  DOM.stressValue.classList.remove("calibrating");
  DOM.cardStress.classList.add("active");

  if (DOM.stressValue.textContent !== stress) {
    DOM.stressValue.textContent = stress;
    triggerValuePop(DOM.stressValue);
  }

  // Renk ve parıltı animasyonu
  DOM.stressValue.style.color      = palette.color;
  DOM.stressValue.style.textShadow = palette.shadow;
  DOM.stressGlow.style.background  = palette.glow;
  DOM.stressGlow.style.opacity     = "1";

  // İkon arka plan
  DOM.stressIconWrap.style.background   = palette.icon;
  DOM.stressIconWrap.style.borderColor  = palette.iconBorder;
  DOM.stressIconWrap.style.color        = palette.color;

  // Aktif gösterge
  const activeEl = document.getElementById(palette.activeId);
  if (activeEl) activeEl.classList.add(palette.activeClass);
}

/* ══════════════════════════════════════════════════════════════════════
   11. FOOTER SAAT
══════════════════════════════════════════════════════════════════════ */

function updateFooterClock() {
  const now = new Date();
  DOM.footerTime.textContent = now.toLocaleTimeString("tr-TR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/* ══════════════════════════════════════════════════════════════════════
   12. SON GÜNCELLEME ZAMANI
══════════════════════════════════════════════════════════════════════ */

function updateLastUpdateBadge() {
  const now = new Date();
  DOM.updateTimer.textContent =
    `Son: ${now.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
}

/* ══════════════════════════════════════════════════════════════════════
   13. ANA POLLİNG DÖNGÜSÜ
══════════════════════════════════════════════════════════════════════ */

/**
 * /metrics endpoint'ini sorgular ve tüm DOM bileşenlerini günceller.
 */
async function fetchAndUpdateMetrics() {
  try {
    const response = await fetch("/metrics", {
      cache: "no-store",
      headers: { "Accept": "application/json" },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    state.errorCount = 0;

    // ── Yüz Durumu ──────────────────────────────────────────────────
    updateFaceStatus(data.face_detected === true);

    // ── Kalibrasyon ──────────────────────────────────────────────────
    const progress = typeof data.buffer_progress === "number"
      ? data.buffer_progress : 0;
    updateCalibration(progress, data.is_ready === true);

    // ── Biyometrik Metrikler ─────────────────────────────────────────
    updateBPM(data.bpm);
    updateHRV(data.hrv);
    updateRR(data.rr);
    updateStress(data.stress);

    // ── Son Güncelleme Rozeti ────────────────────────────────────────
    updateLastUpdateBadge();

  } catch (err) {
    state.errorCount++;

    if (state.errorCount >= 3) {
      // 3 ardışık hata → bağlantı koptu
      DOM.statusCal.classList.remove("active");
      DOM.statusCalText.textContent = "Bağlantı Kesildi";
      DOM.calStatusText.textContent = "⚠ Sunucuya bağlanılamıyor";
    }

    console.warn("[rPPG] Metrik çekme hatası:", err.message);
  }
}

/* ══════════════════════════════════════════════════════════════════════
   14. BAŞLATMA
══════════════════════════════════════════════════════════════════════ */

function init() {
  // İlk sorgu — hemen yap
  fetchAndUpdateMetrics();

  // Periyodik polling
  setInterval(fetchAndUpdateMetrics, CONFIG.POLL_INTERVAL_MS);

  // Footer saati
  updateFooterClock();
  setInterval(updateFooterClock, CONFIG.CLOCK_INTERVAL_MS);

  // Sayfa ilk yüklendiğinde tüm değerleri "—" (kalibre ediyor) göster
  [DOM.bpmValue, DOM.hrvValue, DOM.rrValue].forEach(el => {
    el.classList.add("calibrating");
  });
  DOM.stressValue.classList.add("calibrating");

  console.log(
    "%c rPPG Biometric Dashboard %c v1.0 ",
    "background:#161b22;color:#39d353;font-weight:bold;padding:4px 8px;border-radius:4px 0 0 4px;",
    "background:#1c2230;color:#8b949e;padding:4px 8px;border-radius:0 4px 4px 0;"
  );
}

// DOM hazır olduğunda başlat
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
