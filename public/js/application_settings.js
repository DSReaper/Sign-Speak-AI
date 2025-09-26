// public/js/application-settings.js
(function () {
  const KEY = "appSettings";

  const els = {
    voiceGender: () => document.querySelector('input[name="voiceGender"]:checked'),
    rate: document.getElementById("rate"),
    volume: document.getElementById("volume"),
    volLabel: document.getElementById("volLabel"),
    textSize: document.getElementById("textSize"),
    highContrast: document.getElementById("highContrast"),
    theme: document.getElementById("theme"),
    saveBtn: document.getElementById("saveBtn"),
    resetBtn: document.getElementById("resetBtn"),
    previewTTS: document.getElementById("previewTTS"),
  };

  // --- Load + apply ---
  function load() {
    try {
      const raw = localStorage.getItem(KEY);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }

  function apply(settings) {
    if (!settings) return;

    // Text scaling
    const scale = parseFloat(settings.textSize || 1);
    document.documentElement.style.fontSize = `${100 * scale}%`;

    // High contrast
    document.body.classList.toggle("high-contrast", !!settings.highContrast);

    // Theme
    const theme = settings.theme || "system";
    if (theme === "system") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", theme);
    }
  }

  function populateUI(settings) {
    if (!settings) return;
    const g = settings.voiceGender || "female";
    const gEl = document.querySelector(`input[name="voiceGender"][value="${g}"]`);
    if (gEl) gEl.checked = true;

    els.rate.value = settings.rate || "1";
    els.volume.value = typeof settings.volume === "number" ? settings.volume : 1;
    els.volLabel.textContent = `(${Math.round(els.volume.value * 100)}%)`;
    els.textSize.value = settings.textSize || "1";
    els.highContrast.checked = !!settings.highContrast;
    els.theme.value = settings.theme || "system";
  }

  function save() {
    const settings = {
      voiceGender: els.voiceGender().value,
      rate: parseFloat(els.rate.value),
      volume: parseFloat(els.volume.value),
      textSize: els.textSize.value,
      highContrast: els.highContrast.checked,
      theme: els.theme.value,
    };
    localStorage.setItem(KEY, JSON.stringify(settings));
    apply(settings);
    alert("Settings saved.");
  }

  function reset() {
    const defaults = {
      voiceGender: "female",
      rate: 1,
      volume: 1,
      textSize: "1",
      highContrast: false,
      theme: "system",
    };
    localStorage.setItem(KEY, JSON.stringify(defaults));
    populateUI(defaults);
    apply(defaults);
  }

  // --- TTS Preview ---
  function getVoicesByGender(gender) {
    const all = speechSynthesis.getVoices();
    if (!all.length) return [];

    // Rough gender filter by name
    const isFemale = v => /female|woman|zira|susan|samantha|victoria|zira/i.test(v.name);
    const isMale   = v => /male|man|david|mark|alex|daniel|george/i.test(v.name);

    return all.filter(v => gender === "female" ? isFemale(v) : isMale(v));
  }

  function preview() {
    if (!("speechSynthesis" in window)) {
      alert("Speech preview not supported on this browser.");
      return;
    }
    const gender = els.voiceGender().value;
    const rate = parseFloat(els.rate.value);
    const volume = parseFloat(els.volume.value);

    const utter = new SpeechSynthesisUtterance("This is a preview of your speech settings.");
    utter.rate = rate;
    utter.volume = volume;

    const preferred = getVoicesByGender(gender);
    if (preferred.length) utter.voice = preferred[0];

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  }

  // Events
  document.addEventListener("DOMContentLoaded", () => {
    const current = load();
    populateUI(current || undefined);
    apply(current || undefined);

    els.volume.addEventListener("input", () => {
      els.volLabel.textContent = `(${Math.round(els.volume.value * 100)}%)`;
    });

    els.saveBtn.addEventListener("click", save);
    els.resetBtn.addEventListener("click", reset);
    els.previewTTS.addEventListener("click", preview);
  });
})();
