// public/js/application-settings.js
(function () {
  const KEY = "appSettings";

  const els = {
    voiceGender: () => document.querySelector('input[name="voiceGender"]:checked'),
    rate: document.getElementById("rate"),
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

    // Theme: default to light if not set; only 'light' or 'dark'
    const theme = settings.theme || "light";
    document.documentElement.setAttribute("data-theme", theme);
  }

  function populateUI(settings) {
    if (!settings) return;
    const g = settings.voiceGender || "female";
    const gEl = document.querySelector(`input[name="voiceGender"][value="${g}"]`);
    if (gEl) gEl.checked = true;

    els.rate.value = settings.rate || "1";
  els.theme.value = settings.theme || "light";
  }

  function save() {
    const settings = {
      voiceGender: els.voiceGender().value,
      rate: parseFloat(els.rate.value),
  theme: els.theme.value || "light",
    };
    localStorage.setItem(KEY, JSON.stringify(settings));
    apply(settings);
  }

  function reset() {
    const defaults = {
      voiceGender: "female",
      rate: 1,
  theme: "light",
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

    const utter = new SpeechSynthesisUtterance("This is a preview of your speech settings.");
    utter.rate = rate;

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

    els.saveBtn.addEventListener("click", save);
    els.resetBtn.addEventListener("click", reset);
    els.previewTTS.addEventListener("click", preview);

    // Auto-save on change to avoid mismatches if user navigates away without clicking Save
    try {
      const voiceInputs = document.querySelectorAll('input[name="voiceGender"]');
      voiceInputs.forEach(r => r.addEventListener('change', () => save()));
      if (els.rate) els.rate.addEventListener('change', () => save());
      if (els.theme) els.theme.addEventListener('change', () => save());
    } catch (_) {}
  });
})();
