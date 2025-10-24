(function(global){
  const PLAY_ICON_PATH = '<path d="M8 5v14l11-7z"/>';
  const PLACEHOLDER_TEXTS = [
    'Loading AI detection model...',
    'No detection yet',
    ''
  ];

  let modalEl = null;
  let textareaEl = null;
  let closeBtnEl = null;

  function init(opts = {}){
    const modalSel = opts.modalSelector || '#responseModal';
    const textareaSel = opts.textareaSelector || '#userResponse';
    const closeSel = opts.closeBtnSelector || '#closeModal';

    modalEl = document.querySelector(modalSel);
    textareaEl = document.querySelector(textareaSel);
    closeBtnEl = document.querySelector(closeSel);

    if (closeBtnEl) {
      closeBtnEl.addEventListener('click', () => {
        closeModal();
      });
    }
  }

  function openModal(){
    if (!modalEl) return;
    modalEl.style.display = 'flex';
  }

  function closeModal(){
    if (!modalEl) return;
    modalEl.style.display = 'none';
    if (textareaEl) textareaEl.value = '';
  }

  function setButtonPlayingState(btn){
    if (!btn) return;
    const textEl = btn.querySelector('span');
    btn.classList.add('playing');
    if (textEl) textEl.textContent = 'Playing...';

    // Try to transform icon into a pause icon.
    const svg = btn.querySelector('svg');
    if (svg) {
      // Replace with two rects to represent pause
      svg.innerHTML = `<g><rect x="6" y="4" width="3" height="16" fill="currentColor"/><rect x="14" y="4" width="3" height="16" fill="currentColor"/></g>`;
    }
  }

  function resetButton(btn){
    if (!btn) return;
    const textEl = btn.querySelector('span');
    btn.classList.remove('playing');
    if (textEl) textEl.textContent = 'Play audio';

    const svg = btn.querySelector('svg');
    if (svg) {
      svg.innerHTML = PLAY_ICON_PATH;
    }
  }

  async function playFromElement(btn){
    if (!btn) return;
    if (btn.classList.contains('playing')) return; // prevent double clicks

    // Determine source of phrase text.
    let raw = '';

    // 1. Storage page context: inside a .storage-phrase-item
  const storageItem = btn.closest('.storage-phrase-item');
    if (storageItem){
      const textSpan = storageItem.querySelector('.storage-phrase-text');
      if (textSpan){
        raw = (textSpan.textContent || '').replace(/(^"|"$)/g, '').trim();
      }
    }

    // 2. Fallback to camera detected phrase element
    if (!raw){
      const phraseEl = document.getElementById('detectedPhrase');
      if (phraseEl){
        raw = phraseEl.textContent.trim();
      }
    }

    if (!raw || PLACEHOLDER_TEXTS.includes(raw)){
      try { console.warn('[TTS] No phrase available or placeholder text:', raw); } catch(_) {}
      alert('No phrase available to speak yet.');
      return;
    }

    setButtonPlayingState(btn);
    try {
      // Load user TTS settings from localStorage
      let settings = {};
      try {
        const savedSettings = localStorage.getItem('appSettings');
        settings = savedSettings ? JSON.parse(savedSettings) : {};
      } catch (e) {
        // ignore
      }
      
      // Validate and prepare TTS request with user preferences
      const voiceGender = (settings.voiceGender === 'male') ? 'male' : 'female';
      const rate = (typeof settings.rate === 'number' && settings.rate >= 0.5 && settings.rate <= 2.0) 
                   ? settings.rate : 1.0;
      
      const requestBody = {
        text: raw,
        voiceGender: voiceGender,
        rate: rate
      };
      
      // Include auth if available to ensure server accepts and applies settings
      let headers = { 'Content-Type': 'application/json' };
      try {
        const token = localStorage.getItem('token');
        if (token) headers['Authorization'] = `Bearer ${token}`;
      } catch(_) {}

      const resp = await fetch('/tts', {
        method: 'POST',
        headers,
        credentials: 'same-origin',
        body: JSON.stringify(requestBody)
      });
      if (!resp.ok){
        let errText = 'TTS request failed';
        try { const j = await resp.json(); errText = j.error || errText; } catch (e) {}
        throw new Error(errText);
      }
      const blob = await resp.blob();
      // Client-side debug logs removed by request
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.addEventListener('ended', () => {
        resetButton(btn);
        URL.revokeObjectURL(url);
        openModal();
      });
      audio.addEventListener('error', () => {
        resetButton(btn);
        URL.revokeObjectURL(url);
        alert('Could not play audio');
      });
      audio.play().catch(err => {
        resetButton(btn);
        URL.revokeObjectURL(url);
        alert('Playback error: ' + err.message);
      });
    } catch (e){
      resetButton(btn);
      alert(e.message || 'TTS error');
    }
  }

  function attachToSelector(sel){
    document.querySelectorAll(sel).forEach(el => {
      attachToElement(el);
    });
  }

  function attachToElement(el){
    if (!el) return;
    // Avoid attaching multiple times
    if (el.__playAudioBound) return;
    el.__playAudioBound = true;
    el.addEventListener('click', (e) => {
      e.preventDefault();
      playFromElement(el);
    });
  }

  // Expose API
  global.PlayAudioModule = {
    init,
    playFromElement,
    resetButton,
    attachToSelector,
    attachToElement
  };
})(window);
