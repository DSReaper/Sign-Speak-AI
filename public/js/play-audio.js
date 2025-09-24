(function(global){
  const defaultDuration = 2000;
  const pauseShape = { type: 'rects' };
  const playPoints = '5,3 19,12 5,21';

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

    // Try to transform icon into a pause icon. Support polygon/path or replace svg innerHTML.
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
      svg.innerHTML = `<polygon points="${playPoints}" fill="currentColor"/>`;
    }
  }

  function playFromElement(btn, opts = {}){
    if (!btn) return;
    const duration = opts.duration || defaultDuration;
    setButtonPlayingState(btn);

    setTimeout(() => {
      resetButton(btn);
      openModal();
    }, duration);
  }

  function attachToSelector(sel){
    document.querySelectorAll(sel).forEach(el => {
      attachToElement(el);
    });
  }

  function attachToElement(el){
    if (!el) return;
    el.addEventListener('click', (e) => {
      // allow button default behavior to be prevented externally
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
