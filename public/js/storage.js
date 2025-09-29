document.addEventListener('DOMContentLoaded', () => {
  const searchInput = document.getElementById('storageSearchInput');
  const sortSelect = document.getElementById('storageSortSelect');
  const phraseList = document.querySelector('.storage-phrase-list');

  if (!phraseList) return;

  const getItemsArray = () => Array.from(phraseList.querySelectorAll('.storage-phrase-item'));

  function parseDateToTimestamp(dateStr) {
    if (!dateStr) return 0;
    dateStr = dateStr.trim();

    // ISO-like (YYYY-MM-DD or YYYY/MM/DD)
    if (/^\d{4}[-\/]\d{1,2}[-\/]\d{1,2}$/.test(dateStr)) {
      const parts = dateStr.split(/[-\/]/);
      const y = parseInt(parts[0], 10);
      const m = parseInt(parts[1], 10) - 1;
      const d = parseInt(parts[2], 10);
      return new Date(y, m, d).getTime();
    }

    // D/M/YYYY or M/D/YYYY (slash separated)
    if (/^\d{1,2}\/\d{1,2}\/\d{4}$/.test(dateStr)) {
      const parts = dateStr.split('/').map(p => parseInt(p, 10));
      const a = parts[0], b = parts[1], y = parts[2];

      // If day > 12 it's definitely day-first (DD/MM/YYYY)
      if (a > 12) return new Date(y, b - 1, a).getTime();

      // If month > 12 then it's impossible, fallback
      if (b > 12) return new Date(y, a - 1, b).getTime();

      return new Date(y, b - 1, a).getTime();
    }

    // Fallback: try native parse
    const parsed = Date.parse(dateStr);
    return isNaN(parsed) ? 0 : parsed;
  }

  function getPhraseText(item) {
    const el = item.querySelector('.storage-phrase-text');
    return el ? el.textContent.replace(/(^"|"$)/g, '').trim().toLowerCase() : '';
  }

  function getPhraseDateTimestamp(item) {
    const el = item.querySelector('.storage-phrase-date');
    if (!el) return 0;
    return parseDateToTimestamp(el.textContent || el.innerText || '');
  }

  function applySearchFilter() {
    const q = (searchInput.value || '').trim().toLowerCase();
    const items = getItemsArray();
    items.forEach(item => {
      const text = getPhraseText(item);
      const matches = q === '' || text.indexOf(q) !== -1;
      item.style.display = matches ? '' : 'none';
    });
  }

  function applySort() {
    const mode = sortSelect.value;
    const items = getItemsArray().filter(i => i.style.display !== 'none');

    items.sort((a, b) => {
      if (mode === 'date-new-old') {
        return getPhraseDateTimestamp(b) - getPhraseDateTimestamp(a);
      }
      if (mode === 'date-old-new') {
        return getPhraseDateTimestamp(a) - getPhraseDateTimestamp(b);
      }
      if (mode === 'length-short-long') {
        return getPhraseText(a).length - getPhraseText(b).length;
      }
      if (mode === 'length-long-short') {
        return getPhraseText(b).length - getPhraseText(a).length;
      }
      return 0;
    });

    // Re-append items in order
    items.forEach(it => phraseList.appendChild(it));
  }

  // Debounce helper
  function debounce(fn, wait) {
    let t = null;
    return function(...args) {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), wait);
    };
  }

  const debouncedSearch = debounce(() => {
    applySearchFilter();
    applySort(); // keep sorted order for filtered results
  }, 180);

  searchInput.addEventListener('input', debouncedSearch);
  sortSelect.addEventListener('change', () => {
    applySort();
  });

  applySort();
  // Initialize PlayAudioModule so it can control the modal on this page
  if (window.PlayAudioModule && typeof window.PlayAudioModule.init === 'function') {
    window.PlayAudioModule.init({ modalSelector: '#responseModal', textareaSelector: '#userResponse', closeBtnSelector: '#closeModal' });
  }
  // Attach play button behavior: delegate to PlayAudioModule when available
  function attachStoragePlayHandlers() {
    const playButtons = document.querySelectorAll('.storage-play-btn');
    if (!playButtons || playButtons.length === 0) return;

    if (window.PlayAudioModule && typeof window.PlayAudioModule.attachToElement === 'function') {
      playButtons.forEach(btn => window.PlayAudioModule.attachToElement(btn));
      return;
    }

    // Fallback: simple simulated play behavior (same UX as camera page fallback)
    playButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        const textEl = btn.querySelector('span');
        btn.classList.add('playing');
        if (textEl) textEl.textContent = 'Playing...';
        setTimeout(() => {
          btn.classList.remove('playing');
          if (textEl) textEl.textContent = 'Play audio';
          const modal = document.getElementById('responseModal');
          if (modal) modal.style.display = 'flex';
        }, 2000);
      });
    });
  }

  attachStoragePlayHandlers();

});
