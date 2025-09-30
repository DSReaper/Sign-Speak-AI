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
  // Attach play button behavior using event delegation so dynamically
  function setupDelegatedPlay(){
    if (!phraseList.__delegatedPlayBound){
      phraseList.addEventListener('click', (e) => {
        const btn = e.target.closest('.storage-play-btn');
        if (!btn) return;
        e.preventDefault();
        if (window.PlayAudioModule && typeof window.PlayAudioModule.playFromElement === 'function') {
            window.PlayAudioModule.playFromElement(btn);
        } else {
          // Fallback basic simulation
          btn.classList.add('playing');
          const svg = btn.querySelector('svg');
          if (svg) svg.innerHTML = '<g><rect x="6" y="4" width="3" height="16" fill="currentColor"/><rect x="14" y="4" width="3" height="16" fill="currentColor"/></g>';
          setTimeout(() => {
            btn.classList.remove('playing');
            if (svg) svg.innerHTML = '<path d="M8 5v14l11-7z"/>';
            const modal = document.getElementById('responseModal');
            if (modal) modal.style.display = 'flex';
          }, 2000);
        }
      });
      phraseList.__delegatedPlayBound = true;
    }
  }

  setupDelegatedPlay();

  // Add event listener for star buttons to delete phrase
  function attachStarButtonHandlers() {
    const phraseList = document.getElementById('phrasesList');
    if (!phraseList) return;

    phraseList.addEventListener('click', async (event) => {
      const target = event.target.closest('.storage-star-btn');
      if (!target) return;

      event.stopPropagation(); // Prevent event bubbling to phrase block

      const phraseId = target.getAttribute('data-phrase-id');
      if (!phraseId) return;

      // Confirm deletion (optional)
      if (!confirm('Are you sure you want to delete this phrase?')) return;

      try {
        const response = await fetch(`/translate/phrase/${phraseId}`, {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json'
          }
        });

        if (!response.ok) {
          const errorData = await response.json();
          alert('Failed to delete phrase: ' + (errorData.message || 'Unknown error'));
          return;
        }

        // Remove phrase from UI
        const li = target.closest('li.storage-phrase-item');
        if (li) {
          li.remove();
        }
      } catch (error) {
        console.error('Error deleting phrase:', error);
        alert('Error deleting phrase. Please try again.');
      }
    });
  }

  attachStarButtonHandlers();

});
