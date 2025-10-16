// Global theme bootstrap: apply theme and text scale from localStorage
(function () {
  const KEY = 'appSettings';

  function readSettings() {
    try {
      const raw = localStorage.getItem(KEY);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }

  function apply(settings) {
    if (!settings) return;

    // Theme: 'light' | 'dark' | 'system'
    const theme = settings.theme || 'system';
    const root = document.documentElement;
    if (theme === 'system') {
      root.removeAttribute('data-theme');
    } else {
      root.setAttribute('data-theme', theme);
    }

    // Text scaling
    const scale = parseFloat(settings.textSize || '1');
    root.style.fontSize = `${100 * (isFinite(scale) ? scale : 1)}%`;
  }

  // Apply on DOM ready
  document.addEventListener('DOMContentLoaded', () => {
    apply(readSettings());
  });

  // React to changes from settings page in another tab
  window.addEventListener('storage', (e) => {
    if (e.key === KEY) {
      apply(readSettings());
    }
  });
})();
