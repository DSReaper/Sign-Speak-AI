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

    // Theme: 'light' | 'dark' (default light)
    const theme = (settings.theme === 'dark') ? 'dark' : 'light';
    const root = document.documentElement;
    root.setAttribute('data-theme', theme);
    // Text size scaling removed; default font size is used
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
