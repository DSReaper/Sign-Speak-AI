// Get modal elements (guarded in case markup changes)
const changePasswordModal = document.getElementById('change-password-modal');
const changeEmailModal = document.getElementById('change-email-modal');

// Get button elements
const changePasswordBtn = document.getElementById('change-password-btn');
const changeEmailBtn = document.getElementById('change-email-btn');

// Get close elements
const closeButtons = document.querySelectorAll('.close');

// Event listeners for opening modals
if (changePasswordBtn && changePasswordModal) {
  changePasswordBtn.addEventListener('click', () => {
    changePasswordModal.style.display = 'block';
  });
}

if (changeEmailBtn && changeEmailModal) {
  changeEmailBtn.addEventListener('click', () => {
    changeEmailModal.style.display = 'block';
  });
}

// Event listeners for closing modals
if (closeButtons && (changePasswordModal || changeEmailModal)) {
  closeButtons.forEach(button => {
    button.addEventListener('click', () => {
      if (changePasswordModal) changePasswordModal.style.display = 'none';
      if (changeEmailModal) changeEmailModal.style.display = 'none';
    });
  });
}

// Close modal when clicking outside
window.addEventListener('click', (event) => {
  if (event.target === changePasswordModal && changePasswordModal) {
    changePasswordModal.style.display = 'none';
  }
  if (event.target === changeEmailModal && changeEmailModal) {
    changeEmailModal.style.display = 'none';
  }
});

// ---------------- Camera Selection Logic ----------------
// Persist keys for selected camera
const CAMERA_PREF_KEY = 'ssai_preferred_camera_id';
const CAMERA_PREF_LABEL_KEY = 'ssai_preferred_camera_label';
const cameraSelect = document.getElementById('cameraSelect');
const cameraSelectStatus = document.getElementById('cameraSelectStatus');

function setStatus(text, type = 'info') {
  if (!cameraSelectStatus) return;
  cameraSelectStatus.textContent = text;
  cameraSelectStatus.style.color = type === 'error' ? '#ff6b6b' : '#aaa';
}

async function ensureMediaPermission() {
  // Attempt light permission request to reveal device labels if not already granted
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    throw new Error('Media devices API not supported in this browser');
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  const hasLabels = devices.some(d => d.kind === 'videoinput' && d.label);
  if (!hasLabels) {
    if (ensureMediaPermission.userInitiated !== true) {
      throw new Error('Permission required: interact with the dropdown to grant access');
    }
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    stream.getTracks().forEach(t => t.stop());
  }
}

async function populateCameras() {
  if (!cameraSelect) return;
  if (!window.isSecureContext) {
    setStatus('Camera access requires HTTPS. Please use the secure site URL.', 'error');
    return;
  }
  setStatus('Detecting cameras...');

  try {
    await ensureMediaPermission();
  } catch (_) {
    // Keep going; labels might be blank without permission
    setStatus('Permission not yet granted. Listing devices without labels.', 'error');
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === 'videoinput');

    cameraSelect.innerHTML = '';
    if (!videoInputs.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No cameras found';
      cameraSelect.appendChild(opt);
      cameraSelect.disabled = true;
      setStatus('No video input devices detected.', 'error');
      return;
    }

    cameraSelect.disabled = false;
    const preferredId = localStorage.getItem(CAMERA_PREF_KEY) || '';
    let defaultSelected = false;

    videoInputs.forEach((d, idx) => {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      const label = d.label || `Camera ${idx + 1}`;
      opt.textContent = label;
      if (!defaultSelected && preferredId && preferredId === d.deviceId) {
        opt.selected = true;
        defaultSelected = true;
      }
      cameraSelect.appendChild(opt);
    });

    // If no preference, select first and persist
    if (!defaultSelected) {
      cameraSelect.selectedIndex = 0;
      const first = videoInputs[0];
      localStorage.setItem(CAMERA_PREF_KEY, first.deviceId);
      localStorage.setItem(CAMERA_PREF_LABEL_KEY, first.label || 'Camera 1');
      setStatus('Default camera selected and saved.');
    } else {
      const match = videoInputs.find(v => v.deviceId === preferredId);
      if (match) {
        localStorage.setItem(CAMERA_PREF_LABEL_KEY, match.label || 'Saved camera');
        setStatus(`Using saved camera: ${match.label || 'Saved camera'}`);
      }
    }
  } catch (err) {
    console.error('Failed to populate cameras:', err);
    setStatus('Error listing cameras (permission denied?)', 'error');
  }
}

if (cameraSelect) {
  // Prefill select from localStorage immediately (no detection yet)
  (function prefillFromLocalStorage() {
    const savedId = localStorage.getItem(CAMERA_PREF_KEY) || '';
    const savedLabel = localStorage.getItem(CAMERA_PREF_LABEL_KEY) || '';
    cameraSelect.innerHTML = '';
    if (savedId) {
      const opt = document.createElement('option');
      opt.value = savedId;
      opt.textContent = savedLabel || 'Saved camera';
      opt.selected = true;
      cameraSelect.appendChild(opt);
      cameraSelect.disabled = false;
      setStatus(savedLabel ? `Using saved camera: ${savedLabel}` : 'Using saved camera (name requires permission)');
    } else {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No camera selected — click to list cameras';
      opt.disabled = true;
      opt.selected = true;
      cameraSelect.appendChild(opt);
      cameraSelect.disabled = false; // allow interaction to trigger detection
      setStatus('Click the dropdown to list available cameras.');
    }
  })();

  cameraSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val) {
      localStorage.setItem(CAMERA_PREF_KEY, val);
      const sel = cameraSelect.options[cameraSelect.selectedIndex];
      if (sel && sel.textContent) {
        localStorage.setItem(CAMERA_PREF_LABEL_KEY, sel.textContent);
        setStatus(`Using saved camera: ${sel.textContent}`);
      } else {
        setStatus('Preferred camera updated!');
      }
    }
  });

  // Detect cameras when the user interacts with the dropdown
  let cameraListEverPopulated = false;
  async function populateOnInteract(userInitiated) {
    try {
      ensureMediaPermission.userInitiated = !!userInitiated;
      await populateCameras();
      cameraListEverPopulated = true;
    } catch (_) {
      // ignore
    } finally {
      ensureMediaPermission.userInitiated = false;
    }
  }

  cameraSelect.addEventListener('mousedown', () => {
    if (!cameraListEverPopulated) populateOnInteract(true);
  });
  cameraSelect.addEventListener('focus', () => {
    if (!cameraListEverPopulated) populateOnInteract(false);
  });
  cameraSelect.addEventListener('keydown', (e) => {
    if (!cameraListEverPopulated && (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ')) {
      populateOnInteract(true);
    }
  });

  // Repopulate on device changes
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', () => {
      populateCameras();
    });
  }
}
// ---------------------------------------------------------

// Change password form submission
const changePasswordForm = document.getElementById('change-password-form');
if (changePasswordForm) changePasswordForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const currentPassword = document.getElementById('current-password').value;
  const newPassword = document.getElementById('new-password').value;
  const confirmPassword = document.getElementById('confirm-password').value;

  if (newPassword !== confirmPassword) {
    alert('New passwords do not match');
    return;
  }

  try {
    const response = await fetch('/auth/change-password', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ currentPassword, newPassword }),
    });

    const data = await response.json();
    if (response.ok) {
      alert(data.message);
      changePasswordModal.style.display = 'none';
      document.getElementById('change-password-form').reset();
    } else {
      alert(data.message);
    }
  } catch (error) {
    console.error('Error:', error);
    alert('An error occurred. Please try again.');
  }
});

// Change email form submission
const changeEmailForm = document.getElementById('change-email-form');
if (changeEmailForm) changeEmailForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const newEmail = document.getElementById('new-email').value;

  try {
    const response = await fetch('/auth/change-email', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ newEmail }),
    });

    const data = await response.json();
    if (response.ok) {
      alert(data.message);
      changeEmailModal.style.display = 'none';
      document.getElementById('change-email-form').reset();
    } else {
      alert(data.message);
    }
  } catch (error) {
    console.error('Error:', error);
    alert('An error occurred. Please try again.');
  }
});

// Logout button event listener
const logoutBtn = document.getElementById('logoutBtn');
if (logoutBtn) logoutBtn.addEventListener('click', async function (e) {
  e.preventDefault();
  try {
    await fetch('/auth/logout', { method: 'POST', credentials: 'same-origin' });
  } catch (err) {
    // handle error
  } finally {
    // Clear localStorage
    localStorage.removeItem('token');
    localStorage.removeItem('userId');
    // Clear cookie
    document.cookie = 'token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    window.location.href = '/';
  }
});
