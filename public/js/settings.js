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
// Persist key for selected camera
const CAMERA_PREF_KEY = 'ssai_preferred_camera_id';
const cameraSelect = document.getElementById('cameraSelect');
const cameraSelectStatus = document.getElementById('cameraSelectStatus');
const detectCamerasBtn = document.getElementById('detectCamerasBtn');

function setStatus(text, type = 'info') {
  if (!cameraSelectStatus) return;
  cameraSelectStatus.textContent = text;
  cameraSelectStatus.style.color = type === 'error' ? '#ff6b6b' : '#aaa';
}

async function ensureMediaPermission() {
  // Attempt light permission request to reveal device labels if not already granted
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      throw new Error('Media devices API not supported in this browser');
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    const hasLabels = devices.some(d => d.kind === 'videoinput' && d.label);
    if (!hasLabels) {
      // Only request permission in response to user gesture (button click)
      if (ensureMediaPermission.userInitiated !== true) {
        // caller didn't mark as user-initiated; let populateCameras handle messaging
        throw new Error('Permission required: please click Detect cameras');
      }
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      stream.getTracks().forEach(t => t.stop());
    }
  } catch (err) {
    // Permission might be denied; continue with limited info
    console.warn('Camera permission not granted or unavailable:', err.message);
    throw err;
  }
}

async function populateCameras() {
  if (!cameraSelect) return;
  try {
    if (!window.isSecureContext) {
      setStatus('Camera access requires HTTPS. Please use the secure site URL.', 'error');
      return;
    }
    setStatus('Detecting cameras...');

    try {
      await ensureMediaPermission();
    } catch (permErr) {
      // Update status but still try enumerateDevices (labels may be empty)
      setStatus('Permission not yet granted. Select from available devices or click Detect cameras.', 'error');
    }

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
    videoInputs.forEach((d, idx) => {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      const label = d.label || `Camera ${idx + 1}`;
      opt.textContent = label;
      if (preferredId && preferredId === d.deviceId) opt.selected = true;
      cameraSelect.appendChild(opt);
    });

    if (!preferredId) {
      // Store the first camera as default preference
      localStorage.setItem(CAMERA_PREF_KEY, videoInputs[0].deviceId);
    }
    setStatus('Camera preference saved locally and applied in detection view.');
  } catch (err) {
    console.error('Failed to populate cameras:', err);
    setStatus('Error listing cameras (permission denied?)', 'error');
  }
}

if (cameraSelect) {
  cameraSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val) {
      localStorage.setItem(CAMERA_PREF_KEY, val);
      if (cameraSelectStatus) cameraSelectStatus.textContent = 'Preferred camera updated! Re-open the detection page to apply.';
    }
  });
  // Repopulate on device changes 
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', () => {
      populateCameras();
    });
  }
  // Wire manual detect button to comply with permission gesture policies
  if (detectCamerasBtn) {
    detectCamerasBtn.addEventListener('click', async () => {
      try {
        ensureMediaPermission.userInitiated = true;
        await ensureMediaPermission();
      } catch (e) {
        // If the user dismisses the prompt or blocks, we still try to list devices
      } finally {
        ensureMediaPermission.userInitiated = false;
        await populateCameras();
      }
    });
  }

  // Attempt a best-effort populate (will not force permission prompt)
  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', () => setTimeout(populateCameras, 300));
  } else {
    setTimeout(populateCameras, 300);
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
