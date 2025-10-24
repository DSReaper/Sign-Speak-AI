// Get modal elements
const changePasswordModal = document.getElementById('change-password-modal');
const changeEmailModal = document.getElementById('change-email-modal');

// Get button elements
const changePasswordBtn = document.getElementById('change-password-btn');
const changeEmailBtn = document.getElementById('change-email-btn');

// Get close elements
const closeButtons = document.querySelectorAll('.close');

// Event listeners for opening modals
changePasswordBtn.addEventListener('click', () => {
  changePasswordModal.style.display = 'block';
});

changeEmailBtn.addEventListener('click', () => {
  changeEmailModal.style.display = 'block';
});

// Event listeners for closing modals
closeButtons.forEach(button => {
  button.addEventListener('click', () => {
    changePasswordModal.style.display = 'none';
    changeEmailModal.style.display = 'none';
  });
});

// Close modal when clicking outside
window.addEventListener('click', (event) => {
  if (event.target === changePasswordModal) {
    changePasswordModal.style.display = 'none';
  }
  if (event.target === changeEmailModal) {
    changeEmailModal.style.display = 'none';
  }
});

// ---------------- Camera Selection Logic ----------------
// Persist key for selected camera
const CAMERA_PREF_KEY = 'ssai_preferred_camera_id';
const cameraSelect = document.getElementById('cameraSelect');
const cameraSelectStatus = document.getElementById('cameraSelectStatus');

async function ensureMediaPermission() {
  // Attempt light permission request to reveal device labels if not already granted
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const hasLabels = devices.some(d => d.kind === 'videoinput' && d.label);
    if (!hasLabels) {
      // Request temporary stream to unlock labels
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      stream.getTracks().forEach(t => t.stop());
    }
  } catch (err) {
    // Permission might be denied; continue with limited info
    console.warn('Camera permission not granted or unavailable:', err.message);
  }
}

async function populateCameras() {
  if (!cameraSelect) return;
  try {
    if (cameraSelectStatus) cameraSelectStatus.textContent = 'Detecting cameras...';
    await ensureMediaPermission();
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === 'videoinput');

    cameraSelect.innerHTML = '';
    if (!videoInputs.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No cameras found';
      cameraSelect.appendChild(opt);
      cameraSelect.disabled = true;
      if (cameraSelectStatus) cameraSelectStatus.textContent = 'No video input devices detected.';
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
    if (cameraSelectStatus) {
      cameraSelectStatus.textContent = 'Camera preference saved locally and applied in detection view.';
    }
  } catch (err) {
    console.error('Failed to populate cameras:', err);
    if (cameraSelectStatus) cameraSelectStatus.textContent = 'Error listing cameras (permission denied?)';
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
  // Initial population after DOM and short delay to allow any permission prompts
  window.addEventListener('DOMContentLoaded', () => {
    setTimeout(populateCameras, 300);
  });
  // Fallback populate if script loaded after DOM
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(populateCameras, 300);
  }
}
// ---------------------------------------------------------

// Change password form submission
document.getElementById('change-password-form').addEventListener('submit', async (e) => {
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
document.getElementById('change-email-form').addEventListener('submit', async (e) => {
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
logoutBtn.addEventListener('click', async function (e) {
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
