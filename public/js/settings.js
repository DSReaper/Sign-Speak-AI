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
