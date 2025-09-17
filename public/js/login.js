    // Switching between sign up and sign in form
    document.addEventListener('DOMContentLoaded', function() {
    const signInTab = document.getElementById('sign-in-tab');
    const signUpTab = document.getElementById('sign-up-tab');
    const signInForm = document.getElementById('sign-in-form');
    const signUpForm = document.getElementById('sign-up-form');
    const switchToSignUp = document.getElementById('switch-to-sign-up');
    const switchToSignIn = document.getElementById('switch-to-sign-in');

    function showSignInForm() {
        signInTab.classList.add('active');
        signInTab.setAttribute('aria-selected', 'true');
        signUpTab.classList.remove('active');
        signUpTab.setAttribute('aria-selected', 'false');
        signInForm.classList.add('active');
        signUpForm.classList.remove('active');
    }

    function showSignUpForm() {
        signUpTab.classList.add('active');
        signUpTab.setAttribute('aria-selected', 'true');
        signInTab.classList.remove('active');
        signInTab.setAttribute('aria-selected', 'false');
        signUpForm.classList.add('active');
        signInForm.classList.remove('active');
    }

    signInTab.addEventListener('click', showSignInForm);
    signUpTab.addEventListener('click', showSignUpForm);
    switchToSignUp.addEventListener('click', function(e) {
        e.preventDefault();
        showSignUpForm();
    });
    switchToSignIn.addEventListener('click', function(e) {
        e.preventDefault();
        showSignInForm();
    });


    // Password show and hide functionality
    const passwordToggles = document.querySelectorAll('.password-toggle');
    passwordToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const input = this.previousElementSibling;
            const icon = this.querySelector('i');
            
            if (input.type === 'password') {
                input.type = 'text';
                icon.classList.remove('fa-eye');
                icon.classList.add('fa-eye-slash');
                this.setAttribute('aria-label', 'Hide password');
            } else {
                input.type = 'password';
                icon.classList.remove('fa-eye-slash');
                icon.classList.add('fa-eye');
                this.setAttribute('aria-label', 'Show password');
            }
        });
    });

    // Form validation
    function validateEmail(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(String(email).toLowerCase());
        // Regex for basic email validation
    }

    function validatePassword(password) {
        return password.length >= 8;
        //need more edits here do after data testing
    }

// Sign In Form Validation
    const signInFormEl = document.getElementById('sign-in-form');
    signInFormEl.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const email = document.getElementById('sign-in-email').value;
        const password = document.getElementById('sign-in-password').value;
        let isValid = true;
        
        // Validate email
        if (!validateEmail(email)) {
            document.getElementById('sign-in-email-error').style.display = 'block';
            isValid = false;
        } else {
            document.getElementById('sign-in-email-error').style.display = 'none';
        }
        
        // Validate password
        if (!validatePassword(password)) {
            document.getElementById('sign-in-password-error').style.display = 'block';
            isValid = false;
        } else {
            document.getElementById('sign-in-password-error').style.display = 'none';
        }
        
        if (isValid) {
            // Send signin request
            fetch('/auth/signin', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email, password })
            })
            .then(response => response.json())
            .then(data => {
                if (data.token) {
                    // Store token and redirect or handle success
                    localStorage.setItem('token', data.token);
                    localStorage.setItem('userId', data.userId);
                    // Redirect to dashboard or home
                    window.location.href = '/camera'; // Adjust as needed
                } else {
                    alert(data.message || 'Login failed');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred during login');
            });
        }
    });

    // Sign Up Form Validation
    const signUpFormEl = document.getElementById('sign-up-form');
    signUpFormEl.addEventListener('submit', function(e) {

        e.preventDefault();
        const email = document.getElementById('sign-up-email').value;
        const password = document.getElementById('sign-up-password').value;
        const confirmPassword = document.getElementById('sign-up-confirm-password').value;
        const hearingStatusSelect = document.getElementById('hearingStatus');
        let isValid = true;
        
        // Validate email
        if (!validateEmail(email)) {
            document.getElementById('sign-up-email-error').style.display = 'block';
            isValid = false;
        } else {
            document.getElementById('sign-up-email-error').style.display = 'none';
        }
        
        // Validate password
        if (!validatePassword(password)) {
            document.getElementById('sign-up-password-error').style.display = 'block';
            isValid = false;
        } else {
            document.getElementById('sign-up-password-error').style.display = 'none';
        }
        
        // Validate password confirmation
        if (password !== confirmPassword) {
            document.getElementById('sign-up-confirm-error').style.display = 'block';
            isValid = false;
        } else {
            document.getElementById('sign-up-confirm-error').style.display = 'none';
        }
        
        if (isValid) {
            // Send signup request
            fetch('/auth/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    email, 
                    password, 
                    hearingStatus: hearingStatusSelect ? hearingStatusSelect.value : 'Hard of hearing' 
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.token) {
                    alert('Signup successful! Please login.');
                    showSignInForm();
                } else {
                    alert(data.message || 'Signup failed');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred during signup');
            });
        }
    });

   // Google auth button click
    const googleButtons = document.querySelectorAll('.btn-google');
    googleButtons.forEach(button => {
        button.addEventListener('click', function() {
          // Add google auth functionality
        });
    });

    // Forgot password link
    const forgotPassword = document.querySelector('.forgot-password');
    forgotPassword.addEventListener('click', function(e) {
        e.preventDefault();

        //Add email functionality for the reset password
        
    });

    // Tool tips functionality
    const infoIcon = document.querySelector('.info-icon');
    infoIcon.addEventListener('focus', function() {
        this.querySelector('.tooltip').style.visibility = 'visible';
        this.querySelector('.tooltip').style.opacity = '1';
    });

    infoIcon.addEventListener('blur', function() {
        this.querySelector('.tooltip').style.visibility = 'hidden';
        this.querySelector('.tooltip').style.opacity = '0';
    });
    });
