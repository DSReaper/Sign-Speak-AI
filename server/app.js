const express = require('express');
const path = require('path');
const app = express();
// const translateRoutes = require('../Backend_stuff/routes/translateRoute');

app.use(express.json());
// app.use('/translate', translateRoutes);

// Set EJS as the view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, '../client/views'));

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, '../public')));



// Route to render index.ejs
app.get('/', (req, res) => {
    res.render('index');
});

//login post route
app.use(express.urlencoded({ extended: true }));
// After successful login
app.post('/login', (req, res) => {
  const { email, password } = req.body;

  if (email === 'test@example.com' && password === 'password123') {
    res.redirect('/camera'); // instead of directly rendering
  } else {
    res.status(401).send('Invalid credentials');
  }
});

// Application routes
app.get('/camera', (req, res) => {
  res.render('camera'); // main camera screen
});

app.get('/storage', (req, res) => {
  res.render('storage');
});

app.get('/profile', (req, res) => {
  res.render('profile');
});


// Start the server
const port = 8000;
app.listen(port, () => console.log(`http://localhost:${port}`));