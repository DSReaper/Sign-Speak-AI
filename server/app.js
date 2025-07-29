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

// Start the server
const port = 8000;
app.listen(port, () => console.log(`http://localhost:${port}`));