const express = require('express');
const app = express();
const translateRoutes = require('.\Backend_stuff\routes\translateRoute.js');

app.use(express.json());
app.use('/translate', translateRoutes);

app.listen(8000, () => console.log("Server running on port 8000"));