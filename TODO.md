# TODO: Implement phrase deletion on star button click

- [ ] Backend: Add deletePhrase function in phraseController.js to delete phrase by _id from user's phrases array.
- [ ] Backend: Add DELETE /translate/phrase/:id route in translateRoute.js to call deletePhrase.
- [ ] Frontend: Modify storage.ejs to include phrase._id as data attribute in phrase list items and star buttons.
- [ ] Frontend: Update storage.js to add event listener on star buttons to send DELETE request to /translate/phrase/:id.
- [ ] Frontend: On successful deletion, remove phrase from UI.
- [ ] Test full flow: clicking star deletes phrase from UI and database.
