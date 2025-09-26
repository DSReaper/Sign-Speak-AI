# TODO: Fix 404 Errors for Settings and Application Settings Assets/Routes

## Steps to Complete:
1. [x] Add route for `/application_settings` in `server/app.js` (protected by auth, renders `application_settings.ejs`).
2. [x] Create `public/css/application_settings.css` (empty file to resolve unexpected 404).
3. [x] Edit `client/views/settings.ejs`: Fix JS src to `/js/settings.js`, navigation href to `/application_settings`, logout href to `/`.
4. [x] Edit `client/views/application_settings.ejs`: Fix JS src to `/js/application_settings.js`, back href to `/settings`.
5. [x] Update this TODO.md to mark steps complete.
6. [ ] Followup: Restart server, clear browser cache, test in browser (no 404s, navigation works, settings save/preview functional).

Progress: Edits complete. Test to verify.
