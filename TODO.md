# TODO List for Preventing Camera Page Access Without Sign-In

- [x] Add JWT authentication middleware to server/app.js
- [x] Apply authentication middleware to /camera route
- [x] Add cookie-parser middleware to server/app.js
- [x] Update client-side login.js to set token in cookie
- [x] Add debugging logs to server and client
- [ ] Test accessing /camera without signing in (should redirect to login)
- [ ] Test signing in and accessing /camera (should allow access)
- [ ] Test accessing other protected routes if needed
