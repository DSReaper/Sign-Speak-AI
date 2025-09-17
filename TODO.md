# TODO List for Testing Auth HTTP Requests

- [x] Start the Express server using `npm run dev`
- [ ] Test POST /auth/signup with valid dummy data (email: test@example.com, password: password123, hearingStatus: Hard of hearing)
- [x] Test POST /auth/signin with valid dummy data (email: test@example.com, password: password123)
- [ ] Test POST /auth/signup with duplicate email to check error handling
- [ ] Test POST /auth/signin with invalid password to check error handling
- [ ] Test POST /auth/signin with non-existent email to check error handling
- [ ] Stop the server after testing
