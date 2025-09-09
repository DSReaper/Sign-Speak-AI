# MongoDB Database Configuration and Validation Notes

## Current Database Setup Observations

- The MongoDB collection `db.users` has validation rules enforcing the following required fields:
  - `_id` (ObjectId)
  - `email` (string)
  - `password` (string)
  - `registered` (timestamp)
  - `role` (string)
  - `username` (string)

- Indexes include:
  - `_id` as unique identifier
  - Compound index on `User_Login_text_Login_password_text`
  - Geospatial indexes on `TimeStamp_2dsphere` and `Type_user_2dsphere`

## Issues Identified

- The current Mongoose user schema and signup controller do not include fields for:
  - `registered` (timestamp)
  - `role` (string)
  - `username` (string)

- The database validation requires these fields, so documents missing them will be rejected.

## Required Changes to Backend Code

1. **Update Mongoose User Schema** (`Backend_stuff/controllers/authController.js`):

```js
const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  userType: { type: String, default: 'user' },
  registered: { type: Date, default: Date.now },
  role: { type: String, default: 'user' },
  username: { type: String, default: '' }
});
```

2. **Update Signup Controller** to include the new fields when creating a user:

```js
const user = new User({
  email,
  password: hashedPassword,
  userType: userType || 'user',
  registered: new Date(),
  role: 'user',
  username: '' // or set from signup form if available
});
```

3. **Add Logging** in the signup controller to confirm successful save or catch errors.

## Additional Instructions

- Ensure the MongoDB user account used by the backend has write permissions to the `User` database.
- After applying these changes, test signup again and verify documents appear in the `db.users` collection under the `User` database.
- If any new validation rules are added in the future, update the Mongoose schema accordingly.

---

This document should be reviewed and updated as needed when database schema or validation rules change.
