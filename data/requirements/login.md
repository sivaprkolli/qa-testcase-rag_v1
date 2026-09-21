# Login Requirements

# Login Requirement

## Requirement ID
REQ-LOGIN-001

## Description

The application shall allow registered users to log in using their
registered email address and password.

## Acceptance Criteria

### AC-01 Successful Login

Given the user has a registered account,
when the user enters a valid email address and valid password,
and clicks the Login button,
then the user should be successfully logged in.

### AC-02 Invalid Password

Given the user has a registered account,
when the user enters a valid email address and an invalid password,
and clicks the Login button,
then the login attempt should fail.

### AC-03 Invalid Email

Given the user is on the login page,
when the user enters an unregistered email address and a password,
and clicks the Login button,
then the login attempt should fail.

### AC-04 Empty Email

Given the user is on the login page,
when the user leaves the email field empty
and clicks the Login button,
then the system should indicate that the email is required.

### AC-05 Empty Password

Given the user is on the login page,
when the user enters a valid email address
and leaves the password field empty,
then the system should indicate that the password is required.