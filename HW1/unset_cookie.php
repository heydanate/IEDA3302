<?php
// Retrieve the cookie name from the form submission
$cookieName = $_POST['cookieName'];

// Unset the cookie by setting its expiration time to the past
setcookie($cookieName, '', time() - 3600);

// Get the client's IP address
$clientIP = $_SERVER['REMOTE_ADDR'];

// Generate the message
$message = "Dear user from " . $clientIP . ", I have killed my cookie on your computer.";

// Send the message
echo $message;
?>
