<?php
// Retrieve the form inputs
$name = $_POST['name'];
$pin = $_POST['pin'];
$session = $_POST['session'];

// Create a session ID
$randomNumber = rand(1, 9999);
$sessionID = $name . $pin . $randomNumber;

// Set the cookie
setcookie('ECOM', $sessionID, time() + $session);

// Generate the HTML response
$response = '<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=windows-1252">
    <title>IEDA 3302 HW1: Cookie Set</title>
</head>
<body bgcolor="#DDDDDD">
    <p><b><font color="blue">Cookie Set</font></b></p>
    <p><b>Cookie Name:</b> ECOM</p>
    <p><b>Cookie Value:</b> ' . $sessionID . '</p>
    <p>A cookie has been set with the provided values. To delete the cookie, click on the button.</p>
    <form method="POST" action="http://dcte.student.ust.hk/cgi-bin/unset_cookie.php">
        <input type="hidden" name="cookieName" value="ECOM">
        <input type="submit" value="Unset Cookie">
    </form>
</body>
</html>';

// Send the HTML response
echo $response;
?>
