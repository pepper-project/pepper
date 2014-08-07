<?php
$uploads_dir = "/mnt/computation_state";
if ($_FILES["file"]["error"] > 0)
{
  header("HTTP/1.1 500 Internal Server Error");
}
else
{
  if (is_uploaded_file($_FILES["file"]["tmp_name"])) 
  {
    $tmp_name = $_FILES["file"]["tmp_name"];
    $name = $_FILES["file"]["name"];
    $ret = move_uploaded_file($tmp_name, "$uploads_dir/$name");
    if ($ret == FALSE)
    {
      header("HTTP/1.1 500 Internal Server Error");
    }
    else
    {
      header("HTTP/1.1 200 OK");
    }
  }
  else
  {
    header("HTTP/1.1 500 Internal Server Error");
  }
}
?>
