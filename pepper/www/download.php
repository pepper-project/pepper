<?php
$uploads_dir = "/mnt/computation_state";
$short_file_name = $_GET["file"];
$full_file_name = "$uploads_dir/$short_file_name";

if (($short_file_name == "") || (file_exists($full_file_name) == false))
{
  header("HTTP/1.1 404 File Not Found");
}
else
{
  header("Content-Disposition: attachment; filename=\"$short_file_name\"");
  header("Content-type: application/binary");
  readfile($full_file_name);
}
?>
