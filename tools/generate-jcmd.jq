{
    "directory": $directory,
    "file": $file,
    "command": ($command + " " + ($args | split("\n") | map(select(length > 0)) | join(" ")))
}
