# Written by ChatGPT
{
    "directory": $directory,
    "file": $file,
    "command": (
        [
            $command,
            # 1. Take all named args (args1, args2, foo, bar...)
            # 2. Remove the non-flag metadata keys
            # 3. Process the remaining values
            ($ARGS.named | del(.directory, .file, .command, .output) | .[] 
                | split("\n")[] 
                | select(length > 0))
        ]
        | join(" ")
    ),
    "output": $output
}
