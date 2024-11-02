#!/bin/zsh

# Get the full file name from the argument
fullFileName="$1"

# Check if the file name is provided
if [[ -z "$fullFileName" ]]; then
    echo "No active file provided."
    exit 1
fi

# Use AppleScript to run a command in the Terminal
osascript <<EOF
tell application "Terminal"
    if (count of windows) > 0 then
        do script "manimvideos '$fullFileName'" in front window
        activate
    else
        -- Open a new Terminal window and run the command
        do script "manimvideos '$fullFileName'"
        activate
    end if
end tell
EOF
