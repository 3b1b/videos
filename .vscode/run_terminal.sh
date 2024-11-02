# This will paste whatever is on the clipboard into a new Terminal window, (the clipboard as text)
osascript <<EOF
tell application "Terminal"
    if (count of windows) > 0 then
        do script "checkpoint_paste()" in front window
        activate
    else
        display dialog "No open Terminal window found." buttons {"OK"} default button "OK"
    end if
end tell
EOF
