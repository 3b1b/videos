import sublime_plugin
import sublime
import os
import subprocess as sp
import threading
import re
import time


def get_manim_command(view, window):
    file_path = os.path.join(
        window.extract_variables()["file_path"],
        window.extract_variables()["file_name"],
    )

    # Pull out lines of file
    contents = view.substr(sublime.Region(0, view.size()))
    all_lines = contents.split("\n")

    # Find which lines define classes
    class_lines = [
        (line, all_lines.index(line))
        for line in contents.split("\n")
        if re.match(r"class (.+?)\((.+?)\):", line)
    ]

    # Where is the cursor
    row, col = view.rowcol(view.sel()[0].begin())

    # Find the first class defined before where the cursor is
    try:
        matching_class_str, scene_line_no = next(filter(
            lambda cl: cl[1] <= row,
            reversed(class_lines)
        ))
    except StopIteration:
        raise Exception("No matching classes")
    scene_name = matching_class_str[len("class "):matching_class_str.index("(")]

    cmds = ["manim", file_path, scene_name]
    enter = False

    if row != scene_line_no:
        cmds.append(f"-se {row + 1}")
        enter = True

    return " ".join(cmds), enter


def send_terminus_command(
    command,
    clear=True,
    center=True,
    enter=True,
    terminal_name="Manim",
):
    # Find terminus window
    terminal_sheet = find_terminus_sheet(terminal_name)
    if terminal_sheet is None:
        return
    window = terminal_sheet.window()
    view = terminal_sheet.view()

    # Focus the terminal first
    window.focus_view(view)

    # Small delay to ensure focus has taken effect
    def send_after_focus():
        _, col = view.rowcol(view.size())
        # Ammend command with various keyboard shortcuts
        full_command = "".join([
            "\x7F" * col if clear else "",  # Bad hack
            "\x0C" if center else "",  # Command + l
            command,
            "\n" if enter else "",
        ])
        window.run_command("terminus_send_string", {"string": full_command})

    sublime.set_timeout(send_after_focus, 50)  # 50ms delay


def find_terminus_sheet(terminal_name=None):
    for win in sublime.windows():
        for sheet in win.sheets():
            name = sheet.view().name()
            if terminal_name:
                if name == terminal_name:
                    return sheet
            else:
                # Default behavior: find any terminus terminal
                if name in ["Login Shell", "Manim", "Claude Terminal"] or name.startswith("IPython:"):
                    return sheet
    return None


def is_terminal_in_ipython(terminal_name="Manim"):
    """
    Check if the terminal is currently in an IPython session
    by looking at the last line for IPython prompt pattern
    """
    terminal_sheet = find_terminus_sheet(terminal_name)
    if terminal_sheet is None:
        return False

    view = terminal_sheet.view()
    # Get the last line of the terminal
    last_line_region = view.line(view.size())
    last_line = view.substr(last_line_region).strip()

    # Check if it matches IPython prompt pattern: "In [123]:" or similar
    ipython_pattern = r"^In\s*\[\d*\]:\s*$"
    return bool(re.match(ipython_pattern, last_line))


def ensure_terminus_tab_exists(terminal_name="Manim"):
    """
    If there is no sheet with the specified terminus tab,
    it opens a new window with one.
    Returns a timeout period suitable for
    following commands
    """
    if find_terminus_sheet(terminal_name) is None:
        sublime.run_command('new_window')
        new_window = next(reversed(sublime.windows()))
        new_window.run_command("terminus_open", {"title": terminal_name})
        return 500
    return 0


def checkpoint_paste_wrapper(view, arg_str="", terminal_name="Manim"):
    window = view.window()
    sel = view.sel()
    window.run_command("copy")

    # Modify the command based on the lines
    selected = sel[0]
    lines = view.substr(view.line(selected)).split("\n")
    first_line = lines[0].lstrip()
    starts_with_comment = first_line.startswith("#")

    if len(lines) == 1 and not starts_with_comment:
        command = view.substr(selected) if selected else first_line
    else:
        comment = first_line if starts_with_comment else "#"
        command = f"checkpoint_paste({arg_str}) {comment} ({len(lines)} lines)"

    # Clear selection and put cursor back to the start
    pos = sel[0].begin()
    sel.clear()
    sel.add(sublime.Region(pos))

    send_terminus_command(command, terminal_name=terminal_name)

    # Return focus to the original view
    window.focus_view(view)


class ManimRunScene(sublime_plugin.TextCommand):
    def run(self, edit):
        view = self.view
        window = view.window()
        window.run_command("save")

        # Check if we should reload instead of run fresh
        if is_terminal_in_ipython("Manim"):
            # Terminal is in IPython session, use reload instead
            row, col = view.rowcol(view.sel()[0].begin())
            send_terminus_command(f"reload({row + 1})", terminal_name="Manim")
            # Focus back to the original view within Sublime after a short delay
            sublime.set_timeout(
                lambda: window.focus_view(view),
                100
            )
            return

        # Terminal is not in IPython, proceed with normal run
        command, enter = get_manim_command(view, window)
        timeout = ensure_terminus_tab_exists("Manim")
        sublime.set_timeout(
            lambda: send_terminus_command(command, enter=enter, terminal_name="Manim"),
            timeout
        )


class ManimExit(sublime_plugin.TextCommand):
    def run(self, edit):
        send_terminus_command("\x03quit\n", center=False, terminal_name="Manim")
        time.sleep(0.01)
        send_terminus_command("", clear=False, center=True, enter=False, terminal_name="Manim")


class ManimCheckpointPaste(sublime_plugin.TextCommand):
    def run(self, edit):
        self.view.window().run_command("save")
        checkpoint_paste_wrapper(self.view)


class ManimRecordedCheckpointPaste(sublime_plugin.TextCommand):
    def run(self, edit):
        checkpoint_paste_wrapper(self.view, arg_str="record=True, progress_bar=False")


class ManimSkippedCheckpointPaste(sublime_plugin.TextCommand):
    def run(self, edit):
        checkpoint_paste_wrapper(self.view, arg_str="skip=True")


class ManimReload(sublime_plugin.TextCommand):
    def run(self, edit):
        self.view.window().run_command("save")
        row, col = self.view.rowcol(self.view.sel()[0].begin())
        send_terminus_command(f"reload({row + 1})", terminal_name="Manim")


class ManimRender(sublime_plugin.TextCommand):
    def run(self, edit):
        """Copy manim render command with --prerun --finder -w flags to clipboard"""
        view = self.view
        window = view.window()

        # Get the manim command
        command, enter = get_manim_command(view, window)

        # Add the flags and copy to clipboard
        full_command = command + " --prerun -w"
        sublime.set_clipboard(full_command)

        # Send to terminus
        ensure_terminus_tab_exists("Rendering")
        send_terminus_command(full_command, terminal_name="Rendering", enter=False)


class OpenMirroredDirectory(sublime_plugin.TextCommand):
    def run(self, edit):
        window = self.view.window()
        path = window.extract_variables()["file_path"]
        new_path = os.path.join(
            path.replace("_", "").replace(
                "/Users/grant/cs/videos",
                R"/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos"
            ),
            window.extract_variables()["file_name"].replace(".py", ""),
        )
        print(new_path)
        sp.call(["open", "-R", new_path])
