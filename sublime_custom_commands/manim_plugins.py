import sublime_plugin
import sublime
import os
import subprocess as sp
import threading
import re
import time


def get_command(view, window):
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

    cmds = ["manimgl", file_path, scene_name]
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
):
    # Find terminus window
    terminal_sheet = find_terminus_sheet()
    if terminal_sheet is None:
        return
    window = terminal_sheet.window()
    view = terminal_sheet.view()
    _, col = view.rowcol(view.size())

    # Ammend command with various keyboard shortcuts
    full_command = "".join([
        "\x7F" * col if clear else "",  # Bad hack
        "\x0C" if center else "",  # Command + l
        command,
        "\n" if enter else "",
    ])
    window.run_command("terminus_send_string", {"string": full_command})


def find_terminus_sheet():
    for win in sublime.windows():
        for sheet in win.sheets():
            name = sheet.view().name()
            if name == "Login Shell" or name.startswith("IPython: "):
                return sheet
    return None


def ensure_terminus_tab_exists():
    """
    If there is no sheet with a terminus tab,
    it opens a new window with one.
    Returns a timeout period suitable for
    following commands
    """
    if find_terminus_sheet() is None:
        sublime.run_command('new_window')
        new_window = next(reversed(sublime.windows()))
        new_window.run_command("terminus_open")
        return 500
    return 0


def checkpoint_paste_wrapper(view, arg_str=""):
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

    send_terminus_command(command)


class ManimRunScene(sublime_plugin.TextCommand):
    def run(self, edit):
        view = self.view
        window = view.window()
        window.run_command("save")
        command, enter = get_command(view, window)
        # If one wants to run it in a different terminal,
        # it's often to write to a file
        sublime.set_clipboard(command + " --prerun --finder -w")

        timeout = ensure_terminus_tab_exists()
        sublime.set_timeout(
            lambda: send_terminus_command(command, enter=enter),
            timeout
        )

        if enter:
            # Keep cursor where it started
            sublime.set_timeout(
                lambda: threading.Thread(target=self.focus_sublime).start(),
                1000
            )
        else:
            # Put cursor in terminus window
            sheet = find_terminus_sheet()
            if sheet is not None:
                window.focus_view(sheet.view())

    def focus_sublime(self):
        cmd = "osascript -e 'tell application \"Sublime Text\" to activate'"
        sp.call(cmd, shell=True)


class ManimExit(sublime_plugin.TextCommand):
    def run(self, edit):
        send_terminus_command("\x03quit\n", center=False)
        time.sleep(0.01)
        send_terminus_command("", clear=False, center=True, enter=False)


class ManimCheckpointPaste(sublime_plugin.TextCommand):
    def run(self, edit):
        checkpoint_paste_wrapper(self.view)


class ManimRecordedCheckpointPaste(sublime_plugin.TextCommand):
    def run(self, edit):
        checkpoint_paste_wrapper(self.view, arg_str="record=True")


class ManimSkippedCheckpointPaste(sublime_plugin.TextCommand):
    def run(self, edit):
        checkpoint_paste_wrapper(self.view, arg_str="skip=True")


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


class CommentFold(sublime_plugin.TextCommand):
    def run(self, edit):
        view = self.view
        regions = view.sel()
        regions_to_fold = []
        for region in regions:
            reg_str = view.substr(region)

            lines = reg_str.split("\n")
            view_index = region.begin()

            indent_level = None
            last_full_line_end = view_index
            last_comment_line_end = None
            last_line_was_comment = False
            for line in lines:
                line_end_point = view_index + len(line)
                if line.lstrip().startswith("#"):
                    if indent_level is None:
                        indent_level = len(line) - len(line.lstrip())
                    if len(line) - len(line.lstrip()) == indent_level and not last_line_was_comment:
                        if last_comment_line_end:
                            regions_to_fold.append(sublime.Region(
                                last_comment_line_end,
                                last_full_line_end,
                            ))
                        last_comment_line_end = line_end_point
                else:
                    last_line_was_comment = False
                if line.strip():
                    last_full_line_end = line_end_point
                view_index = line_end_point + 1
            if last_comment_line_end:
                regions_to_fold.append(sublime.Region(
                    last_comment_line_end, region.end(),
                ))
        self.view.fold(regions_to_fold)
