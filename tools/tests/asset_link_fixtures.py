"""Temporary-directory link fixtures, including privilege-free Windows junctions."""
import os
from pathlib import Path
import subprocess


def link_directory(link, target):
    link, target = Path(link), Path(target)
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            raise
    # Literal PowerShell strings; these are caller-owned temporary fixture dirs.
    literal = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = f"New-Item -ItemType Junction -Path {literal(link)} -Target {literal(target)} -ErrorAction Stop | Out-Null"
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                   check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
