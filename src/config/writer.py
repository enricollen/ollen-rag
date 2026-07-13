"""In-place .env writer: update KEY=value lines while preserving the file's
comment/section structure, so the UI settings editor never clobbers the hand-authored layout."""
import re
from pathlib import Path

# Matches "KEY=value" with an optional trailing " # inline comment" (which python-dotenv strips on read).
_LINE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>.*?)(?P<comment>\s+#.*)?$")

def merge_into_env(env_path: Path, changes: dict[str, str]) -> None:
    """Replace the value of each changed key in place; append keys not already present.

    Preserves comments, blank lines, section banners and trailing inline comments.
    Creates the file when it does not yet exist (first save on a fresh install / volume).
    Guarantees no trailing blank line at EOF."""
    # A brand-new install (or an empty Docker config volume) has no .env yet: start from empty.
    text = env_path.read_text() if env_path.exists() else ""
    lines = text.split("\n") if text else []
    remaining = dict(changes)
    out: list[str] = []
    for line in lines:
        m = _LINE.match(line)
        if m and m.group("key") in remaining:
            key = m.group("key")
            comment = m.group("comment") or ""
            out.append(f"{key}={remaining.pop(key)}{comment}")
        else:
            out.append(line)
    # Append any keys that had no existing line
    for key, val in remaining.items():
        out.append(f"{key}={val}")
    # Drop trailing blanks, then join (house rule: no empty line at EOF)
    while out and out[-1] == "":
        out.pop()
    env_path.write_text("\n".join(out))