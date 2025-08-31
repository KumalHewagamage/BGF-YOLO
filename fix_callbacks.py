#!/usr/bin/env python3
"""
Script to fix ultralytics imports in callback files
"""

import os
import re
import glob


def fix_callback_file(file_path):
    """Fix ultralytics imports in a callback file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Replace ultralytics imports
        patterns = [
            (r"from ultralytics\.yolo\.utils import", "from yolo.utils import"),
            (
                r"from ultralytics\.yolo\.utils\.torch_utils import",
                "from yolo.utils.torch_utils import",
            ),
            (r"from ultralytics\.hub\.utils import", "from hub.utils import"),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)

        # Add sys.path modification if content changed and not already present
        if content != original_content and "sys.path.insert" not in content:
            lines = content.split("\n")

            # Find the first import line
            first_import_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(
                    ("import ", "from ")
                ) and not line.strip().startswith("from __future__"):
                    first_import_idx = i
                    break

            # Add sys.path modification
            sys_path_lines = [
                "import os",
                "import sys",
                "",
                "# Add the project root to the Python path",
                "project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))",
                "sys.path.insert(0, project_root)",
                "",
            ]

            # Insert before the first import
            lines = lines[:first_import_idx] + sys_path_lines + lines[first_import_idx:]
            content = "\n".join(lines)

        # Write the modified content back
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed imports in: {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix all callback files"""
    callback_dir = "/home/avishka/Kumal/BGF-YOLO/yolo/utils/callbacks"
    callback_files = glob.glob(os.path.join(callback_dir, "*.py"))

    fixed_count = 0
    for file_path in callback_files:
        if file_path.endswith("__init__.py") or file_path.endswith("base.py"):
            continue  # Skip these files
        if fix_callback_file(file_path):
            fixed_count += 1

    print(f"Fixed imports in {fixed_count} callback files")


if __name__ == "__main__":
    main()
