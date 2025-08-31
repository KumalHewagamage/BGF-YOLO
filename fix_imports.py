#!/usr/bin/env python3
"""
Script to fix relative imports in BGF-YOLO project
This script will replace all relative imports (from xxx import) with absolute imports
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import os
import re
from pathlib import Path


def fix_imports_in_file(file_path):
    """Fix relative imports in a single file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Pattern to match relative imports like "from xxx import"
        # This handles cases like:
        # from nn.modules import ...
        # from yolo.utils import ...
        # from hub.auth import ...

        # Replace three-dot imports (from xxx import)
        content = re.sub(r"from \.\.\.([^.\s]+)", r"from \1", content)

        # If content changed, add the sys.path modification at the top
        if content != original_content:
            lines = content.split("\n")

            # Find the first import line
            first_import_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(
                    ("import ", "from ")
                ) and not line.strip().startswith("from __future__"):
                    first_import_idx = i
                    break

            # Check if sys.path modification is already there
            has_sys_path = any(
                "sys.path" in line for line in lines[: first_import_idx + 10]
            )

            if not has_sys_path:
                # Calculate the relative path from current file to project root
                file_path_obj = Path(file_path)
                project_root = Path("/home/avishka/Kumal/BGF-YOLO")

                # Count how many levels deep we are from project root
                try:
                    relative_path = file_path_obj.relative_to(project_root)
                    levels_up = len(relative_path.parts) - 1
                except ValueError:
                    # File is not under project root, skip
                    return False

                # Create the path modification code
                dirname_calls = (
                    "os.path.dirname(" * (levels_up + 1)
                    + "os.path.abspath(__file__)"
                    + ")" * (levels_up + 1)
                )

                sys_path_lines = [
                    "import os",
                    "import sys",
                    "",
                    "# Add the project root to the Python path",
                    f"project_root = {dirname_calls}",
                    "sys.path.insert(0, project_root)",
                    "",
                ]

                # Insert before the first import
                lines = (
                    lines[:first_import_idx] + sys_path_lines + lines[first_import_idx:]
                )
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
    """Main function to fix all Python files in the project"""
    project_root = Path("/home/avishka/Kumal/BGF-YOLO")

    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(project_root):
        # Skip certain directories
        skip_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
        }
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    print(f"Found {len(python_files)} Python files")

    fixed_count = 0
    for file_path in python_files:
        if fix_imports_in_file(file_path):
            fixed_count += 1

    print(f"Fixed imports in {fixed_count} files")


if __name__ == "__main__":
    main()
