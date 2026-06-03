from pathlib import Path

IGNORE = {
    ".git",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules",
    "build",
    "dist",
    ".DS_Store",
}

def tree(dir_path: Path, prefix=""):
    files = sorted(
        [p for p in dir_path.iterdir() if p.name not in IGNORE],
        key=lambda x: (x.is_file(), x.name.lower())
    )

    for i, path in enumerate(files):
        connector = "└── " if i == len(files) - 1 else "├── "
        print(prefix + connector + path.name)

        if path.is_dir():
            extension = "    " if i == len(files) - 1 else "│   "
            tree(path, prefix + extension)

if __name__ == "__main__":
    root = Path(".")
    print(root.resolve().name + "/")
    tree(root)