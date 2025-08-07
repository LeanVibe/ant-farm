import os


def remove_trailing_whitespace(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, encoding="utf-8") as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    try:
                        with open(filepath, encoding="latin-1") as f:
                            lines = f.readlines()
                    except Exception as e:
                        print(f"Could not read {filepath}: {e}")
                        continue

                with open(filepath, "w", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line.rstrip() + "\n")


if __name__ == "__main__":
    remove_trailing_whitespace(".")
