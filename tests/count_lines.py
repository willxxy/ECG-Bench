import os


def count_lines(directory):
    total_lines = 0
    code_lines = 0
    total_size = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".py", ".rs")):
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                total_size += size

                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()
                    total_lines += len(lines)

                    code_lines += sum(1 for line in lines if line.strip() and not line.strip().startswith(("#", "//")))

    return total_lines, code_lines


if __name__ == "__main__":
    directory = "ecg_bench"
    total, loc = count_lines(directory)
    print(f"{total} lines ({loc} loc)")
