import os
import tokenize
from io import StringIO


def remove_comments_and_docstrings(source):
    """
    Removes comments, docstrings, and empty lines from the Python source code.
    """
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0

    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type, token_string, (start_line, start_col), (end_line, end_col), _ = tok
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += " " * (start_col - last_col)
        if token_type == tokenize.COMMENT:
            continue
        elif token_type == tokenize.STRING:
            # Ignore docstrings
            if prev_toktype != tokenize.INDENT and prev_toktype != tokenize.NEWLINE:
                out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line

    # Remove empty lines
    out = '\n'.join(line for line in out.splitlines() if line.strip())
    return out


def minify_codebase(root_dir, output_file, exclude_files=None, preserve_files=None):
    """
    Processes .py files under root_dir, minifying most but fully preserving
    specified files. Skips excluded files.
    """
    exclude_files = set(exclude_files or [])
    preserve_files = set(preserve_files or [])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if not file.endswith('.py'):
                    continue

                rel_path = os.path.relpath(os.path.join(subdir, file), root_dir)
                if file in exclude_files or rel_path in exclude_files:
                    continue
                file_path = os.path.join(subdir, file)

                with open(file_path, 'r', encoding='utf-8') as infile:
                    source_code = infile.read()
                    if file in preserve_files or rel_path in preserve_files:
                        processed_code = source_code  # Preserve fully
                    else:
                        processed_code = remove_comments_and_docstrings(source_code)

                    outfile.write(f"# FILE: {file_path}\n")
                    outfile.write(processed_code + "\n\n")


if __name__ == "__main__":
    root_directory = "."  # Start scanning here
    output_filename = "./outputs/min-codebase.txt"

    # File-level filters (filename or relative path from root)
    excluded_files = {"file_minifier.py"}
    # # preserved_files = {"important_script.py", "nested_folder/special_case.py"}
    # preserved_files = {"experiment_config.py"}
    preserved_files = {}

    minify_codebase(
        root_directory,
        output_filename,
        exclude_files=excluded_files,
        preserve_files=preserved_files
    )
    print(f"Minified codebase saved to {output_filename}")
