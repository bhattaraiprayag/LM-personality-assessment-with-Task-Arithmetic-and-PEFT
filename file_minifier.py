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

def minify_codebase(root_dir, output_file):
    """
    Traverses the directory structure from the given root_dir, collects all .py files,
    processes them to remove comments and docstrings, and writes the minified content
    into a single file.
    """
    with open(output_file, 'w') as outfile:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py') and file != os.path.basename(__file__):
                    file_path = os.path.join(subdir, file)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        source_code = infile.read()
                        minified_code = remove_comments_and_docstrings(source_code)
                        outfile.write(f"# FILE: {file_path}\n")
                        outfile.write(minified_code + "\n\n")

if __name__ == "__main__":
    root_directory = "."  # Root directory to start scanning
    output_filename = "./outputs/min-codebase.txt"  # Output file for the minified code
    minify_codebase(root_directory, output_filename)
    print(f"Minified codebase saved to {output_filename}")
