#!/usr/bin/python3

import os
import re
import sys

import regex as my_regex

# <details open>
# <summary><code>def __init__(self, num_nodes, inputs, activation_function, loss_function)</code></summary>
# <p>

# ```python
# def __init__(self, num_nodes, inputs, activation_function, loss_function):
#         """
#             Layer constructor

#             Parameters
#             ----------
#             num_nodes : int
#                 No. of nodes in the layer

#             inputs : int
#                 No. of inputs to the layer

#             activation_function

#             Returns
#             -------
#             None
#         """
# ```
# </p>
# </details>


def parse_docstring(docstring):
    parts = re.findall(my_regex.IDENTIFY_EACH_PART_REGEX, docstring, re.MULTILINE)
    groups = parts[0]
    description = groups[0].strip()
    parameters = groups[2].strip()
    returns = groups[5].strip()
    print(description)
    print(parameters)
    print(returns)
    # print(len(parts), len(groups))
    print("-" * 30)


def generate_markdown_api(pycode, md_filename):
    definitions = re.findall(my_regex.DEF_WITH_DOCS_REGEX, pycode, re.MULTILINE)

    md_defs = list()
    for definition in definitions:
        def_with_docstring_group = definition[0]
        just_def = def_with_docstring_group.split("\n")[0]
        just_docstring = "\n".join(def_with_docstring_group.split("\n")[1:])

        summary = f"<summary><code>{just_def}</code></summary>"
        p_tag = f"<p>\n\n```python\n{def_with_docstring_group}\n```\n</p>"
        details = f"<details>{summary}\n{p_tag}\n</details>"

        md = details
        # md = f"```python\n{def_with_docstring_group}\n```\n"

        md_defs.append(md)

    markdown = "\n\n".join(md_defs)
    markdown.strip()
    with open(md_filename, "w") as fp:
        fp.write(markdown)
        print("Successfully written markdown into ", md_filename)

    return md_defs, markdown


def Main():
    argv = sys.argv
    argc = len(argv)
    if argc != 3:
        print(f"Usage: {argv[0]} <py-file|src-dir> <out-dir>")
        quit()
    srcpath = argv[1]
    outdir_path = argv[2]

    if os.path.isdir(srcpath):
        all_files = os.listdir(srcpath)

        for filename in all_files:
            srcfile = os.path.join(srcpath, filename)
            if srcfile.endswith(".py"):
                filename = srcfile.split("/")[-1].split(".")[0]
                md_file = f"{filename}.md"
                outfile_path = os.path.join(outdir_path, md_file)
                # print(srcfile)
                # print(filename)
                # print(outfile_path)

                with open(srcfile) as f:
                    pycode = f.read()
                    generate_markdown_api(pycode, outfile_path)
    else:
        filename = srcpath.split("/")[-1].split(".")[0]
        md_file = f"{filename}.md"
        outfile_path = os.path.join(outdir_path, md_file)
        # print(srcpath)
        # print(filename)
        # print(outfile_path)

        with open(srcpath) as f:
            pycode = f.read()
            generate_markdown_api(pycode, outfile_path)


if __name__ == "__main__":
    Main()
