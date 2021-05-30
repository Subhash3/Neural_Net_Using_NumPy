#!/usr/bin/python3

import re
import typing
import sys
import os


def get_all_docstrings(pycode):
    indices = [m.start() for m in re.finditer('"""', pycode)]

    doc_strings = list()
    for i in range(0, len(indices), 2):
        x = indices[i]
        y = indices[i+1]

        doc_strings.append(pycode[x+3:y])

    return doc_strings


def style_doc_strings(doc_strings: typing.List[str]):
    for ds in doc_strings:
        print(ds)
        data = re.split(r"\n\s*Parameters\n\s*-*\n", ds)
        desc = data[0]
        print(desc.strip())
        remaining = data[1]
        data = re.split(r"\n\s*Returns\n\s*-*\n", remaining)
        params_string = data[0]
        remaining = data[1]
        print(params_string)
        print(remaining)
        break


def generate_markdown_api(pycode, md_filename):
    def_with_docs_regex = r'((def|class).*((\s*->\s*.*)|):\n\s*"""(\n\s*.*?)*""")'
    definitions = re.findall(def_with_docs_regex, pycode, re.MULTILINE)

    md_defs = list()
    for definition in definitions:
        def_with_docstring_group = definition[0]
        # just_def = def_with_docstring_group.split('\n')[0]
        md = f"```python\n{def_with_docstring_group}\n```\n"
        md_defs.append(md)
        # print(md)
        # break

    markdown = '\n\n'.join(md_defs)
    markdown.strip()
    with open(md_filename, 'w') as fp:
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
            if srcfile.endswith('.py'):
                filename = srcfile.split('/')[-1].split('.')[0]
                md_file = f"{filename}.md"
                outfile_path = os.path.join(outdir_path, md_file)
                # print(srcfile)
                # print(filename)
                # print(outfile_path)

                with open(srcfile) as f:
                    pycode = f.read()
                    generate_markdown_api(pycode, outfile_path)
    else:
        filename = srcpath.split('/')[-1].split('.')[0]
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
