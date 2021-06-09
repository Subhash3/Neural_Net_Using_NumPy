#!/usr/bin/python3

import re
import os
import sys
import glob


def write_on_readme(filename, doc_text, readme_path='README.md'):
    with open(readme_path, 'r') as fp:
        readme_text = fp.read()
    print(filename)

    pattern = re.compile(r"#### \[`{}`.*\.md\)(.|\n)*?<br\/>".format(filename))

    matches = pattern.finditer(readme_text)

    # print(list(matches))
    # match = next(matches)
    for match in matches:
        start = match.span()[0]
        end = match.span()[1]

        prev_text = readme_text[:start]
        next_text = readme_text[end:]

        mid_text = f"#### [`{filename}`](https://github.com/Subhash3/Neural_Net_Using_NumPy/blob/master/docs/markdown/{filename}.md)\n"
        mid_text += doc_text
        mid_text += "\n\n<br/>"

        final_text = prev_text + mid_text + next_text

        with open('README.md', 'w') as fp:
            readme_text = fp.write(final_text)


def Main():
    argv = sys.argv
    argc = len(argv)

    # Copy the main README file
    os.system('cp ../README.md .')
    if argc != 2:
        print(f"Usage: {argv[0]} <md-files-directory | md-file-path>")
        quit()
    srcpath = argv[1]

    if os.path.isdir(srcpath):
        all_files = os.listdir(srcpath)

        for filename in all_files:
            filepath = os.path.join(srcpath, filename)
            filename = filename.split('.')[0]

            with open(filepath, 'r') as fp:
                doc_text = fp.read()

            write_on_readme(filename, doc_text)

    else:
        filename = srcpath.split('/')[-1].split('.')[0]

        with open(srcpath, 'r') as fp:
            doc_text = fp.read()

        write_on_readme(filename, doc_text)


if __name__ == "__main__":
    Main()
