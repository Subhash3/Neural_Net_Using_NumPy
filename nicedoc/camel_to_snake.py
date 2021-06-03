#!/usr/bin/python3

import re
import sys
import time
import regex_utils


def convert_camel_to_snake(data):
    matches = re.finditer(regex_utils.EXTRACT_CAMEL, data, re.MULTILINE)

    camel_to_snake_map = dict()
    for match in matches:
        # print(match.groups())
        beggining, camel_stuff, _, _, _ = match.groups()

        # Convert all the groups of uppercase letters to lowercase and prepend them with '_'.
        def callback(pattern): return f"_{pattern.group(1).lower()}"

        pascal_to_snake = re.sub(r'([A-Z]+)', callback, camel_stuff)

        print(camel_stuff, "=>", pascal_to_snake)

        # store the required changes to be made in a dictionary
        camel_to_snake_map[camel_stuff] = pascal_to_snake

    # Now replace all the camel-cases with snake-cases.
    new_data = data
    for camel_case in camel_to_snake_map:
        snake_case = camel_to_snake_map[camel_case]
        new_data = new_data.replace(camel_case, snake_case)

    return new_data


def Main():
    argv = sys.argv
    argc = len(argv)

    if argc < 2 or argc > 3:
        print(f"Usage: {argv[0]} <py-file> [overwrite-flag=1/0]")
        quit()

    overwrite_flag = False
    if argc > 2 and argv[2] == '1':
        overwrite_flag = True

    filename = argv[1]
    with open(filename, 'r+') as fp:
        data = fp.read()

        start = time.time()
        new_data = convert_camel_to_snake(data)
        end = time.time()

        print(end - start)

    if overwrite_flag:
        outfile = filename
    else:
        outfile = f"{filename}.out"
    with open(outfile, 'w') as fp_out:
        fp_out.write(new_data)
        print(f"Successfully wrote new data into {outfile}")


if __name__ == '__main__':
    Main()
