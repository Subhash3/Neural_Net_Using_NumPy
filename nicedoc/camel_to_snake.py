#!/usr/bin/python3

import re
import sys
import time
from . import regex as my_regex


def convert_camel_to_snake(data):
    matches = re.finditer(my_regex.EXTRACT_CAMEL, data)

    camel_to_snake_map = dict()
    for match in matches:
        # print(match.groups())
        beggining, lowercase_stuff, pascal_stuff, _last_pascal_group = match.groups()
        camel_case_to_replace = f"{beggining}{lowercase_stuff}{pascal_stuff}"

        # Convert all the groups of uppercase letters to lowercase and prepend them with '_'.
        def callback(pattern): return f"_{pattern.group(1).lower()}"

        pascal_to_snake = re.sub(r'([A-Z]+)', callback, pascal_stuff)
        final_snake_case = f"{beggining}{lowercase_stuff}{pascal_to_snake}"

        print(camel_case_to_replace, "=>", final_snake_case)

        # store the required changes to be made in a dictionary
        camel_to_snake_map[camel_case_to_replace] = final_snake_case

    # Now replace all the camel-cases with snake-cases.
    new_data = data
    for camel_case in camel_to_snake_map:
        snake_case = camel_to_snake_map[camel_case]
        new_data = new_data.replace(camel_case, snake_case)

    return new_data


def Main():
    argv = sys.argv
    argc = len(argv)

    if argc != 2:
        print(f"Usage: {argv[0]} <py-file>")
        quit()

    filename = argv[1]
    with open(filename, 'r') as fp:
        data = fp.read()

        start = time.time()
        new_data = convert_camel_to_snake(data)
        end = time.time()

        print(end - start)

        with open(f"{filename}.out", 'w') as fp_out:
            fp_out.write(new_data)


if __name__ == '__main__':
    Main()
