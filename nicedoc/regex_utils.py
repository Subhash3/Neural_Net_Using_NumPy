# Extract class or function along with its docstring
# Docstring will be captured in group 1 and group 0 contains the whole class or function along with its docstring.
DEF_WITH_DOCS_REGEX = r'((def|class).*((\s*->\s*.*)|):\n\s*"""(\n\s*.*?)*""")'

# Given a docstring, identity each part, i.e, parameters and return values.
# Note: This regex works only with numpy style docstrings.
# Part          Captured Group
# Description           1
# Parameters            3
# Returns               6
IDENTIFY_EACH_PART_REGEX = r'"""\n\s*(.*\s*)*?(Parameters\s*-*\s*((.*\s*)*?))?(Returns\s*-*\s*(.*\s*)*?)?"""'

# Given a python file, extracts all the camelcased variable/methods/functions.
EXTRACT_CAMEL = r"(^(?!\s*print.*).*)((\b[a-z]+)(([A-Z]+[a-z]+)+?)\b)"
