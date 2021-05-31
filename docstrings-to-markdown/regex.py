# Extract class or function along with its docstring
# Docstring will be captured in group 1 and group 0 contains the whole class or function along with its docstring.
def_with_docs_regex = r'((def|class).*((\s*->\s*.*)|):\n\s*"""(\n\s*.*?)*""")'

# Given a docstring, identity each part, i.e, parameters and return values.
# Note: This regex works only with numpy style docstrings.
# Part          Captured Group
# Description           1
# Parameters            3
# Returns               6
identify_each_part_regex = r'"""\n\s*(.*\s*)*?(Parameters\s*-*\s*((.*\s*)*?))?(Returns\s*-*\s*(.*\s*)*?)?"""'
