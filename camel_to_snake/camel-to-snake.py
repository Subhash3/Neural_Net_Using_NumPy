import re
import regex_utils as ru

ans = dict()
def callback(pattern): 
    return f"_{pattern.group(1).lower()}"

def callback1(pattern):
    print(pattern.group(2), "=>" , ans[pattern.group(2)])
    return pattern.group(1) + ans[pattern.group(2)]

print("enter filename")
filename = input()
file = open(filename, 'r')
data = file.read()
reg = list(set(re.findall(ru.EXTRACT_CAMEL, data)))
# print(reg)
for i in range(len(reg)):
    temp = reg[i][1]
    while re.search(r"[A-Z]",temp):
        temp = re.sub(r"([A-Z]+)", callback, temp)
    ans[reg[i][1]] = temp
# print(ans)
while re.search(ru.EXTRACT_CAMEL, data):
    print("blah")
    data = re.sub(ru.EXTRACT_CAMEL, callback1, data)
# print(data)
file.close()
file = open(filename,'w')
file.write(data)
file.close()