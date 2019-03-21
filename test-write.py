import os

def write():
    string = ''
    for i in range(5):
        string += str(i) + '\r'
    return string

with open('test-write.txt','w') as f:
    f.write(write())
