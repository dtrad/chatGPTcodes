# read from commnad line the name of a file and a pattern to remove
# remove all lines that contain the pattern
# write the result to a new file
# usage: python removelines.py filename pattern
# example: python removelines.py test.txt hello

import sys
import re
import os
with open(sys.argv[1]) as f:
    lines = f.readlines()
pattern = sys.argv[2]   
newlines = [line for line in lines if not re.search(pattern, line)]
newfile = os.path.splitext(sys.argv[1])[0] + '_new.txt'
with open(newfile, 'w') as f:
    f.writelines(newlines)
print('New file:', newfile)

          