import sys
import re

for line in sys.stdin:
    line = line.replace(u'✳','')
    line = line.replace('&#13','')
    #line = line.replace('ͣ', 'a')
    #line = line.replace('ͤ', 'e')
    #line = line.replace('ͦ','o')
    #line = line.replace('ͧ', 'u',)
    #line = line.replace('ƺ', '?')
    #line = line.replace('ꝰ', 'us')
    #line = line.replace('ꝝ', 'rum')
    line = line.replace('\u0304', '') # subscript '-'
    line = line.replace('\u0303' '') # tilde
    line = re.sub(r'  +',' ', line)

    print(line, end='')
