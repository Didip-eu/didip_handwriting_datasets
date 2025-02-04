#!/usr/bin/env python3

import sys
import random

USAGE="{} <set1> <set2> <larger ratio>".format( sys.argv[0])

if len(sys.argv) < 4:
    print(USAGE)
    sys.exit()

file1, file2 = sys.argv[1:3]
ratio = float(sys.argv[-1])

count1, count2 = 0, 0
lines1, lines2 = [], []
with open(file1, 'r') as f1:
    lines1 = list([ line[:-1] for line in f1 ]) 
with open(file2, 'r') as f2:
    lines2 = list([ line[:-1] for line in f2 ]) 
count1, count2 = len(lines1), len(lines2)

actual_ratio = count1 / (count1 + count2 )
print("Actual ratio =", actual_ratio)

final_count1, final_count2 = 0, 0
# if actual share is smaller than desired, then take all of first set and compute second share
if actual_ratio <= ratio:
    final_count1 = count1
    print("(1/ratio)-1)=", 1/ratio-1)
    final_count2 = int(final_count1*((1/ratio) - 1))
# otherwise, take all of second set and compute first share
else:
    final_count2 = count2
    ratio_comp = 1-ratio
    final_count1 = int(final_count2*((1/ratio_comp)-1))

print("Count1\tCount2\tRatio\tFinal1\tFinal2\n{}\t{}\t{}\t{}\t{}".format(count1, count2, ratio, final_count1, final_count2))


sample_1 = random.sample( lines1, final_count1)
sample_2 = random.sample( lines2, final_count2)
print(sample_1)
print(sample_2)
