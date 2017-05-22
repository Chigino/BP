import re
import numpy as np

regex = [None] * 11
regex[0] = re.compile(",(0.[0-1]\d*)")
regex[1] = re.compile(",(0.[2]\d*)")
regex[2] = re.compile(",(0.[3]\d*)")
regex[3] = re.compile(",(0.4[0-4]\d*)")
regex[4] = re.compile(",(0.4[5-9]\d*)")
regex[5] = re.compile(",(0.5[0-4]\d*)")
regex[6] = re.compile(",(0.5[5-9]\d*)")
regex[7] = re.compile(",(0.6[0-4]\d*)")
regex[8] = re.compile(",(0.6[5-9]\d*)")
regex[9] = re.compile(",(0.[7]\d*)")
regex[10] = re.compile(",(1.\d*|0.[8-9]\d*)")

with open('values', 'r') as f:
	content = f.read()
for i in range(0,11):
	found = re.findall(regex[i], content)
	values = []
	for val in found:
		values.append(float(val))

	values = np.asarray(values)
	print "For {}. class the average is: {}".format(i, np.average(values))