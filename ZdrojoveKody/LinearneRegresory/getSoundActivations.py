from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import sys
import numpy as np


fName = sys.argv[1]
extra,agree,consc,neuro,openness = sys.argv[2].split(",")


(rate,sig) = wav.read(fName)
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)

print fbank_feat.shape

iterator = 1
activations = ""

for number in np.nditer(fbank_feat):
	if (number != 0):
		activations = activations + ' ' + str(iterator) + ':' + str(number)
	iterator += 1


f = open('extra', 'a')
f.write(extra + ' ' + activations + '\n')
f.close()

f = open('agree', 'a')
f.write(agree + ' ' + activations + '\n')
f.close()

f = open('consc', 'a')
f.write(consc + ' ' + activations + '\n')
f.close()

f = open('neuro', 'a')
f.write(neuro + ' ' + activations + '\n')
f.close()

f = open('openness', 'a')
f.write(openness + ' ' + activations + '\n')
f.close()