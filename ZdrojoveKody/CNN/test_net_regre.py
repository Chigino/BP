import caffe
import numpy as np
import cPickle
import sys

class Data(object):
    data = []
    label = []

    def __init__(self):
        self.data = []
        self.label = []

    def add_data(self, data, label):
        self.data = np.append(self.data, data)
        self.label = np.append(self.label, label)

tstData = []
tstLabels = []

caffe.set_mode_cpu()
deployNet = caffe.Net('net_train_val.prototxt', sys.argv[1], caffe.TEST)

deployNet.blobs['data'].reshape(1, 1, 256, 90)
deployNet.blobs['labels'].reshape(1, 5, 1, 1)
deployNet.reshape()

with open('pickle11') as f:
        print "Loading file"
        data = cPickle.load(f)
        tstData.append(data.data)
        tstLabels.append(data.label)


tstData = np.reshape(tstData, (-1, 1, 256, 750))
tstLabels = np.reshape(tstLabels,(-1, 5, 1, 1))

tstData = tstData.astype(np.float32) / 255 - 0.5
#print "{} {}".format(tstData.shape, tstLabels.shape)

extra_diff = []
agree_diff = []
consc_diff = []
neuro_diff = []
openn_diff = []


print "Starting"
for i in range(0, np.size(tstData, 0)):
    deployNet.blobs['data'].data[...] = tstData[i, :, :, 50:50+90]
    deployNet.blobs['labels'].data[...] = tstLabels[i].reshape(1, 5, 1, 1)
    deployNet.forward()
    predicted = deployNet.blobs['fc3'].data
    extra_diff.append(np.absolute(predicted[0,0] - tstLabels[i,0,0,0]))
    agree_diff.append(np.absolute(predicted[0,1] - tstLabels[i,1,0,0]))
    consc_diff.append(np.absolute(predicted[0,2] - tstLabels[i,2,0,0]))
    neuro_diff.append(np.absolute(predicted[0,3] - tstLabels[i,3,0,0]))
    openn_diff.append(np.absolute(predicted[0,4] - tstLabels[i,4,0,0]))

print "Extra distance = {}".format(np.average(extra_diff))
print "Agree distance = {}".format(np.average(agree_diff))
print "Consc distance = {}".format(np.average(consc_diff))
print "Neuro distance = {}".format(np.average(neuro_diff))
print "Openn distance = {}".format(np.average(extra_diff))
