import numpy as np
import cPickle
import caffe

trnData = []
trnLabels = []
tstData = []
tstLabels = []

# class of pickled object
class Data(object):
    data = []
    label = []

    def __init__(self):
        self.data = []
        self.label = []

    def add_data(self, data, label):
        self.data = np.append(self.data, data)
        self.label = np.append(self.label, label)

def testNet(testNet, tstData, tstLabels, newTstLabels):
    batchSize = 1
    # change batch size
    testNet.blobs['data'].reshape(1, 1, 256, 256)
    testNet.blobs['labels'].reshape(1, 5, 1, 1)
    testNet.reshape()
    loss = 0
    counter = 0
    for i in range(0, 545):
        testNet.blobs['data'].data[...] = tstData[pos:pos + batchSize,:, :, 256:512]
        testNet.blobs['labels'].data[...] = newTstLabels[pos:pos + batchSize].reshape(-1,5,1,1)
        testNet.forward()
        print testNet.blobs['distance'].data
    return


# Loading pickled files
for i in range(1,11):
    with open('pickle{}'.format(i)) as f:
        print "Loading {}. file".format(i)
        data = cPickle.load(f)
        if (i == 11):
            tstData.append(data.data)
            tstLabels.append(data.label)
        else:
            trnData.append(data.data)
            trnLabels.append(data.label)

"""
with open('pickle1') as f:
        print "Loading file"
        data = cPickle.load(f)
        trnData.append(data.data)
        trnLabels.append(data.label)
        tstData.append(data.data)
        tstLabels.append(data.label)
"""

print "Reshaping"
trnData = np.reshape(trnData, (-1, 256, 750))
trnData = np.expand_dims(trnData, axis=1)
tstData = np.reshape(tstData, (-1, 256, 750))
tstData = np.expand_dims(tstData, axis=1)
print trnData.shape
trnLabels = np.reshape(trnLabels, (-1, 5))
tstLabels = np.reshape(trnLabels, (-1, 5))
"""
newValues = []
newTrnLabels = []

for labels in trnLabels:
    for value in labels:
        if (value < 0.2):
            newValues.append(0)
        elif (value < 0.3):
            newValues.append(1)
        elif (value < 0.4):
            newValues.append(2)
        elif (value < 0.45):
            newValues.append(3)
        elif (value < 0.5):
            newValues.append(4)
        elif (value < 0.55):
            newValues.append(5)
        elif (value < 0.6):
            newValues.append(6)
        elif (value < 0.65):
            newValues.append(7)
        elif (value < 0.7):
            newValues.append(8)
        elif (value < 0.8):
            newValues.append(9)
        else:
            newValues.append(10)
    newTrnLabels.append(newValues)
    newValues = []
newTrnLabels = np.reshape(newTrnLabels, (-1, 5, 1, 1))    
newValues = []
newTstLabels = []
for labels in trnLabels:
    for value in labels:
        if (value < 0.2):
            newValues.append(0)
        elif (value < 0.3):
            newValues.append(1)
        elif (value < 0.4):
            newValues.append(2)
        elif (value < 0.45):
            newValues.append(3)
        elif (value < 0.5):
            newValues.append(4)
        elif (value < 0.55):
            newValues.append(5)
        elif (value < 0.6):
            newValues.append(6)
        elif (value < 0.65):
            newValues.append(7)
        elif (value < 0.7):
            newValues.append(8)
        elif (value < 0.8):
            newValues.append(9)
        else:
            newValues.append(10)
    newTstLabels.append(newValues)
    newValues = []

newTstLabels = np.reshape(newTstLabels, (-1, 5, 1, 1))
"""
#normalizing data
trnData = trnData.astype(np.float32) / 255 - 0.5
tstData = tstData.astype(np.float32) / 255 - 0.5
print "Normalized" 

#caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.get_solver('net_solver.prototxt')
print "Successfully loaded solver"

batchSize = solver.net.blobs['data'].data.shape[0]
print batchSize
print('Blobs:', solver.net.blobs.keys())
testIter = 5000


print "Starting training"
limit = trnData.shape[0] - batchSize
pos = 1
posW = 12345 % 660
for i in range(100000):
    pos = (pos + batchSize * i) % limit
    solver.net.blobs['data'].data[...] = trnData[pos:pos + batchSize,:, :, posW:posW + 90]
    solver.net.blobs['labels'].data[...] = trnLabels[pos:pos + batchSize].reshape(-1,5,1,1)
    posW = (posW * pos) % 660
    solver.step(1)
    """
    if solver.iter % testIter == 0:
        testNet(solver.test_nets[0], tstData, tstLabels, newTstLabels)
    """ 
    
solver.net.save('trained.model')



