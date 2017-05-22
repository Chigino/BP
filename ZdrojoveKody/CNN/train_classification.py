import numpy as np
import caffe
import random
import logging

trnData = []
trnLabels = []
tstData = []
tstLabels = []


def Classer(value):
    if (value < 0.2):
        return 0
    elif (value < 0.3):
        return 1
    elif (value < 0.4):
        return 2
    elif (value < 0.45):
        return 3
    elif (value < 0.5):
        return 4
    elif (value < 0.55):
        return 5
    elif (value < 0.6):
        return 6
    elif (value < 0.65):
        return 7
    elif (value < 0.7):
        return 8
    elif (value < 0.8):
        return 9
    else:
        return 10

def deClasser(classed):
    return np.sum(classed.reshape(-1) * np.asarray([0.1519, 0.259, 0.354, 0.426, 0.4748, 0.5261, 0.5749, 0.6245, 0.6749, 0.7424, 0.8467]))
    if (classed == 0):
        return 0.15
    elif (classed == 1):
        return 0.25
    elif (classed == 2):
        return 0.35
    elif (classed == 3):
        return 0.425
    elif (classed == 4):
        return 0.475
    elif (classed == 5):
        return 0.525
    elif (classed == 6):
        return 0.575
    elif (classed == 7):
        return 0.625
    elif (classed == 8):
        return 0.675
    elif (classed == 9):
        return 0.75
    else:
        return 0.85


def testNet(testNet, tstData, tstLabels, newTstLabels):
    extra_diff = []
    agree_diff = []
    consc_diff = []
    neuro_diff = []
    openn_diff = []

    extra_right = 0
    agree_right = 0
    consc_right = 0
    neuro_right = 0
    openn_right = 0

    batchSize = 1
    # change batch size
    testNet.blobs['data'].reshape(1, 3, 350, 3)
    testNet.blobs['labels'].reshape(1, 5, 1, 1)
    testNet.reshape()

    for i in range(0, np.size(tstData, 0)):
        testNet.blobs['data'].data[...] = tstData[i, :, :, :]
        testNet.forward()
        class_extra = np.argmax(testNet.blobs['loss_extra'].data)
        class_agree = np.argmax(testNet.blobs['loss_agree'].data)
        class_consc = np.argmax(testNet.blobs['loss_consc'].data)
        class_neuro = np.argmax(testNet.blobs['loss_neuro'].data)
        class_openn = np.argmax(testNet.blobs['loss_openn'].data)
        if (class_extra == newTstLabels[i,0,0,0]):
            extra_right += 1
        if (class_agree == newTstLabels[i,1,0,0]):
            agree_right += 1
        if (class_consc == newTstLabels[i,2,0,0]):
            consc_right += 1
        if (class_neuro == newTstLabels[i,3,0,0]):
            neuro_right += 1
        if (class_openn == newTstLabels[i,4,0,0]):
            openn_right += 1
        extra_diff.append(np.absolute(deClasser(class_extra) - tstLabels[i,0]))
        agree_diff.append(np.absolute(deClasser(class_agree) - tstLabels[i,1]))
        consc_diff.append(np.absolute(deClasser(class_consc) - tstLabels[i,2]))
        neuro_diff.append(np.absolute(deClasser(class_neuro) - tstLabels[i,3]))
        openn_diff.append(np.absolute(deClasser(class_openn) - tstLabels[i,4]))

    logging.info("Extra distance = {} with accuracy of {}".format(np.average(extra_diff), float(extra_right)/np.size(tstData, 0)))
    logging.info("Agree distance = {} with accuracy of {}".format(np.average(agree_diff), float(agree_right)/np.size(tstData, 0)))
    logging.info("Consc distance = {} with accuracy of {}".format(np.average(consc_diff), float(consc_right)/np.size(tstData, 0)))
    logging.info("Neuro distance = {} with accuracy of {}".format(np.average(neuro_diff), float(neuro_right)/np.size(tstData, 0)))
    logging.info("Openn distance = {} with accuracy of {}".format(np.average(openn_diff), float(openn_right)/np.size(tstData, 0)))


# Start of the script 
#
#
#

# Loading pickled files
print "Loading data"
for i in range(1,7):
    temp = np.load("norm_batch{}".format(i))
    if (i == 1):
        tstData = np.reshape(temp['data'], (-1, 2, 68, 350))
        tstLabels = np.reshape(temp['labels'], (-1, 5))
    else:
        trnData = np.append(trnData, temp['data'])
        trnLabels= np.append(trnLabels, temp['labels'])
print "Loaded"
logging.basicConfig(filename='results.log',level=logging.DEBUG)

print "Reshaping"
trnData = np.asarray(trnData)
print trnData.shape
trnData = np.reshape(trnData, (-1, 2, 68, 350))
print trnData.shape
trnLabels = np.reshape(trnLabels, (-1, 5))

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
for labels in tstLabels:
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



#caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.get_solver('net_solver.prototxt')
print "Successfully loaded solver"
#solver.net.copy_from('deploy.caffemodel')

batchSize = solver.net.blobs['data'].data.shape[0]
print batchSize
print('Blobs:', solver.net.blobs.keys())
testIter = 10000


print "Starting training"
limit = trnData.shape[0] - batchSize
pos = 1

for i in range(500000):
    pos = random.randint(0,limit -1)
    solver.net.blobs['data'].data[...] = trnData[pos:pos + batchSize,:, :, :]
    solver.net.blobs['labels'].data[...] = newTrnLabels[pos:pos + batchSize].reshape(-1,5,1,1)
    solver.step(1)
    
    if solver.iter % testIter == 0:
        logging.info('Data at iteration {}'.format(solver.iter))
        testNet(solver.test_nets[0], tstData, tstLabels, newTstLabels)
    
    
solver.net.save('trained.model')



