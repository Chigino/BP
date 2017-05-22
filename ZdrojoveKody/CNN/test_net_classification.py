import caffe
import numpy as np
import sys


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
        return 0.1519
    elif (classed == 1):
        return 0.259
    elif (classed == 2):
        return 0.354
    elif (classed == 3):
        return 0.426
    elif (classed == 4):
        return 0.4748
    elif (classed == 5):
        return 0.5261
    elif (classed == 6):
        return 0.5749
    elif (classed == 7):
        return 0.6245
    elif (classed == 8):
        return 0.6749
    elif (classed == 9):
        return 0.7424
    else:
        return 0.8467

tstData = []
tstLabels = []

caffe.set_mode_cpu()
deployNet = caffe.Net('deploy.prototxt', sys.argv[1], caffe.TEST)

deployNet.blobs['data'].reshape(1, 2, 68, 350)
deployNet.reshape()


print "Loading file"
data = np.load('norm_batch1')
tstData.append(data['data'])
tstLabels.append(data['labels'])


tstData = np.reshape(tstData, (-1, 2, 68, 350))
tstLabels = np.reshape(tstLabels,(-1, 5))



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


print "Starting"
for i in range(0, np.size(tstData, 0)):
    deployNet.blobs['data'].data[...] = tstData[i, :, :, :]
    deployNet.forward()
    class_extra = np.argmax(deployNet.blobs['loss_extra'].data)
    class_agree = np.argmax(deployNet.blobs['loss_agree'].data)
    class_consc = np.argmax(deployNet.blobs['loss_consc'].data)
    class_neuro = np.argmax(deployNet.blobs['loss_neuro'].data)
    class_openn = np.argmax(deployNet.blobs['loss_openn'].data)
    if (class_extra == Classer(tstLabels[i,0])):
        extra_right += 1
    if (class_agree == Classer(tstLabels[i,1])):
        agree_right += 1
    if (class_consc == Classer(tstLabels[i,2])):
        consc_right += 1
    if (class_neuro == Classer(tstLabels[i,3])):
        neuro_right += 1
    if (class_openn == Classer(tstLabels[i,4])):
        openn_right += 1
    #print "{} to {}".format(class_extra, deClasser(class_extra))
    '''
    extra_diff.append(np.absolute(deClasser(class_extra) - tstLabels[i,0,0,0]))
    agree_diff.append(np.absolute(deClasser(class_agree) - tstLabels[i,1,0,0]))
    consc_diff.append(np.absolute(deClasser(class_consc) - tstLabels[i,2,0,0]))
    neuro_diff.append(np.absolute(deClasser(class_neuro) - tstLabels[i,3,0,0]))
    openn_diff.append(np.absolute(deClasser(class_openn) - tstLabels[i,4,0,0]))
    '''
    extra_diff.append(np.absolute(deClasser(deployNet.blobs['loss_extra'].data) - tstLabels[i,0]))
    agree_diff.append(np.absolute(deClasser(deployNet.blobs['loss_agree'].data) - tstLabels[i,1]))
    consc_diff.append(np.absolute(deClasser(deployNet.blobs['loss_consc'].data) - tstLabels[i,2]))
    neuro_diff.append(np.absolute(deClasser(deployNet.blobs['loss_neuro'].data) - tstLabels[i,3]))
    openn_diff.append(np.absolute(deClasser(deployNet.blobs['loss_openn'].data) - tstLabels[i,4]))
    
print "Extra distance = {} with accuracy of {}".format(np.average(extra_diff), float(extra_right)/np.size(tstData, 0))
print "Agree distance = {} with accuracy of {}".format(np.average(agree_diff), float(agree_right)/np.size(tstData, 0))
print "Consc distance = {} with accuracy of {}".format(np.average(consc_diff), float(consc_right)/np.size(tstData, 0))
print "Neuro distance = {} with accuracy of {}".format(np.average(neuro_diff), float(neuro_right)/np.size(tstData, 0))
print "Openn distance = {} with accuracy of {}".format(np.average(openn_diff), float(openn_right)/np.size(tstData, 0))
