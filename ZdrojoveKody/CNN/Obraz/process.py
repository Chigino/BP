import os.path
import numpy as np

leftGazeMatrix = []
rightGazeMatrix = []
headPoseMatrix = []
landmarks = []

data = []
labels = []
gazing = []
j = 1
alls = 0

with open('../BP/first impression/values','r') as file:
    for line in file:
        name, extra, agree, consc, neuro, openn = line.split(',')
        label = np.array([float(extra), float(agree), float(consc), float(neuro), float(openn)])
        #exchange file extension
        name, num, ext = name.split('.')
        name = name + '.' + num + "_landmarks.txt"
        #print name
        if not (os.path.isfile(name)):
            continue
        with open(name, 'r') as f:
            for line in f:
                words = line.split()
                frameID = int(words[0])
                personID = int(words[1])
                headSize = [float(x) for x in words[2].split(':')]
                xSize = float(headSize[2] - headSize[0])
                ySize = float(headSize[3] - headSize[1])
                if (xSize < 1):
                    xSize = 1
                if (ySize < 1):
                    ySize = 1
                
                landmarksX = []
                landmarksY = []
                for i in range(3,71):
                    x,y = words[i].split(':')
                    landmarksX.append(int(x) / xSize)
                    landmarksY.append(int(y) / ySize)
                landmarksX = np.asarray(landmarksX) 
                landmarksY = np.asarray(landmarksY) 
                landmarks = np.append(landmarks, np.vstack((landmarksX,landmarksY)))
                """
                leftGaze = [float(x) for x in words[71].split(':')]
                rightGaze = [float(x) for x in words[72].split(':')]
                headPosition = [float(x) for x in words[73].split(':')]
                headPose = [float(x) for x in words[74].split(':')]

                leftGazeMatrix.append(leftGaze)
                rightGazeMatrix.append(rightGaze)
                headPoseMatrix.append(headPose)
                """

            """
            leftGazeMatrix = np.asarray(leftGazeMatrix)
            rightGazeMatrix = np.asarray(rightGazeMatrix)
            headPoseMatrix = np.asarray(headPoseMatrix)

            gazing = np.append(gazing, leftGazeMatrix)
            gazing = np.append(gazing, rightGazeMatrix)
            gazing = np.append(gazing, headPoseMatrix)

            leftGazeMatrix = []
            rightGazeMatrix = []
            headPoseMatrix = []
            gazing = np.reshape(gazing, (3, -1, 3))

            if not (gazing.shape[1] < 350):
                data = np.append(data, gazing[:,0:350,:])
                labels = np.append(labels, label)
                alls += 1
                print alls
                if (alls == 1000):
                    with open('headmove/batch{}'.format(j), 'w') as out:
                        np.savez(out, data=data, labels=labels)
                        j += 1
                        alls = 0
                        data = []
                        labels = []

            gazing = []
            """
            marks = []
            dimens = []
            landmarks = np.reshape(landmarks, (-1,2,68))
            if not (landmarks.shape[0] < 350):
                toNormal = landmarks[0:350,:,:]
                toNormal = np.transpose(toNormal, (1,2,0))
                for dims in toNormal:
                    for landmark in dims:
                        avg = np.average(landmark)
                        landmark = landmark - avg
                        marks = np.append(marks, landmark)
                    dimens = np.append(dims, marks)
                    marks = []
                dimens = np.asarray(dimens) 
                dimens = np.transpose(np.reshape(dimens, (2, 68, -1)), (2,0,1))
                print dimens.shape
                data = np.append(data,dimens)
                labels = np.append(labels, label)
                alls += 1
                print alls
                if (alls == 1000):
                    with open('landmarks/sizenorm_batch{}'.format(j), 'w') as out:
                        np.savez(out, data=data, labels=labels)
                        j += 1
                        alls = 0
                        data = []
                        labels = []

            landmarks = []
            
        #leftGazeMatrix = np.asarray(leftGazeMatrix)
        #rightGazeMatrix = np.asarray(rightGazeMatrix)
        #headPoseMatrix = np.asarray(headPoseMatrix)
        #print leftGazeMatrix.shape
        #print rightGazeMatrix.shape
        #print headPoseMatrix.shape
"""
with open('landmarks/batch{}'.format(j), 'w') as out:
    np.savez(out, data=data, labels=labels)
    j += 1
    data = []
    labels = []
"""
with open('landmarks/sizenorm_batch{}'.format(j), 'w') as out:
    np.savez(out, data=data, labels=labels)
