import numpy as np
import sys
import caffe
from PIL import Image

def init_caffe(self, solver_file):
    """
    Initialize the caffe solver.
    """
    if self.use_gpu:
        self.log.info(" Using GPU id {}".format(self.gpu_id))
        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
    else:
        self.log.info(" Using CPU")
        caffe.set_mode_cpu()

    self.log.info(" Reading solver file: %s", solver_file)
    solver = caffe.get_solver(solver_file)

    if self.caffe_solver_state:
        self.log.info(" Loading solver state: {}".format(self.caffe_solver_state))
        solver.restore(self.caffe_solver_state)
    elif self.caffe_weights:
        self.log.info(" Loading weights: %s", self.caffe_weights)
        solver.net.copy_from(self.caffe_weights)
    return solver


fName = sys.argv[1]
extra,agree,consc,neuro,openness = sys.argv[2].split(",")

net = caffe.Net('places205CNN_deploy.prototxt', 'places205CNN_iter_300000.caffemodel', caffe.TEST)
size = 227,227

for i in range(1,10):
	im = Image.open(fName + '-' + str(i) + '.jpeg')
	w, h = im.size
	if (w > h):
		x = (w-h)/2
		im = im.crop((x, 0, w-x, h))
	else:
		x = (h-w)/2
		im = im.crop((0, x, w, h-x))
	im.thumbnail(size, Image.ANTIALIAS)
	img = np.array(im)
	imgBlob = np.expand_dims(img.transpose(2, 0, 1), axis=0)
	if (i == 1):
		blob = imgBlob
	else:
		blob += imgBlob

net.blobs['data'].data[...] = blob

net.forward()
#print net.blobs['fc8'].data

iterator = 1
activations = ""
arr = net.blobs['fc8'].data
amin = np.amin(arr)
amax = np.amax(arr)

print amin
print amax
for number in np.nditer(arr):
	if (number != 0):
		num = 2*((number - amin)/(amax - amin))-1
		activations = activations + ' ' + str(iterator) + ':' + str(num)
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
