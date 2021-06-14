import os

# @TODO 
# use subprocess instead of os.system
# change paths

PATH = '../dataset/'

files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(PATH)) for f in fn]

for f in files:
    cmd = 'python ./hed-edge-detector/edge_detector.py --input ' + f + \
        ' --prototxt ./hed-edge-detector/deploy.prototxt --caffemodel ./hed-edge-detector/hed_pretrained_bsds.caffemodel'
    os.system(cmd)
