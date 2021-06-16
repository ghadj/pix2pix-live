import os
import cv2 as cv

import edge_detector as ed

# Absolute path to the module directory
package_dir = os.path.dirname(os.path.abspath(__file__))

SRC_PATH = os.path.join(package_dir, '..', 'Dataset', 'train')
DEST_PATH =  os.path.join(package_dir, '..', 'Dataset', 'test')

img_extentions = ['.jpg', '.png']

detector = ed.EdgeDetector()

files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(SRC_PATH)) for f in fn if f.endswith(tuple(img_extentions))]

for f in files:
    path, filename = os.path.split(f)
    f_hed = DEST_PATH + filename[:-4] + '_hed' + filename[-4:]
    print(f + ' => ' + f_hed)
    input = cv.imread(f)
    output = detector.run(input)
    cv.imwrite(f_hed, output)
