import requests
from pycocotools.coco import COCO
import urllib.request
import zipfile
import os
from tqdm import tqdm


# Absolute path to the module directory
package_dir = os.path.dirname(os.path.abspath(__file__))

ANNOTATIONS_PATH = os.path.join(package_dir, '..')
ANNOTATIONS_LINK = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
DATASET_PATH = os.path.join(package_dir, '..', 'train')

# Download annotations file
print('Downloading annotations')
urllib.request.urlretrieve(ANNOTATIONS_LINK, filename=os.path.join(ANNOTATIONS_PATH,
                           'annotations_trainval2017.zip'))

# Unzip
print('Unzip annotations file')
with zipfile.ZipFile(os.path.join(ANNOTATIONS_PATH, 'annotations_trainval2017.zip'), 'r') as zip_ref:
    zip_ref.extractall(ANNOTATIONS_PATH)

# Instantiate COCO specifying the annotations json path
coco = COCO(os.path.join(ANNOTATIONS_PATH,
            'annotations', 'instances_train2017.json'))
# Specify a list of category names of interest
catIds = coco.getCatIds(catNms=['person'])
# Get the corresponding image ids and images using loadImgs
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)

# Save the images into a local folder
for im in tqdm(images, desc='Downloading images'):
    img_data = requests.get(im['coco_url']).content
    with open(os.path.join(DATASET_PATH, im['file_name']), 'wb') as handler:
        handler.write(img_data)
