import requests
from pycocotools.coco import COCO
import urllib.request
import zipfile


ANNOTATIONS_PATH = '../'
ANNOTATIONS_LINK = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
DATASET_PATH = '../train/'

# Download annotations file
print('Downloading annotations')
urllib.request.urlretrieve(ANNOTATIONS_LINK, filename=ANNOTATIONS_PATH +
                   'annotations_trainval2017.zip')

# Unzip
print('Unzip annotations file')
with zipfile.ZipFile(ANNOTATIONS_PATH + 'annotations/trainval2017.zip', 'r') as zip_ref:
    zip_ref.extractall(ANNOTATIONS_PATH)

# Instantiate COCO specifying the annotations json path
coco = COCO(ANNOTATIONS_PATH + 'annotations/instances_train2017.json')
# Specify a list of category names of interest
catIds = coco.getCatIds(catNms=['person'])
# Get the corresponding image ids and images using loadImgs
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)

# Save the images into a local folder
for im in images:
    print('Downloading ' + im['file_name'])
    img_data = requests.get(im['coco_url']).content
    with open(DATASET_PATH + im['file_name'], 'wb') as handler:
        handler.write(img_data)
