#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyyaml==5.1')
# workaround: install old version of pytorch since detectron2 hasn't released packages for pytorch 1.9 (issue: https://github.com/facebookresearch/detectron2/issues/3158)
get_ipython().system('pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html')

# install detectron2 that matches pytorch 1.8
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
get_ipython().system('pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html')
# exit(0)  # After installation, you need to "restart runtime" in Colab. This line can also restart runtime


# In[2]:


# cd drive/MyDrive/roofing/


# In[3]:


# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.8")   # please manually install torch 1.8 if Colab changes its default version


# In[4]:


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# In[5]:


get_ipython().system('git clone https://github.com/facebookresearch/detectron2.git')


# In[6]:


import os
import numpy as np
import json
from detectron2.structures import BoxMode
def get_microcontroller_dicts(directory):
    classes = ['roof', 'window']
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        record["height"] = 600
        record["width"] = 800
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "test"]:
    DatasetCatalog.register("roof12_" + d, lambda d=d: get_microcontroller_dicts('../input/roofing/roof/' + d))
    MetadataCatalog.get("roof12_" + d).set(thing_classes=['roof', 'window'])
microcontroller_metadata = MetadataCatalog.get("roof12_train")


# In[7]:


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import detectron2.data.detection_utils as a
def my_check_image_size(dataset_dict, image):
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]
a.check_image_size = my_check_image_size

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("roof12_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "../input/roofing-model/output/model_final.pth"

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=True)
# trainer.train()
get_ipython().system('cp ../input/roofing-model/output . -r')


# In[8]:


microcontroller_metadata = MetadataCatalog.get("roof12_train")


# In[9]:


pwd


# In[10]:


f = open('config.yml', 'w')
f.write(cfg.dump())
f.close()


# In[11]:


# downloand texture from drive
get_ipython().system('pip install gdown')
get_ipython().system('gdown --id 1RwevZk2oqUNOd4RCsioWKdV1Bxonx91R')


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import pandas as pd
from detectron2.utils.visualizer import ColorMode
import detectron2.utils.visualizer as a
import os

def my_create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            print("class_name", class_names)
            print("classes", classes)
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels
  
a._create_text_labels = my_create_text_labels

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = ("roof12_test", )
predictor = DefaultPredictor(cfg)


 
original_im = cv2.imread("../input/roofing/roof/roof/test/img11.jpg") 
im = original_im.copy()

H,W,_ = im.shape

outputs = predictor(im)
#print(outputs)
temp = outputs['instances'].get('pred_masks')



v = Visualizer(im[:, :, ::-1],
                metadata=microcontroller_metadata, 
                scale=0.8, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.figure(figsize = (14, 10))
plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
plt.show()


# In[21]:



# texture = cv2.imread("./1637541__140.jpg")
# plt.imshow(texture)
# plt.show()
# texture.shape
from PIL import Image

# Opens an image
bg = Image.open("./1637541__140.jpg")

# The width and height of the background tile
bg_w, bg_h = bg.size

# Creates a new empty image, RGB mode, and size 1000 by 1000
new_im = Image.new('RGB', (1000,1000))

# The width and height of the new image
w, h = new_im.size

# Iterate through a grid, to place the background tile
for i in range(0, w, bg_w):
    for j in range(0, h, bg_h):
        # Change brightness of the images, just to emphasise they are unique copies
        bg = Image.eval(bg, lambda x: x+(i+j)/1000)

        #paste the image at location i, j:
        new_im.paste(bg, (i, j))


# resize image
resized = cv2.resize(np.array(new_im), (W,H), interpolation = cv2.INTER_AREA)
texture =resized
plt.imshow(texture)
plt.show()


# In[22]:


temp = outputs['instances'].get('pred_masks').detach().cpu().numpy()


# In[23]:


masks = [temp[i].astype('uint8') * 1 for i in range(len(temp))]
mask = masks[0]
for i in range(1,len(masks)):
    mask+=masks[i]*(1-mask)

plt.imshow(mask)
plt.show()


# In[24]:


out = original_im*(1-mask.reshape((H,W,1))==1)+(mask.reshape((H,W,1))==1)*texture
oo = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
plt.imshow(oo)
plt.show()


# In[25]:


im = Image.fromarray(oo)
im


# In[59]:


def run(image_path):
    im = cv2.imread(image_path)
    H,W,_ = im.shape
#     print(H,W)
    outputs = predictor(im)
    temp = outputs['instances'].get('pred_masks').detach().cpu().numpy()
    masks = [temp[i].astype('uint8') * 1 for i in range(len(temp))]
    if len(masks)==0:
        print("No mask has been detected")
        return Image.fromarray(im)
    mask = masks[0]
    for i in range(1,len(masks)):
        mask+=masks[i]*(1-mask)
#     print(mask.shape)
    resized = cv2.resize(np.array(new_im), (W,H), interpolation = cv2.INTER_AREA)
    texture =resized
    out = im*(1-mask.reshape((H,W,1))==1)+(mask.reshape((H,W,1))==1)*texture
    oo = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(oo)
    return im


# In[37]:


run("../input/roofing/img49.jpg")


# In[39]:


run("../input/roofing/img50.jpg")


# In[40]:


run("../input/roofing/img54.jpg")


# In[41]:


run("../input/roofing/img55.jpg")


# In[42]:


run("../input/roofing/img56.jpg")


# In[43]:


run("../input/roofing/img57.jpg")


# In[44]:


run("../input/roofing/img58.jpg")


# In[46]:


run("../input/roofing/img60.jpg")


# In[47]:


run("../input/roofing/img58.jpg")


# In[48]:


run("../input/roofing/img60.jpg")


# In[49]:


run("../input/roofing/img63.jpg")


# In[53]:


run("../input/roofing/img70.jpg")


# In[60]:


run("../input/roofing/img71.jpg")

