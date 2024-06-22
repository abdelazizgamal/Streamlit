import streamlit as st

import subprocess
import sys

def install():
    subprocess.call([sys.executable, '-m', 'pip', 'install', "pyyaml==5.1"])
    # subprocess.call([sys.executable, '-m', 'pip', 'install', gitpython])
    # from git import Repo  # pip install gitpython

    # Repo.clone_from('https://github.com/facebookresearch/detectron2', repo_dir)

    subprocess.call([sys.executable, '-m', 'pip', 'install', "git+https://github.com/facebookresearch/detectron2.git"])

# # pip install pyyaml==5.1
# import sys, os, distutils.core
# # Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# # See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
# !git clone 'https://github.com/facebookresearch/detectron2'
# dist = distutils.core.run_setup("./detectron2/setup.py")
# !python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
# sys.path.insert(0, os.path.abspath('./detectron2'))

install()
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import detectron2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# Properly install detectron2. (Please do not install twice in both ways)
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

@st.cache(allow_output_mutation=True)

def install():
    subprocess.call([sys.executable, '-m', 'pip', 'install', "pyyaml==5.1"])
    # subprocess.call([sys.executable, '-m', 'pip', 'install', gitpython])
    # from git import Repo  # pip install gitpython

    # Repo.clone_from('https://github.com/facebookresearch/detectron2', repo_dir)

    subprocess.call([sys.executable, '-m', 'pip', 'install', "git+https://github.com/facebookresearch/detectron2.git"])

# # pip install pyyaml==5.1
# import sys, os, distutils.core
# # Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# # See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
# !git clone 'https://github.com/facebookresearch/detectron2'
# dist = distutils.core.run_setup("./detectron2/setup.py")
# !python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
# sys.path.insert(0, os.path.abspath('./detectron2'))
with st.spinner('Model is being loaded..'):
  install()
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow


import detectron2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# def load_model():
#   model=tf.keras.models.load_model('/content/my_model2.hdf5')
#   return model

#   model=load_model()

st.write("""
         # Image_Analysis
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png","jpeg"])

from PIL import Image, ImageOps
st.set_option('deprecation.showfileUploaderEncoding', False)
cfg = get_cfg()
def import_and_predict(image_data):

      cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
      cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
      # Find a model from detectron2's model zoo.  https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
      cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
      predictor = DefaultPredictor(cfg)
      outputs = predictor(image_data)


      return outputs
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, use_column_width=True)

    with st.spinner('analyzing image..'):
      predictions = import_and_predict(opencvImage)
    instances = predictions["instances"]

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(opencvImage[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
    out = v.draw_instance_predictions(instances.to("cpu"))

    out_img = Image.fromarray(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

    st.image(out_img, use_column_width=True)

    detected_class_indexes = instances.pred_classes.tolist()

    # Retrieve class names from metadata
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    class_catalog = metadata.thing_classes
    pred_class_names = [class_catalog[idx] for idx in detected_class_indexes]

    # insert the list to the set
    list_set = set(pred_class_names)
    # convert the set to the list
    unique_class_names = (list(list_set))

    s = ''
    for i in unique_class_names:
        s += "- " + i + "\n"

    st.markdown(s)
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
)

