from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer  import Visualizer
from detectron2.data.datasets import register_coco_instances
import cv2 as cv
import os
from src.utils.utils import encodeImageIntoBase64
from src.utils.register_data import register

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Predict:

    def __init__(self, image_file, image_dir, coco_file):
    
        self.image_file = image_file
        self.model_cfg = get_cfg()
        self.image_dir = image_dir
        self.coco_file = coco_file
        self.register_obj = register(self.image_dir,  self.coco_file)


    def get_prediction(self):

        register_coco_instances("sample", {}, self.coco_file, self.image_dir)
        sample_metadata = MetadataCatalog.get("sample")
        dataset_dicts = DatasetCatalog.get("sample")

        self.model_cfg.merge_from_file('config.yaml')
        predictor = DefaultPredictor(self.model_cfg)
        im = cv.imread(self.image_file)
        outputs = predictor(im)

        ## visualizing and saving graph
        v = Visualizer(im[:, :, ::-1], metadata = sample_metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        predicted_image = v.get_image()
        im_rgb = cv.cvtColor(predicted_image, cv.COLOR_RGB2BGR)
        cv.imshow('image1', im_rgb)
        cv.imwrite(f'{self.image_file}_OD.jpg', im_rgb)


if __name__ == '__main__':

    Predictor = Predict('2021_3$largeimg_426024703.jpg', 'images', 'coco/output.json')
    Predictor.get_prediction()



