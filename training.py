import os
import sys
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from src.utils.register_data import register

setup_logger()

class Train:

    def __init__(self, image_dir, coco_file, model_name):
    
        self.image_dir = image_dir
        self.coco_file = coco_file
        self.model_name = model_name
        self.model_cfg = get_cfg()
        self.register_obj = register(self.image_dir,  self.coco_file)

    def trainmodel(self):

        try:

            metadata = self.register_obj.load_metadata()
            self.model_cfg.merge_from_file("config.yaml")
            os.makedirs(self.model_cfg.OUTPUT_DIR, exist_ok=True)
            trainer = DefaultTrainer(self.model_cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()

        except Exception as e:
            print(e)
            sys.exit(1)


if __name__ == '__main__':

    Trainer = Train('images', 'coco/output.json', 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
    Trainer.trainmodel()

