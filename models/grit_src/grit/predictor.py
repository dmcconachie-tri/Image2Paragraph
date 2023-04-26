# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Jialian Wu from https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/visualizer.py
import torch

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


class Visualizer_GRiT(Visualizer):
    def __init__(self, image, instance_mode=None):
        super().__init__(image, instance_mode=instance_mode)

    def draw_instance_predictions(self, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        object_description = predictions.pred_object_descriptions.data
        # uncomment to output scores in visualized images
        # object_description = [c + '|' + str(round(s.item(), 1)) for c, s in zip(object_description, scores)]

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=None,
            boxes=boxes,
            labels=object_description,
            keypoints=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer_GRiT(image, instance_mode=self.instance_mode)
        instances = predictions["instances"].to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output
