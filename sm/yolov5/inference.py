from base import Base

import numpy as np
import sys
import cv2
from .boundingbox import BoundingBox


import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from .processing import preprocess, postprocess, non_max_suppression
from .render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS, plot_one_box
from .labels import COCOLabels


class Yolov5(Base):
    
    def inference(self, img):
        INPUT_SHAPE = (640, 640)

        IMAGE_WIDTH=640
        IMAGE_HEIGHT=640
        TRITON_IP = "localhost"
        TRITON_PORT = 8001
        MODEL_NAME = "yolov5s"
        INPUTS = []
        OUTPUTS = []
        INPUT_LAYER_NAME = "images"
        OUTPUT_LAYER_NAME = "output"
        CONFIDENCE=0.5
        NMS=0.45


        triton_client = grpcclient.InferenceServerClient(
            url=f"{TRITON_IP}:{TRITON_PORT}",
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)


        INPUTS.append(grpcclient.InferInput(INPUT_LAYER_NAME, [1, 3, IMAGE_WIDTH, IMAGE_HEIGHT], "FP32"))
        OUTPUTS.append(grpcclient.InferRequestedOutput(OUTPUT_LAYER_NAME))
        TRITON_CLIENT = grpcclient.InferenceServerClient(url=f"{TRITON_IP}:{TRITON_PORT}")


        input_image_buffer = preprocess(img, [IMAGE_WIDTH, IMAGE_HEIGHT])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        INPUTS[0].set_data_from_numpy(input_image_buffer)

        results = triton_client.infer(model_name=MODEL_NAME,
                                    inputs=INPUTS,
                                    outputs=OUTPUTS,
                                    client_timeout=30)

        result = results.as_numpy(OUTPUT_LAYER_NAME)
        print(f"Received result buffer of size {result.shape}")
        print(f"Naive buffer sum: {np.sum(result)}")
        pred = non_max_suppression(result, CONFIDENCE, NMS, None, False, max_det=1000)

        boxes=pred[0].numpy()
        print(len(boxes))
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5].astype(np.int32) if len(boxes) else np.array([])

        detected_objects = []
        for box, score, label in zip(result_boxes, result_scores, result_classid):
            #detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], img.shape[1], img.shape[0]))
            detected_objects.append([label, score, box[0], box[2], box[1], box[3], img.shape[1], img.shape[0]])


        return detected_objects
