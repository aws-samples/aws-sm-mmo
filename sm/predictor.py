
from __future__ import print_function

import os
import json
import cv2
import numpy as np
import torch
import io
import flask
import base64
from PIL import Image
from resnet.inference import Resnet
from yolov5.inference import Yolov5
from bert_base.inference import Bertbase

from json import JSONEncoder

prefix = '/opt/ml/'

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return JSONEncoder.default(self, obj)


# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = True #MyService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = json.loads(flask.request.data)
    modelname=data.get("modelname",None)

    if (modelname=='resnet') or (modelname=="yolov5"):
        img_str=data.get("payload").get("img",None)
        jpg_original = base64.b64decode(img_str)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)
    elif (modelname=="bert_base"):
        text=data.get("payload").get("text",None)

    result=None
    if (modelname=='resnet'):
        instance = Resnet()
        result=instance.inference(img)
    elif (modelname=="yolov5"):
        instance = Yolov5()
        result=instance.inference(img)
    elif (modelname=="bert_base"):
        instance = Bertbase()
        result=instance.inference(text)
    print(result)
        
    rJson = { 'modelname':modelname,'result': result }
    return flask.Response(response=json.dumps(rJson, cls=NumpyArrayEncoder), status=200, mimetype='application/json')

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)
