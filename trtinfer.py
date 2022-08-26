import sys
import torch
import cv2
import opencv_latency as ocv
import random
import time
import numpy as np
import tensorrt as trt
from pathlib import Path
from collections import OrderedDict, namedtuple

device = torch.device('cuda:0')

def load(weights, device=torch.device('cuda:0')):
    # load model
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()
    for _ in range(10):
        tmp = torch.randn(1, 3, 640, 640).to(device)
        binding_addrs['images'] = int(tmp.data_ptr())
        context.execute_v2(list(binding_addrs.values()))

    return binding_addrs, bindings, context


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def postprocess(boxes, r, dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= r
    return boxes


def prep_image(img, model_res):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    im = torch.from_numpy(im).to(device)
    return im/255, ratio, dwdh

def infer(image, context, bindings, binding_addrs):
    binding_addrs['images'] = int(image.data_ptr())
    context.execute_v2(list(binding_addrs.values()))
    nums = bindings['num_dets'].data
    boxes = bindings['det_boxes'].data
    scores = bindings['det_scores'].data
    classes = bindings['det_classes'].data

    boxes = boxes[0, :nums[0][0]]
    scores = scores[0, :nums[0][0]]
    classes = classes[0, :nums[0][0]]

    return nums.tolist(), boxes.tolist(), scores.tolist(), classes.tolist()


weights = './models/yolov7-tiny-nms.trt'
device = torch.device('cuda:0')
binding_addrs, bindings, context = load(weights, device)
#vid = ocv.WebcamVideoStream(src=0).start()
vid = cv2.VideoCapture(0)
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
framerate = []
while True:

    #img = cv2.imread('/home/gg-dev/yolov7/inference/images/horses.jpg')
    start = time.time()
    img = vid.read()
    image, ratio, dwdh = prep_image(img, 640)
    dets, bboxes, conf, classes = infer(image, context, bindings, binding_addrs)

    for i in range(dets[0][0]):
        cv2.rectangle(img, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])), (0,255,0), 2)
        cv2.putText(img, f"{names[classes[i]]}: {conf[i]*100}", (int(bboxes[i][0]), int(bboxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), thickness=2)

    cv2.imshow("Output", img)
    run_time = time.time()-start
    fps = 1/run_time
    framerate.append(fps)
    print(f'''Results:
    Number of detections: {dets[0][0]}
    Classes: {classes}
    Confidences: {conf}\n
    Coordinates: {bboxes}\n
    Total runtime: {run_time}
    Framerate: {fps}\n''')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Program ended with framerate: {sum(framerate)/len(framerate)}")
        break
cv2.destroyAllWindows()
vid.release()



