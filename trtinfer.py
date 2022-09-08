import logging, tracemalloc

verbose_level = logging.DEBUG
logging.basicConfig(format= '[%(levelname)s]: %(message)s', level=verbose_level)
if verbose_level == logging.DEBUG:
    tracemalloc.start()
    logging.debug("Started memory tracing")

import torch
import cv2
import Label_xtr
import opencv_latency as ocv
import random
import time
import numpy as np
import tensorrt as trt
from collections import OrderedDict, namedtuple
from numpy import mean
import socket



device = torch.device('cuda:0')
sox = False
film = False
text_color = (255,255,0)
box_color = (0, 255, 0)
src = 0  # 0 for webcam, 1 for ext-webcam, "testvid.mp4" for trash vid, link for ip cam
addr = ("localhost", 5000)
warmup = True
warmup_time = 3
warmup_rounds = 1000
warmup_res = [640, 480]


def load(weights, device=torch.device('cuda:0')):
    # load model
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.VERBOSE)
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

    return nums, boxes, scores, classes


def draw_bbox_og(img, boxes, scores, classes, color, ratio, dwdh, names):
    for box, score, cl in zip(boxes, scores, classes):
        box = postprocess(box, ratio, dwdh).round().int()
        name = names[cl]
        name += ' ' + str(round(float(score), 3))
        cv2.rectangle(img, box[:2].tolist(), box[2:].tolist(), color, 2)
        cv2.putText(img, name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)
        return box


def filter_mcd(boxes, scores, min_confidence, ratio, dwdh):
    boxes = postprocess(boxes, ratio, dwdh)
    mcd = scores.tolist().index(max(scores))
    if scores[mcd] > min_confidence:
        return [int(x) for x in boxes[mcd]], scores[mcd], mcd
    else:
        return None, None, None


def pos_servo(det_center, frame, client, px=25, send=True):
    ccy, ccx = [x/2 for x in frame.shape[:2]]

    if ccx < det_center[0] - px:
        xServ = '-'
    elif ccx > det_center[0] + px:
        xServ = '+'
    else:
        xServ = "="

    if ccy < det_center[1] - px:
        yServ = '-'
    elif ccy > det_center[1] + px:
        yServ = '+'
    else:
        yServ = "="

    msg = f'{xServ} {yServ}'
    if send:
        logging.info(f'Sending: {msg}')
        client.send(f'{xServ} {yServ}'.encode("utf-8"))
        return client.recv(2048).decode("utf-8")
    else:
        logging.debug(msg)


def frame_debug(frame, px=25):
    cy, cx = frame.shape[:2]
    ccx = cx//2
    ccy = cy//2
    cv2.rectangle(frame, (ccx-px, ccy-px), (ccx+px, ccy+px), (255, 0, 0), 2, cv2.LINE_AA)

    cv2.line(frame, (0, ccy), (ccx-px, ccy), (0, 0, 255), 2, cv2.LINE_AA)  # left line
    cv2.line(frame, (ccx, 0), (ccx, ccy-px), (0, 0, 255), 2, cv2.LINE_AA)  # upper line
    cv2.line(frame, (ccx+px, ccy), (cx, ccy), (0, 0, 255), 2, cv2.LINE_AA)  # right line
    cv2.line(frame, (ccx, ccy+px), (ccx, cy), (0, 0, 255), 2, cv2.LINE_AA)  # lower line


framerate = [120]
if sox:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    pcld = []
    s.connect(addr)
    print('Connected to', addr)
else:
    s = ""

weights = './models/yolov7-tiny-nms.trt'
device = torch.device('cuda:0')
binding_addrs, bindings, context = load(weights, device)
vid = ocv.WebcamVideoStream(src=src).start()
#vid = cv2.VideoCapture("testvid.mp4")
names = Label_xtr.coco
logging.debug(f'Loaded {len(names)} names')
logging.debug(names)
w, h = vid.get()
logging.debug(f'Video size: {w}x{h}')
if verbose_level == logging.DEBUG or film == True:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Soji-TRT-Out.avi', fourcc, 30.0, (int(w), int(h)))

logging.debug("Memory Allocated: " + str(torch.cuda.memory_allocated(device)))
logging.debug("Memory Cached: " + str(torch.cuda.memory_cached(device)))
logging.debug("Memory Reserved: " + str(torch.cuda.memory_reserved(device)))
logging.debug(f"Max Memory Used Before Main Loop: {tracemalloc.get_traced_memory()[1]}")
logging.debug(f"Current Memory Usage: {tracemalloc.get_traced_memory()[0]}")

if warmup:
    print("Warming up model. Please wait...")
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_time:
        test_data = torch.randn(1, 3, warmup_res[0], warmup_res[1]).to(device)
        infer(test_data, context, bindings, binding_addrs)
        if verbose_level == logging.DEBUG:
            cv2.imshow("Warmup Data", test_data.cpu().numpy()[0].transpose(1, 2, 0))
            cv2.waitKey(1)
    cv2.destroyAllWindows()

    print('Model Warmup complete.')
    print("Warming up camera. Starting in 3 seconds...")
    for i in range(warmup_time*100):
        vid.read()
        time.sleep(0.01)


while True:
    try:
        # img = cv2.imread('/home/gg-dev/yolov7/inference/images/horses.jpg')
        start = time.time()
        img = vid.read().copy()

        image, ratio, dwdh = prep_image(img, 640)
        dets, bboxes, conf, classes = infer(image, context, bindings, binding_addrs)

        fps = 1/(time.time() - start)
        framerate.append(fps)

        # bbox = draw_bbox_og(img, bboxes, conf, classes, (0,255,0), ratio, dwdh, names)
        bbox, best_conf, name_index = filter_mcd(bboxes, conf, 0.5, ratio, dwdh) if len(bboxes) > 0 else (None, None, None)

        if bbox is not None:
            logging.debug((bbox, best_conf, names[classes[name_index]]))
            cv2.rectangle(img, bbox[:2], bbox[2:],  box_color, 2)
            cv2.putText(img, f"{names[classes[name_index]]}: {best_conf*100}", bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, thickness=2)
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

            try:
                if sox:
                    pcld.append(pos_servo((cx, cy), img, s))

            except Exception as ex:
                logging.info(ex)
                logging.info('Disconnecting from Client')
                s.send(b'disconnect')
                s.close()
                continue

            if verbose_level == logging.DEBUG:
                frame_debug(img)
                cv2.putText(img, f"FPS: {round(fps)}", (int(w - 200), int(h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, thickness=2)
                pos_servo((cx, cy), img, s, send=False)

        cv2.imshow("Output", img)
        logging.debug(f"Memory Used: {tracemalloc.get_traced_memory()[0]}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise RuntimeError("Program terminated by Key Press")

    except Exception as e:
        logging.error(e)
        break

    except KeyboardInterrupt as k:
        logging.error("Keyboard Interrupt")
        break

cv2.destroyAllWindows()
#vid.release()
vid.stop()
print(f'Average FPS: ', mean(framerate))
logging.debug(f"Peak Memory Used: {tracemalloc.get_traced_memory()[1]}")



