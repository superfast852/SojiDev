def getLabelsFromTxt(path="coco-lbl.txt", verbose=True):
    with open(path, "r") as lbls:  # Open txt file
        a = lbls.read()  # Read txt file
        b = a.split('\n')  # Split by every line into list
        if verbose: print("Labels Extracted: ", b)  # print extracted list
        return b

def getLabelsFromYaml(path="data.yaml", verbose=True):  # generally the same but with a yaml file
    import yaml
    with open(path, 'r') as file:
        a = yaml.full_load(file)["names"]
    if verbose: print(a)
    return a

coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

if __name__=="__main__":
    getLabelsFromTxt()