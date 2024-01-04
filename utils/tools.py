import numpy as np  
import torch
import torchvision
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tqdm

#parse model layer configuration
def parse_model_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] 

    module_defs = []
    type_name = None
    for line in lines:
        if line.startswith('['):
            type_name = line[1:-1].rstrip()
            if type_name == 'net':
                continue
            module_defs.append({})
            module_defs[-1]['type'] = type_name
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            if type_name == 'net':
                continue
            key, value = line.split('=')
            key = key.strip()
            module_defs[-1][key] = value.strip()
            
    return module_defs
            

#parse the yolov3 configuration
def parse_hyperparam_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    module_defs = []
    for line in lines:
        if line.startswith('['):
            type_name = line[1:-1].rstrip()
            if type_name != "net":
                continue
            module_defs.append({})
            module_defs[-1]['type'] = type_name
            # if module_defs[-1]['type'] == 'convolutional':
            #     print("conv")
            #     module_defs[-1]['batch_normalize'] = 0
        else:
            if type_name != 'net':
                continue
            key, value = line.split('=')
            key = key.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def get_hyperparam(data):
    for d in data:
        if d['type'] == 'net':
            batch = int(d['batch'])
            subdivisions = int(d['subdivisions'])
            in_width = int(d['width'])
            in_height = int(d['height'])
            in_channels = int(d['channels'])
            classes = int(d['class'])
            momentum = float(d['momentum'])
            decay = float(d['decay'])
            saturation = float(d['saturation'])
            ignore_clas = int(d['ignore_cls'])
            lr = float(d['learning_rate'])
            burn_in = int(d['burn_in'])
            max_batches = int(d['max_batches'])
            lr_policy = d['policy']

            return {'batch' : batch,
                    'subdivisions' : subdivisions,
                    'in_width' : in_width,
                    'in_height' : in_height,
                    'in_channels' : in_channels,
                    'classes' : classes,
                    'momentum' : momentum,
                    'decay' : decay,
                    'saturation' : saturation,
                    'ignore_clas' : ignore_clas,
                    'lr' : lr,
                    'burn_in' : burn_in,
                    'max_batch' : max_batches,
                    'lr_policy' : lr_policy}
        else:
            continue

def xywh2xyxy_np(x : np.array):
    y = np.zeros_like(x)
    y[...,0] = x[...,0] - x[...,2] / 2 #min_x
    y[...,1] = x[...,1] - x[...,3] / 2 #min_y
    y[...,2] = x[...,0] + x[...,2] / 2 #max_x
    y[...,3] = x[...,1] + x[...,3] / 2 #max_y
    return y

def drawBox(img):
    img = img * 255

    if img.shape[0] == 3:
        img_data = np.array(np.transpose(img, (1,2,0)), dtype=np.uint8)
        img_data = Image.fromarray(img_data)

    # draw = ImageDraw.Draw(img_data)
    plt.imshow(img_data)
    plt.show()

#box1, box2 IOU
def bbox_iou(box1, box2, xyxy=False, eps = 1e-9):
    box2 = box2.T

    if xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    else:
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    #intersection
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    #union
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    union = b1_w * b1_h + b2_w * b2_h - inter + eps

    iou = inter / union
    
    return iou

def boxes_iou(box1, box2, xyxy=False, eps = 1e-9):
    if xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    else:
        b1_x1, b1_y1 = box1[:,0] - box1[:,2] / 2, box1[:,1] - box1[:,3] / 2
        b1_x2, b1_y2 = box1[:,0] + box1[:,2] / 2, box1[:,1] + box1[:,3] / 2
        b2_x1, b2_y1 = box2[:,0] - box2[:,2] / 2, box2[:,1] - box2[:,3] / 2
        b2_x2, b2_y2 = box2[:,0] + box2[:,2] / 2, box2[:,1] + box2[:,3] / 2

    #intersection
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    #union
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    union = b1_w * b1_h + b2_w * b2_h - inter + eps

    iou = inter / union
    
    return iou

def cxcy2minmax(box):
    y = box.new(box.shape)
    xmin = box[...,0] - box[...,2] / 2
    ymin = box[...,1] - box[...,3] / 2
    xmax = box[...,0] + box[...,2] / 2
    ymax = box[...,1] + box[...,3] / 2

    y[...,0] = xmin
    y[...,1] = ymin
    y[...,2] = xmax
    y[...,3] = ymax

    return y
 
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def non_max_suppression(prediction, conf_thresh=0.25, iou_thresh=0.45):
    
    #num of class
    nc = prediction.shape[2] - 5 #13 - 5 = 8

    #setting
    max_wh = 4096 #width height
    max_det = 300 #detection num
    max_nms = 30000 #total nms num

    output = [torch.zeros((0,6), device='cpu')] * prediction.shape[0] #prediction.shape[0] : batch

    for xi, x in enumerate(prediction):
        x = x[x[...,4] > conf_thresh]

        if not x.shape[0]:
            continue

        x[:,5:] *= x[:, 4:5] #class *= objectness

        box = cxcy2minmax(x[:,:4])

        conf, j = x[:, 5:].max(1, keepdim=True) #highest value of class confidence -> conf value, index
        x = torch.cat((box, conf, j.float()), dim=1)[conf.view(-1) > conf_thresh]

        #number of box
        n = x.shape[0]
        
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:,4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh #class idx * max_wh, to differentiate by class 

        boxes, scores = x[:,:4] + c, x[:, 4]

        i = torchvision.ops.nms(boxes, scores, iou_thresh)
        
        if i.shape[0] > max_det:
            i = i[:max_det]
        
        output[xi] = x[i].detach().cpu()

    return output

def get_batch_statistics(predicts, targets, iou_threshold=0.5):
    batch_mertrics = []
    for p in range(len(predicts)):
        if predicts[p] is None:
            continue

        predict = predicts[p]
        pred_boxes = predict[:,:4]
        pred_scores = predict[:,4]
        pred_labels = predict[:,-1]

        true_positive = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:,0] == p][:, 1:]
        target_labels = annotations[:,0] if len(annotations) else []

        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                if len(detected_boxes) == len(annotations):
                    break

                if pred_label not in target_labels:
                    continue

                filtered_target_position, filtered_targets = zip(*filter(lambda x : target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                iou, box_filterd_index = boxes_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                box_index = filtered_target_position[box_filterd_index]

                if iou > iou_threshold and box_index not in detected_boxes:
                    true_positive[pred_i] = 1
                    detected_boxes += [box_index]
    
        batch_mertrics.append([true_positive, pred_scores, pred_labels])
    
    return batch_mertrics #shape : [batch, true_positive, pred_scores, pred_labels]


#compute the average precision, given the racaall and precision curves
def ap_per_class(tp, conf, pred_cls, target_cls):
    #sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    #find unique classes
    unique_classes = np.unique(target_cls) #remove duplicate classes

    #create precision-recall curve and compute AP for each class
    ap, p, r = [], [], [] #ap, precision, recall
    for c in tqdm.tqdm(unique_classes, desc='Compution AP'):
        i = pred_cls == c
        n_gt = (target_cls == c).sum() #num of gt objects
        n_p = i.sum() #num of prediction objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            #accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum() #Cumulative sum
            tpc = (tp[i]).cumsum()

            #recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            #precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            #AP
            ap.append(compute_ap(recall_curve, precision_curve))

    #cumpute F1 score
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    #compute the precision envelope
    for i in range(mpre.size -1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap