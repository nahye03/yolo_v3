import os, sys
import torch
import torch.nn as nn

from utils.tools import *

class Yololoss(nn.Module):
    def __init__(self, device, num_class):
        super(Yololoss, self).__init__()
        self.device = device
        self.num_class = num_class
        self.mseloss = nn.MSELoss().to(device) #mean squarted error
        self.bceloss = nn.BCELoss().to(device) #binary cross entropy
        self.bcelogloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device = device)).to(device)


    def compute_loss(self, pred, targets, yololayer):
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        
        # pout shape : [batch, anchors, grid x, grid y, 13(box + n_class 8)]
        # the number of boxes in each yolo layer : anchors(3) * grid_h * grid_w
        # yolo 0 -> 3 * 19 * 19, yolo1 -> 3 * 38 * 38, yolo2 -> 3 * 76 * 76
        # total boxex : 22743

        # positive predition vs negative predictions
        # pos : neg = 0.01 : 0.99 
        # Only in positive prediction, we can get box_loss and class_loss
        # in negative prediction, only obj_loss
        
        #get positive targets
        tcls, tbox, tindices, tanchors = self.get_targets(pred, targets, yololayer)

        #3 yolo layers
        for pidx, pout in enumerate(pred):
            # print("yolo {}, shape {}".format(pidx, pout.shape))
            batch_id, anchor_id, gy, gx = tindices[pidx]

            tobj = torch.zeros_like(pout[...,0], device=self.device)

            num_targets = batch_id.shape[0]

            if num_targets:
                #pout shape : [batch, anchor, grid_h, grid_w, 13 box_attrib]
                #ps shape : [object??, 13]
                ps = pout[batch_id, anchor_id, gy, gx] 

                pxy = torch.sigmoid(ps[...,0:2])
                pwh = torch.exp(ps[...,2:4]) * tanchors[pidx]
                pbox = torch.cat((pxy, pwh), dim=1)

                iou = bbox_iou(pbox.T, tbox[pidx], xyxy=False)

                #box loss
                #MSE (Mean Squared Error) tobj
                # loss_wh = self.mseloss(pbox[..., 2:4], tbox[pidx][..., 2:4])
                # loss_xy = self.mseloss(pbox[..., 0:2], tbox[pidx][..., 0:2])
                lbox += (1 - iou).mean()

                #objectness loss
                #gt box and predicted box -> positive : 1 / negative : 0 using IOU
                tobj[batch_id, anchor_id, gy, gx] = iou.detach().clamp(0).type(tobj.dtype)

                #class loss
                if ps.size(1) - 5 > 1: #ps.size(1) [13]
                    t = torch.zeros_like(ps[...,5:], device=self.device)
                    t[range(num_targets), tcls[pidx]] = 1
                    
                    lcls += self.bcelogloss(ps[:,5:], t)
                
            lobj += self.bcelogloss(pout[...,4], tobj)
        
        #loss weight
        lcls *= 0.05
        lobj *= 1.0
        lbox *= 0.5
        
        #total loss
        loss = lcls + lbox + lobj
        loss_list = [loss.item(), lobj.item(), lcls.item(), lbox.item()]
        
        return loss, loss_list

    def get_targets(self, preds, targets, yololayer):

        num_anchor = 3
        num_targets = targets.shape[0] #7, targets (batch_id, cls, cx, cy, w, h)
        tcls, tboxes, indices, anch = [], [], [], []

        gain = torch.ones(7, device = self.device)

        #anchor index
        ai = torch.arange(num_anchor, device=targets.device).float().view(num_anchor, 1).repeat(1, num_targets)

        #targets.shape [3(anchor), object, 7]
        # 7 : [batch_id, class_id, box_cx, box_cy, box_w, box_h, anchor_id]
        targets = torch.cat((targets.repeat(num_anchor, 1, 1), ai[:,:,None]), dim=2) 

        
        for yi, yl in enumerate(yololayer):
            anchors = yl.anchor / yl.stride
            gain[2:6] = torch.tensor(preds[yi].shape)[[3,2,3,2]] #grid_w, grid_h #ex) gain [1, 1, 19, 19, 19, 19, 1]

            t = targets * gain #normalized cx, cy, w, h to fit grid size

            if num_targets:
                r = t[:, :, 4:6] / anchors[:, None] #targets's w,h is divided by anchors's w, h

                #select threshold ratios less than 4 -> true or false
                j = torch.max(r, 1./r).max(dim=2)[0] < 4
                t = t[j] # t shape [box_num, 7]
            else:   
                t = targets[0]     

            #batch_id, class_id
            b, c = t[:, :2].long().T

            gxy = t[:, 2:4] #target's x,y, shape: [object, 2]
            gwh = t[:, 4:6] #target's w,h

            gij = gxy.long() #box index x, y (it means cx, cy)
            gi, gj = gij.T

            #anchor index
            a = t[:, 6].long()

            #add indices ((batch_id, anchor_id, target data grid index y, index x))
            indices.append((b, a, gj.clamp_(0, gain[3].long()-1), gi.clamp_(0, gain[2].long()-1)))

            #add target_box 
            tboxes.append(torch.cat((gxy-gij, gwh), dim=1)) # torch.cat((gxy-gij, gwh), dim=1) shape : (object, 4)
            
            #add anchor
            anch.append(anchors[a])

            #add class
            tcls.append(c)
        
        return tcls, tboxes, indices, anch







