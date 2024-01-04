import os, sys
import torch
import torch.optim as optim
from prettytable import PrettyTable

from utils.tools import *
from train.loss import *

class Trainer:
    def __init__(self, model, train_loader, eval_loader, hparam, device, torch_writer):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hparam['max_batch']
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.yololoss = Yololoss(self.device, self.model.n_classes)
        self.optimizer = optim.SGD(model.parameters(), lr=hparam['lr'], momentum=hparam['momentum'])
        
        self.scheduler_multistep = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                             milestones=[20,40,60],
                                                             gamma=0.5)
        self.torch_writer = torch_writer

    def run_iter(self):
        for i, batch in enumerate(self.train_loader):
            #drop the batch when invalid values
            if batch is None:
                continue
            input_img, targets, anno_path = batch #input_img.shape [batch, channel, height, width], target.shape [object, 6(batch_idx, class, box coordinate 4)]
            
            # print("input : {} {}".format(input_img.shape, targets.shape))

            input_img = input_img.to(self.device, non_blocking = True)

            output = self.model(input_img) #yolo_result -> 3 feature map, output[0].shape [batch, anchor, grid, grid, 13(bbox attribute 5 + n_class 8)]
            # print("output  - len : {}, shape : {}".format(len(output), output[0].shape))

            #get loss between output and target
            loss, loss_list = self.yololoss.compute_loss(output, targets, self.model.yolo_layers)

            #get gradient
            loss.backward()
            self.optimizer.step() #update weight using gradient
            self.optimizer.zero_grad() #initialize w, b
            self.scheduler_multistep.step(self.iter)
            self.iter += 1

            loss_name = ['total_loss', 'obj_loss','cls_loss', 'box_loss']

            if i % 10 == 0 :
                print("epoch {} / iter {} lr {} loss {}".format(self.epoch, self.iter, get_lr(self.optimizer), loss.item()))
                self.torch_writer.add_scalar('lr', get_lr(self.optimizer), self.iter)
                self.torch_writer.add_scalar('total_loss', loss, self.iter)
                for ln, lv in zip(loss_name, loss_list):
                    self.torch_writer.add_scalar(ln, lv, self.iter)

        return loss

    #evaluation
    def run_eval(self):
        predict_all = []
        gt_labels = []
        for i, batch in enumerate(self.eval_loader):
            if i == 2:
                break
            if batch is None:
                continue
            input_img, targets, anno_path = batch
            input_img = input_img.to(self.device, non_blocking = True)
            
            with torch.no_grad():
                output = self.model(input_img)
                #len(best_box_list) : batch
                #best_box_list[0].shape : [num_of_box, 6], 6: xmin, ymin, xmax, ymax, max_class_conf, max_class_idx
                best_box_list = non_max_suppression(output, conf_thresh=0.0, iou_thresh=0.45)

                print("eval output shape : ", output.shape, " best_box_list : ", len(best_box_list), best_box_list[0].shape)

            #target.shape [object, 6(batch_idx, class, box coordinate 4)]
            gt_labels += targets[...,1].tolist()
            targets[...,2:6] = cxcy2minmax(targets[...,2:6])
            input_wh = torch.tensor([input_img.shape[3], input_img.shape[2], input_img.shape[3], input_img.shape[2]]) #whwh
            targets[...,2:6] *= input_wh #un_normalization

            predict_all += get_batch_statistics(best_box_list, targets, iou_threshold=0.5)
        
        true_positive, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*predict_all))]

        #get mAP, recalls
        metric_output = ap_per_class(true_positive, pred_scores, pred_labels, gt_labels)

        if metric_output is not None:
            precision, recall, ap, f1, ap_class = metric_output
            ap_table = PrettyTable()
            ap_table.field_names = ['index', 'ap']
            for i, c in enumerate(ap_class):
                ap_table.add_row([c, "%.5f"%ap[i]])
            
            print(ap_table)
        return


    def run(self):
        while True:
            if self.max_batch <= self.iter:
                break

            self.model.train()
            #loss calculation
            loss = self.run_iter()
            self.epoch += 1

            #evaluation
            self.model.eval()
            self.run_eval()

            #save model (checkpoint)
            checkpoint_path = os.path.join("./output", "model_epoch"+str(self.epoch)+".pth")
            torch.save({'epoch' : self.epoch,
                        'iteration' : self.iter,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict' : self.optimizer.state_dict(),
                        'loss' : loss}, checkpoint_path)
            
            



    
    

