import numpy as np
import os
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import cv2 
import torchvision
import pandas as pd
import torch
import torchvision.io as io
from torchvision import ops
import torchvision.transforms as tr
import PIL
from utils import *


@torch.no_grad()
def get_cam_max(cam):
    cam = F.relu(cam)
    C,H,W = cam.size()
    cam = cam/(torch.max(cam.view(C,-1), dim=-1, keepdim=True)[0].view(C,1,1) + 1e-9)
    return cam
    
    
class BBdataset(torch.utils.data.Dataset):
    def __init__(self, df, unique_img_id, size = 256):
        super().__init__()
        self.size = size
        self.df = df
        self.unique_img_id = unique_img_id
        self.transforms = tr.Compose([tr.ToTensor(),
                                        tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])
    
    
    def __getitem__(self, index):
        img_id =  self.unique_img_id[index] 
        img_path = self.df[self.df['Image Index']==img_id]['path'].unique()[0]
        image = PIL.Image.open(img_path).resize((self.size, self.size), resample= PIL.Image.BILINEAR)
        image = image.convert('RGB')
        ori_image = image
        image = self.transforms(image)
        return image , np.array(ori_image)
    
    
    def __len__(self):
        return len(self.unique_img_id)
        
class BBAcc_dataset(BBdataset):
    def __getitem__(self, index):
        img_id =  self.unique_img_id[index] 
        img_path = self.df[self.df['Image Index']==img_id]['path'].unique()[0]
        image = PIL.Image.open(img_path).resize((self.size, self.size), resample= PIL.Image.BILINEAR)
        image = image.convert('RGB')
        image = self.transforms(image)
        return image   
    
    
    def __len__(self):
        return len(self.unique_img_id)
        
class Heatmaps():
    def __init__(self, model, model_type = 'default',df = None, img_size = 256):
        self.size = 256
        self.model = model
        self.model_type = model_type
        self.df = df
        self.unique_img_id = df['Image Index'].unique().tolist()
        self.dataset = BBdataset(self.df, self.unique_img_id, size = img_size)
        self.acc_dataset = BBAcc_dataset(self.df, self.unique_img_id, size = img_size)
        
        self.class_to_index =  {'Atelectasis':0, 'Cardiomegaly':1, 'Effusion':2 ,'Infiltrate':3,
                                    'Mass':4,'Nodule':5,'Pneumonia':6,'Pneumothorax':7,'Consolidation':8,
                                    'Edema':9,'Emphysema':10,'Fibrosis':11,'Pleural_Thickening':12, 
                                    'Hernia':13}   
        self.index_to_class = {val:key for key,val in self.class_to_index.items()}
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device = self.device)
        self.model.eval()
        
         
    @staticmethod
    def T_to_N(img):
        img = img.permute(1,2,0).numpy()
        return img    
    

    @torch.no_grad()    
    def get_logit_maps14(self, img, threshold = None, pred_thresh = 0.5): 
        img = img.unsqueeze(0) 
        if self.model_type == 'default':
            logit_maps, preds = self.model.predict(img)  
            
        elif self.model_type == 'pcm':
            logit_maps,_, preds = self.model(img)
        
        elif self.model_type == 'pcm_rv':
            _,logit_maps, preds = self.model(img)
                
        else:
            raise RuntimeError  
        
        logit_maps = logit_maps.squeeze(0)
        preds = preds.squeeze(0).sigmoid()
        preds = (preds>pred_thresh).float()
        logit_maps = get_cam_max(logit_maps)
        
        logit_maps = logit_maps.permute(1,2,0)* preds

        if threshold is not None:
            logit_maps = (logit_maps>threshold).int().cpu().numpy() 
        else:      
            logit_maps = logit_maps.cpu().numpy()
            
        logit_maps = np.split(logit_maps[:,:,:14], 14, axis = -1)
            
        return logit_maps, preds.cpu().numpy()
    
  
    def _plot_heatmaps(self,img, logit_maps, figure_scale = 7):   
        dx, dy = 0.05, 0.05
        x = np.arange(-3.0, 3.0, dx)
        y = np.arange(-3.0, 3.0, dy)
        extent = np.min(x), np.max(x), np.min(y), np.max(y)
        rows = len(logit_maps)
        columns = 1

        fig = plt.figure(figsize=(rows*figure_scale, columns*figure_scale))

        for i,mask in enumerate(logit_maps):
            fig.add_subplot(rows,columns,i+1)
            mask = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)###
            overlayed_img = cv2.addWeighted(img, 0.5, mask, 0.5,0)
            overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
            plt.imshow(overlayed_img, extent=extent)
            plt.title(self.index_to_class[i])   


    # plots heatmaps for all 14 classes
    def plot_heatmaps14(self, img_index, threshold = None, figure_scale = 64): 
        img, ori_image = self.dataset[img_index]
        img = img.to(device = self.device)
        logit_maps, preds = self.get_logit_maps14(img, threshold=threshold)
        print(self.unique_img_id[img_index])
        print(preds)
        self._plot_heatmaps(ori_image, logit_maps[:8], figure_scale = figure_scale) 
    
    
    @torch.no_grad()
    def print_logits(self, img_index): 
        img, _ = self.dataset[img_index]
        img = img.to(device = self.device)
        _, logits = self.get_logit_maps14(img)
        print(self.unique_img_id[img_index])
        print(logits)
        
        
    # plots heatmaps for all 8 bb classes
    def plot_heatmaps8(self, img):
        logit_maps = self.get_logit_maps8(img)  
        self._plot_heatmaps(logit_maps) 
    
    
    #generate bounding boxes list for one class and single image  
    def generate_bb_one_class_one_image(self, threshold_mask):  ######
        cnts,h = cv2.findContours(np.uint8(threshold_mask[:,:,0]*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bb_list = []
        
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            bb_list.append([x,y,x+w,y+h])    
        return bb_list #[[x,y,h,w],[x,y,h,w],[...]]
         
    
    @staticmethod
    def _draw_boxes(img, bb_list, colour, box_width = 4):
        assert type(colour) == tuple
        for bb in bb_list:
            x,y,x2,y2 = bb
            x,y,x2,y2 = int(x), int(y), int(x2), int(y2)
            cv2.rectangle(img, (x, y), (x2, y2), colour, box_width)
        return img    
            
   
    def get_img_path(self, img_id):
        return self.df[self.df['Image Index']==img_id]['path'].unique()[0]      
    
    
    def load_image(self, path):
        image = io.read_image(path, mode = io.image.ImageReadMode.GRAY)
        image = self.transforms(image)
        image = image.expand(3,-1,-1)
        return image
    
    
    def get_list_gt_bb(self,img_index, scale_factor = 4):
        img_id = self.unique_img_id[img_index]
        tdf = self.df[self.df['Image Index']== img_id]
        bb_list = [[] for _ in range(8)]
        for row in tdf.iterrows():
            row1 = row[1] 
            x,y,h,w,c = row1['Bbox [x']//scale_factor, row1['y']//scale_factor, row1['h]']//scale_factor, row1['w']//scale_factor, self.class_to_index[row1['Finding Label']]
            bb_list[c].append([x,y,x+w,y+h])    
        return bb_list      #(class,num_boxes, 4)
    
        
    #plots imgs overlayed with heatmaps and bb boxes (predicted & GT)
    #gt_bb_list: class[bb[x,y,h,w]]
    def _plot_bb(self, img, heat_maps, threshold_masks, gt_bb_list, figure_scale = 64):  ########################################
        img = np.uint8(img)
        
        dx, dy = 0.05, 0.05
        x = np.arange(-3.0, 3.0, dx)
        y = np.arange(-3.0, 3.0, dy)
        extent = np.min(x), np.max(x), np.min(y), np.max(y)
        rows = len(threshold_masks)
        columns = 1
        
        fig = plt.figure(figsize=(rows*figure_scale, columns*figure_scale))
        
        class_bb_list = [[] for _ in range(8)]
        
        for i, mask in enumerate(threshold_masks):
            bb_list = self.generate_bb_one_class_one_image(mask)
            class_bb_list[i] = class_bb_list[i] + bb_list
        
        print(class_bb_list)###########(class, num_boxes, 4)
         
            
        for i in range(len(threshold_masks)):
            img_temp = img.copy()#np.tile(threshold_masks[i],3).copy().astype(np.float32) #img.copy()
            img_temp = self._draw_boxes(img_temp, class_bb_list[i], (0,0,255))
            img_temp = self._draw_boxes(img_temp, gt_bb_list[i], (0,255,0))
            
            
            heatmap = cv2.applyColorMap(np.uint8(heat_maps[i]*255), cv2.COLORMAP_JET)
            overlayed_img = cv2.addWeighted(img_temp, 0.5, heatmap, 0.5,0)
            overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
            
            fig.add_subplot(rows,columns,i+1)
            plt.imshow(overlayed_img, interpolation='bilinear', extent=extent)
            plt.title(self.index_to_class[i])   


    def plot_bb8(self, img_index, threshold = 0.9, figure_scale = 64, heatmap_thresh = None):
        img, ori_image = self.dataset[img_index]
        img = img.to(device = self.device)
        heat_maps, preds = self.get_logit_maps14(img, threshold= heatmap_thresh)
        threshold_masks14, _= self.get_logit_maps14(img, threshold)
        gt_bb_list = self.get_list_gt_bb(img_index)
        print(gt_bb_list)
        print(preds[:8])
        self._plot_bb(ori_image, heat_maps[:8], threshold_masks14[:8], gt_bb_list, figure_scale=figure_scale)
    

    @torch.no_grad()
    def get_localization_metrics(self, threshhold = 0.7, iou=0.1):
        
        #create cams
        cams = [[] for _ in range(8)]
        for img in tqdm(iter(self.acc_dataset), desc='Images'):  
            img = img.to(self.device)
            logit_maps, _ = self.get_logit_maps14(img, threshold=threshhold) 
            for i in range(8):
                cams[i].append(logit_maps[i])
        print('logit maps computed') 
                
        #get classwise-imagewise bounding boxes
        pred_bb_class_numImage_numBoxes_4 = [[] for _ in range(8)]
        for i in range(8):
            for logitmap in cams[i]:
                pred_bb_class_numImage_numBoxes_4[i].append(self.generate_bb_one_class_one_image(logitmap))
                
        print('bboxes computed')
        
        #get classwise image wise bounding boxes
        gt_bb_class_numImage_numBoxes_4 = [[] for _ in range(8)]    
        for i in range(len(self.unique_img_id)):
            gt_bb_class_numBoxes_4 = self.get_list_gt_bb(i)
            for j in range(8):
                gt_bb_class_numImage_numBoxes_4[j].append(gt_bb_class_numBoxes_4[j])
        print('ground truth bboxes computed')        
        
        # compute metrics
        total_TP = torch.tensor(0.)
        total_FP = torch.tensor(0.)
        
        LAcc_list = []
        Ave_FPN_list = []
        TP_list = []
        FP_list = []
        num_GT_list = []
        num_img_list = []
        
        
        for class_num in range(8):
            TP = torch.tensor(0.)
            FP = torch.tensor(0.)
            FN = torch.tensor(0.)
            NUM_GT = torch.tensor(0.)
            NUM_IMG = torch.tensor(0.)
            for image_num in range(len(pred_bb_class_numImage_numBoxes_4[class_num])):
                bb_list = pred_bb_class_numImage_numBoxes_4[class_num][image_num]
                gt_bb_list = gt_bb_class_numImage_numBoxes_4[class_num][image_num]
                
                num_bb = len(bb_list)
                num_gt_bb = len(gt_bb_list)
                
                if num_bb == 0 and num_gt_bb >0:
                    FN+=num_gt_bb
                    # NUM_GT+=num_gt_bb
                
                elif num_bb>0 and num_gt_bb == 0:
                    FP+=num_bb
                
                elif num_bb == 0 and num_gt_bb == 0:
                    pass
                
                else:
                    gt_pred_iou_matrix = ops.box_iou(torch.tensor(gt_bb_list), torch.tensor(bb_list))
                    gt_pred_iou_matrix = (gt_pred_iou_matrix>iou).float()
                    # for TP
                    tp_max = torch.max(gt_pred_iou_matrix, dim=1)[0]
                    TP+= tp_max.sum()
                    
                    #for FP
                    fp_mat = (gt_pred_iou_matrix==0).float().sum()
                    FP+=fp_mat
                    
                    # for FN
                    fn_max = (tp_max==0).float().sum()
                    FN+=fn_max
                    
                
                NUM_GT+=num_gt_bb
                NUM_IMG+=1
            
            total_TP+=TP
            total_FP+=FP    
            LAcc_list.append(round((TP/NUM_GT).item(),3))
            Ave_FPN_list.append(round((FP/NUM_IMG).item(),3))
            TP_list.append(TP.item())
            FP_list.append(FP.item())
            num_GT_list.append(NUM_GT.item())
            num_img_list.append(NUM_IMG.item())
        
        data = {'class':list(self.class_to_index.keys())[:8],
                'LAcc': LAcc_list, 
                'Ave_FPN': Ave_FPN_list, 
                'TP':TP_list, 
                'FP': FP_list, 
                'NUM_GT': num_GT_list, 
                'NUM_IMG': num_img_list}
        
        metric_df = pd.DataFrame(data)            
        print(f'total_TP: {total_TP.item()}   total_FP: {total_FP.item()}')   
        return metric_df, total_TP, total_FP         
                      
   