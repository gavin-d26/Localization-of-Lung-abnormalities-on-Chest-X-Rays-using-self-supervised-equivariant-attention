import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import heatmaps

CLASS_TO_INDEX =  {'Atelectasis':0, 'Cardiomegaly':1, 'Effusion':2 ,'Infiltrate':3,
                                    'Mass':4,'Nodule':5,'Pneumonia':6,'Pneumothorax':7,'Consolidation':8,
                                    'Edema':9,'Emphysema':10,'Fibrosis':11,'Pleural_Thickening':12, 
                                    'Hernia':13, 'No_finding':14}

INDEX_TO_CLASS = {val:key for key,val in CLASS_TO_INDEX.items()}

def compute_classwise_dict(metrics, metric_name, type='TRAIN'):
    metrics_dict={}
    for n in range(metrics.shape[0]):
        metrics_dict[metric_name + '_' +INDEX_TO_CLASS[n] + '_' + type] = metrics[n]
    return metrics_dict


class BCElossWithLogits(nn.Module):
    def __init__(self, PW=None):
        super().__init__()
        self.PW = PW
        
        
    def forward(self, pred, target):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, pos_weight=self.PW)
        return loss, None, None
        
    
class BCElossWithLogitsBalanced(nn.Module):
    def __init__(self,*args, **kwargs):
        super().__init__()
    
    
    def forward(self, pred, target):
        C = target.shape[0]
        P = target.sum(dim=0)
        N = C-P
        Bp = C/(P+ 1E-6)
        Bn = C/(N+ 1E-6)
        pred = torch.sigmoid(pred)
        return -(Bp*target*(torch.log(pred.clamp(min=1e-6)))+Bn*(1-target)*torch.log((1-pred).clamp(min=1e-6))).mean(), Bp, Bn


def save_localization_metrics(model, heatmap_type, bb_df, wandb, run_name, model_version):
    viz = heatmaps.Heatmaps(model, df=bb_df, model_type=heatmap_type)
    metrics_df, total_TP, total_FP = viz.get_localization_metrics()
    wandb.summary['total_TP'+'_'+heatmap_type+'_'+model_version]=total_TP
    wandb.summary['total_FP'+'_'+heatmap_type+'_'+model_version]=total_FP
    wandb.log({run_name+'-CW-'+heatmap_type+'_'+model_version: wandb.Table(dataframe=metrics_df)})


def plot_roc(pred, target, wandb_p, model_version):
    pred = pred[:,:14]
    num_classes = pred.shape[-1]
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    line_type = ["-" if i<7 else "--" for i in range(14)]
    line_type.append("-.")
    colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g','c', 'r', 'm', 'y', 'k', 'k']

    plt.figure(figsize=(8,8))
    lw = 1

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="-.")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title("Receiver operating characteristic curve", fontsize = 15)

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(target[:, i], pred[:, i])
        plt.plot(
        fpr,
        tpr,
        label=INDEX_TO_CLASS[i],
        color = colour[i],
        linestyle=line_type[i],
        linewidth=1)

    plt.legend(loc="lower right")
    wandb_p.log({'ROC_CURVE_'+model_version: wandb_p.Image(plt)})


def train_step_function(model, 
                        optimizer = None, 
                        metrics_fn_dict=None, 
                        train_loader = None,
                        device = None,
                        loss_fn = None,
                        grad_scalar = None,
                        lr_scheduler = None
                        ):
    
    if lr_scheduler is None:
        lr_scheduler={}
    
    model.train()
    loss_epoch = 0
    count = 0+1e-6
    preds_list = None
    targets_list = None
    
    metric_value_dict = {}
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Train')):
        data, target = data.to(device = device), target.to(device= device)
        loss, preds, target = model.train_step(data, target, batch_idx, loss_fn, optimizer, grad_scalar)
        if 'CA' in lr_scheduler:
            lr_scheduler['CA'].step()
            
        loss_epoch+=loss.detach()
        count+=1
        
        if preds_list == None:
            preds_list = preds.detach()
        else:    
            preds_list = torch.concat((preds_list, preds.detach()), dim=0)
            
        if targets_list == None:
            targets_list = target.detach()
        else:    
            targets_list = torch.concat((targets_list, target.detach()), dim=0)
        
    loss_epoch = loss_epoch/count
    
    targets_list = targets_list.int()
    if metrics_fn_dict is not None:        
        for key in metrics_fn_dict.keys():        
            metrics = metrics_fn_dict[key](preds_list, targets_list)
            if metrics.dim()==0:
                metric_value_dict[key+'_TRAIN'] = metrics
            
            else:
                class_wise_metric_dict = compute_classwise_dict(metrics, key, 'TRAIN')
                metric_value_dict.update(class_wise_metric_dict)
              
    metric_value_dict['Loss'+'_TRAIN'] = loss_epoch
    return model, metric_value_dict
    

@torch.no_grad()    
def eval_step_function(model,  
               metrics_fn_dict=None, 
               eval_loader = None,
               device = None,
               loss_fn = None,
               metric_value_dict = None
               ):
    
    model.eval()
    loss_epoch = 0
    count = 0+1e-6
    preds_list = None
    targets_list = None
    
    if metric_value_dict is None:
        metric_value_dict = {}
    
    for batch_idx, (data, target) in enumerate(tqdm(eval_loader, desc='Eval')):
        data, target = data.to(device = device), target.to(device = device)
        loss, preds, target = model.eval_step(data, target, batch_idx, loss_fn)
        loss_epoch+=loss.detach()
        count+=1
        
        if preds_list == None:
            preds_list = preds.detach()
        else:    
            preds_list = torch.concat((preds_list, preds.detach()), dim=0)
            
        if targets_list == None:
            targets_list = target.detach()
        else:    
            targets_list = torch.concat((targets_list, target.detach()), dim=0)
    
    loss_epoch = loss_epoch/count
    targets_list = targets_list.int()
    if metrics_fn_dict is not None:        
        for key in metrics_fn_dict.keys():        
            metrics = metrics_fn_dict[key](preds_list, targets_list)
            if metrics.dim()==0:
                metric_value_dict[key+'_EVAL'] = metrics
            else:
                class_wise_metric_dict = compute_classwise_dict(metrics, key, 'EVAL')
                metric_value_dict.update(class_wise_metric_dict) 
            
    metric_value_dict['Loss'+'_EVAL'] = loss_epoch
    return model, metric_value_dict  


def train_model(model, 
               optimizer = None, 
               metrics_fn_dict=None, #{'train_acc':AccuracyObj,}
               train_loader = None,
               eval_loader = None,
               device = None,
               loss_fn = None,
               grad_scalar = None,
               epochs = None,
               wandb_p = None,
               check_point= None, #{'name':str, 'value':float, 'type':'max', 'path':str}
               final_save_path = None,
               lr_scheduler = None
               ):  
    
    
    if lr_scheduler is None:
        lr_scheduler={}
    
    if hasattr(model, 'num_train_batches') is False:
        model.num_train_batches = len(train_loader)
    
    if check_point is not None:
        if check_point['value'] is None:
            if check_point['type']=='max':
                check_point['value']=-1e9
            
            elif check_point['type']=='min': 
                check_point['value']= 1e9 
            
            else:
                raise RuntimeError    
   
    if final_save_path is not None:
        model.save_model_weights(final_save_path)
    
    for epoch in range(epochs):
        
        print('Epoch', epoch)
        
        model, metric_value_dict = train_step_function(model, 
                                                        optimizer = optimizer, 
                                                        metrics_fn_dict=metrics_fn_dict, 
                                                        train_loader = train_loader,
                                                        device = device,
                                                        loss_fn= loss_fn,
                                                        grad_scalar = grad_scalar,
                                                        lr_scheduler=lr_scheduler
                                                       )
        
        model, metric_value_dict = eval_step_function(model,  
                                                        metrics_fn_dict= metrics_fn_dict, 
                                                        eval_loader = eval_loader,
                                                        device = device,
                                                        loss_fn= loss_fn,
                                                        metric_value_dict = metric_value_dict)
        
        if 'ROP' in lr_scheduler:
            lr_scheduler['ROP'].step(metric_value_dict['AUROC_MACRO_EVAL'])
        
        # wandb log
        if wandb_p is not None:
            wandb_p.log(metric_value_dict)
        
        #checkpointing
        if check_point is not None:
            if check_point['type'] == 'max':
                if check_point['value']<metric_value_dict[check_point['name']]:
                    check_point['value'] = metric_value_dict[check_point['name']]
                    model.save_model_weights(check_point['path']) 
                    if wandb_p is not None:
                        wandb_p.run.summary[check_point['name']+'_CHECKPOINT'] = check_point['value']
                    
            
            elif check_point['type'] == 'min':
                if check_point['value']>metric_value_dict[check_point['name']]:
                    check_point['value'] = metric_value_dict[check_point['name']]
                    model.save_model_weights(check_point['path']) 
                    if wandb_p is not None:
                        wandb_p.run.summary[check_point['name']+'_CHECKPOINT'] = check_point['value']
                        
            else:
                raise RuntimeError    
              
    if final_save_path is not None:
        model.save_model_weights(final_save_path)                     
   
    if check_point is not None:
        return model, check_point   
    else:
        return model


@torch.no_grad()
def test_model(model,  
               wandb_p,
               metrics_fn_dict=None, 
               test_loader = None,
               device = None,
               loss_fn = None,
               model_version = None,
               ):
    
    model.eval()
    loss_epoch = 0
    count = 0+1e-6
    preds_list = None
    targets_list = None
    
    
    for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc='Test')):
        data, target = data.to(device = device), target.to(device = device)
        loss, preds, target = model.eval_step(data, target, batch_idx, loss_fn)
        loss_epoch+=loss.detach()
        count+=1
        
        if preds_list == None:
            preds_list = preds.detach()
        else:    
            preds_list = torch.concat((preds_list, preds.detach()), dim=0)
            
        if targets_list == None:
            targets_list = target.detach()
        else:    
            targets_list = torch.concat((targets_list, target.detach()), dim=0)
    
    
    loss_epoch = loss_epoch/count
    targets_list = targets_list.int()
    if metrics_fn_dict is not None:        
        for key in metrics_fn_dict.keys():        
            metrics = metrics_fn_dict[key](preds_list, targets_list)
            if metrics.dim()==0:
                wandb_p.run.summary[key+'_TEST_'+ model_version] = metrics
            
            else:
                class_wise_metric_dict = compute_classwise_dict(metrics, key, 'TEST')
                for key1, value1 in class_wise_metric_dict.items():
                    wandb_p.run.summary[key1 + '_' + model_version] = value1
                
    wandb_p.run.summary['Loss'+'_TEST_'+ model_version] = loss_epoch
    plot_roc(preds_list, targets_list, wandb_p, model_version)
    return None