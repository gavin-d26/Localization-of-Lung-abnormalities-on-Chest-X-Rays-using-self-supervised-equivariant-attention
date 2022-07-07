import argparse
import os
import numpy as np 
import pandas as pd
import torch
import torchmetrics as tm
import random
import wandb
import create_model 
import config
import data
import train_utils


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.autograd.set_detect_anomaly(True)

NUM_CLASSES=14
if __name__=="__main__":
    
    #####
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedirectory', type=str, help="path to save directory", required=True)
    parser.add_argument('--name', type=str, help="name of current run", required=True)
    parser.add_argument('--project', type=str, help="name of wandb project", required=True)
    parser.add_argument('--entity', type=str, help="name of wandb entity", required=True)
    parser.add_argument('--model', type=str, help="Type of model (resnet50, resnet50PCM, efficientnetb4, efficientnetb4PCM)", required=True)
    args = parser.parse_args()
    
    PROJECT_NAME=args.project
    ENTITY_NAME=args.entity
    RUN_NAME=args.name
    SAVE_MODEL_PATH=args.savedirectory
    CHECKPOINT_PATH_WEIGHTS = os.path.join(SAVE_MODEL_PATH, RUN_NAME + '_CW_.pt') 
    FINAL_SAVE_PATH_WEIGHTS = os.path.join(SAVE_MODEL_PATH, RUN_NAME + '_FW_.pt') 
    TEST_PATH = r'./files/test_df.csv'
    BB_DF_PATH = r'./files/bounding_boxes.csv'
    K_PATH= r'./files/splits'
    DEFAULT_HP=config.DEFAULT_HP
    
    DEFAULT_HP['model']=args.model
    if DEFAULT_HP['pooling_type']!='LSE':
        DEFAULT_HP['gamma']=None

    if 'PCM' not in DEFAULT_HP['model']:
        DEFAULT_HP['lambda_ecr']=None
        DEFAULT_HP['lambda_msml']=None

    if 'PCM' in DEFAULT_HP['model']:
        DEFAULT_HP['heatmap_type']='pcm_rv-pcm'

    else:
        DEFAULT_HP['heatmap_type']='default'

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    split_index = [DEFAULT_HP['split_index'],]
    train_id_list, validation_id_list  = data.load_test_validation_df(K_PATH, split_index)
    test_id_list = pd.read_csv(TEST_PATH)
    bb_df = df = pd.read_csv(BB_DF_PATH)
    
    train_dataset = data.CXRDataset(train_id_list)
    validation_dataset = data.CXRDataset(validation_id_list, validation = True)
    test_dataset = data.CXRDataset(test_id_list, validation = True)
    
    # train_dataset = torch.utils.data.Subset(train_dataset, range(32)) ##########
    # validation_dataset = torch.utils.data.Subset(validation_dataset, range(32))  ############
    # test_dataset = torch.utils.data.Subset(test_dataset, range(32))  ############
    
    wandb.init(project = PROJECT_NAME, entity = ENTITY_NAME, name = RUN_NAME, config = DEFAULT_HP)
    wandb_config = wandb.config   
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size = wandb_config.batch_size, 
                                               pin_memory = True,
                                               shuffle = True,
                                               drop_last = False,
                                               num_workers = 2)
    
    validation_loader = torch.utils.data.DataLoader(validation_dataset, 
                                                    batch_size = wandb_config.batch_size, 
                                                    pin_memory = True,
                                                    shuffle = False,
                                                    drop_last = False,
                                                    num_workers = 2)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=wandb_config.batch_size, 
                                                pin_memory=True,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=2)
    
    print(RUN_NAME)
    
    metrics_fn_dict = {'AUROC_MACRO':tm.AUROC(num_classes= NUM_CLASSES, average='macro').to(device = DEVICE),
                       'AUROC':tm.AUROC(num_classes=NUM_CLASSES,average=None).to(device=DEVICE),
                       
                       'Precision_MACRO':tm.Precision(num_classes=NUM_CLASSES, average='macro').to(device = DEVICE),
                       'Precision':tm.Precision(num_classes=NUM_CLASSES,average=None).to(device = DEVICE), 
                       
                       'Recall_MACRO':tm.Recall(num_classes=NUM_CLASSES, average='macro').to(device = DEVICE),
                       'Recall':tm.Recall(num_classes=NUM_CLASSES,average=None).to(device=DEVICE), 
                       
                       'F1Score_MACRO':tm.F1Score(num_classes=NUM_CLASSES, average='macro').to(device = DEVICE),
                       'F1Score':tm.F1Score(num_classes=NUM_CLASSES,average=None).to(device=DEVICE),
                       
                       'Specificity_MACRO':tm.Specificity(num_classes=NUM_CLASSES, average='macro').to(device = DEVICE),
                       'Specificity':tm.Specificity(num_classes=NUM_CLASSES,average=None).to(device=DEVICE)
                       }
    
    loss_fn = train_utils.BCElossWithLogitsBalanced()

    if wandb_config.model =='resnet50':
        model = create_model.Resnet50(gamma=wandb_config.gamma, pooling_type=wandb_config.pooling_type)
        model.requires_grad_(False)
        for n, m in model.feature_extractor.backbone.named_children():
            if (n=='layer4') or (n=='layer3'):
                m.requires_grad_(True)
                
        for n,m in model.named_modules():
            if ('bn' in n):
                m.requires_grad_(True)
        
        print('USING RESNET50')
        
        
    elif wandb_config.model=='resnet50PCM':
        model = create_model.Resnet50_PCM(pooling_type = wandb_config.pooling_type, 
                                            gamma = wandb_config.gamma, 
                                            mask_scale = wandb_config.mask_scale,
                                            rotate_angle = wandb_config.rotate_angle,
                                            lambda_ecr=wandb_config.lambda_ecr,
                                            lambda_msml=wandb_config.lambda_msml) 
        
        model.requires_grad_(False)
        for n, m in model.feature_extractor.feature_extractor.backbone.named_children():
            if (n=='layer4') or (n=='layer3'):
                m.requires_grad_(True)
                
        for n,m in model.named_modules():
            if ('bn' in n):
                m.requires_grad_(True)
        
        model.pcm.requires_grad_(True)
        
        print('USING RESNET50PCM')
 
    elif wandb_config.model=='efficientnetb4':
        model = create_model.EfficientNetB4(pooling_type=wandb_config.pooling_type,
                                            gamma=wandb_config.gamma)
        model.requires_grad_(True)
        model.backbone.features[:5].requires_grad_(False)
        for n,m in model.named_modules():
            if ("bn" in n):
                m.requires_grad_(True)
        
        print("USING EFFICIENTNETB4")
    
    
    elif wandb_config.model=='efficientnetb4PCM':
        model = create_model.EfficientNetB4_PCM(pooling_type = wandb_config.pooling_type, 
                                                gamma=wandb_config.gamma, 
                                        mask_scale = wandb_config.mask_scale,
                                        rotate_angle = wandb_config.rotate_angle,
                                        lambda_ecr=wandb_config.lambda_ecr,
                                        lambda_msml=wandb_config.lambda_msml) 
        print("USING EFFICIENTNETB4PCM")
        model.requires_grad_(True)
        model.feature_extractor.backbone.features[:5].requires_grad_(False)
        for n,m in model.named_modules():
            if ("bn" in n):
                m.requires_grad_(True)
    

    print("model type: ", type(model))
    num_params = torch.nn.utils.parameters_to_vector(model.parameters()).numel()
    wandb.run.summary['num_params'] = num_params
    
    model.to(device = DEVICE)
    
    if 'PCM' in wandb_config.model:
        optimizer1 = torch.optim.Adam(model.feature_extractor.parameters(), 
                                    lr=wandb_config.lr, 
                                    betas=(wandb_config.beta1, wandb_config.beta2),
                                    weight_decay = wandb_config.weight_decay)
        lr_scheduler1={}
        lr_scheduler1['ROP'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 
                                                                    patience=wandb_config.patience, 
                                                                    min_lr = 1e-7,
                                                                    mode='max', 
                                                                    verbose=True)
        print("lrs1: ROP")
        

    lr_scheduler2={}
    if 'PCM' in wandb_config.model:
        optimizer2 = torch.optim.Adam(model.pcm.parameters(), 
                                 lr=wandb_config.lr_pcm, 
                                 betas=(wandb_config.beta1, wandb_config.beta2),
                                 weight_decay = wandb_config.weight_decay)
        
        epochs2 = wandb_config.epochs_pcm
        print("optimizer2 pcm")
        
    else:
        optimizer2 = torch.optim.Adam(model.parameters(), 
                                 lr=wandb_config.lr, 
                                 betas=(wandb_config.beta1, wandb_config.beta2),
                                 weight_decay = wandb_config.weight_decay)
        
        lr_scheduler2['ROP'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 
                                                                    patience=wandb_config.patience, 
                                                                    min_lr = 1e-7,
                                                                    mode='max', 
                                                                    verbose=True)
        epochs2 = wandb_config.epochs
        print("lrs2: ROP")
        
    if "PCM" in wandb_config.model:
        print("Training Feature Extractor for PCM model")
        model_fe = model.feature_extractor
        model_fe, check_point = train_utils.train_model(
                            model_fe, 
                            optimizer = optimizer1, 
                            metrics_fn_dict=metrics_fn_dict, #{'train_acc':AccuracyObj,}
                            train_loader = train_loader,
                            eval_loader = validation_loader,
                            device = DEVICE,
                            loss_fn = loss_fn,
                            epochs = wandb_config.epochs,
                            wandb_p = wandb,
                            check_point= {'name':'AUROC_MACRO_EVAL','value':None, 'path':CHECKPOINT_PATH_WEIGHTS, 'type':'max'}, #{'name':str, 'value':float, 'type':'max', 'path':str}
                            final_save_path = FINAL_SAVE_PATH_WEIGHTS,
                            lr_scheduler = lr_scheduler1
                            )
        
        model.feature_extractor.load_model_weights(check_point['path'])
        
        model.cuda()
        model.requires_grad_(False)
        model.pcm.requires_grad_(True)
       
    
    print("Training regular model or pcm module")
    model, check_point = train_utils.train_model(
                        model, 
                        optimizer = optimizer2, 
                        metrics_fn_dict=metrics_fn_dict, #{'train_acc':AccuracyObj,}
                        train_loader = train_loader,
                        eval_loader = validation_loader,
                        device = DEVICE,
                        loss_fn = loss_fn,
                        epochs = epochs2,
                        wandb_p = wandb,
                        check_point= {'name':'AUROC_MACRO_EVAL','value':None, 'path':CHECKPOINT_PATH_WEIGHTS, 'type':'max'}, #{'name':str, 'value':float, 'type':'max', 'path':str}
                        final_save_path = FINAL_SAVE_PATH_WEIGHTS,
                        lr_scheduler = lr_scheduler2
                        )
    
    
    model.eval()
    train_utils.test_model(model,
               test_loader = test_loader,
               metrics_fn_dict = metrics_fn_dict,
               wandb_p = wandb, 
               device=DEVICE,
               loss_fn=loss_fn,
               model_version='Final')
    
    for hmtype in wandb_config.heatmap_type.split('-'):
        print("heatmap_type: ", hmtype)
        train_utils.save_localization_metrics(model, hmtype, bb_df, wandb, RUN_NAME, 'Final')
        
    
    model.load_model_weights(check_point['path'])
    model.eval()
    
    train_utils.test_model(model,
               test_loader = test_loader,
               metrics_fn_dict = metrics_fn_dict,
               wandb_p = wandb, 
               device=DEVICE,
               loss_fn=loss_fn,
               model_version='Checkpoint'
               )        
    
    model.cpu()
    for hmtype in wandb_config.heatmap_type.split('-'):
        print("heatmap_type: ", hmtype)
        train_utils.save_localization_metrics(model, hmtype, bb_df, wandb, RUN_NAME, 'Checkpoint')
    
    