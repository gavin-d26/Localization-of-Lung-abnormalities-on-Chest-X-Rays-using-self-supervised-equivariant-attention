import os

DEFAULT_HP = {
            'model':'resnet50PCM', #'resnet50', 'resnet50PCM','efficientnetb4', 'efficientnetb4PCM'

            'heatmap_type': 'pcm_rv-pcm',#'pcm_rv-pcm',
            'accumulation': 3,
            'split_index': 4,    #[0,4]

            'lr': 3E-04,
            'lr_pcm':1E-05,
            'batch_size': 32,
            'beta1': 0.9,
            'beta2': 0.9,   ###must be .99 or less
            'weight_decay': 0.0, #1E-04,
            'dropout_rate':0,  #????
            'epochs':35, 
            'epochs_pcm':1,

            'pooling_type':'LSE',
            'gamma': float(10), #for LSE pooling
            'mode':'bilinear',
 
            #lr schedulers
            'factor':0.1, #reduce on plateau decay factor 
            'patience':6, # reduceOnPlateau
          
            'mask_scale':0.2,   
            'rotate_angle':30,
            
            'lambda_ecr':0.5,
            'lambda_msml':0.5,
            }
