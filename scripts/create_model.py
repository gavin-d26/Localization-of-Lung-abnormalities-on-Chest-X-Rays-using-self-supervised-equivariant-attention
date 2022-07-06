import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision
import utils
import config

torch.manual_seed(0)
# np.random.seed(0)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ACCUMULATION = config.DEFAULT_HP['accumulation']

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
        
    def save_model_weights(self, path):
        torch.save(self.state_dict(), path)


    def load_model_weights(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


class ClsModel(BaseModel):
    def __init__(self):
        super().__init__()
    
    
    def train_step(self, data, target, batch_idx, loss_fn, optimizer, grad_scalar):
        preds = self(data)
        loss, _, _ = loss_fn(preds, target)
        
        loss = loss/ACCUMULATION
        loss.backward()
        
        if (batch_idx % ACCUMULATION==0) or (batch_idx + 1==self.num_train_batches):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        return loss, preds, target
    
    
    def eval_step(self, data, target, batch_idx, loss_fn):
        preds = self(data)
        loss,_, _ = loss_fn(preds, target)
        return loss, preds, target 


class PcmModel(BaseModel):
    def __init__(self):
        super().__init__()
    
    
    def affine_transform(self, x, a):
        return torchvision.transforms.functional.rotate(x, a, interpolation= torchvision.transforms.functional.InterpolationMode.BILINEAR)    
    
    
    def train_step(self, data, target, batch_idx, loss_fn, optimizer, grad_scalar):
        self.eval()
        img1, labels = data, target
        
        a = torchvision.transforms.RandomRotation.get_params([-self.rotate_angle, self.rotate_angle])
        img2 = self.affine_transform(img1, a)
        N, C, H, W = img1.size()
        
        bg_score = torch.ones((N, 1), device=DEVICE)
        labels = torch.cat((labels, bg_score), dim=1)
        labels = labels.unsqueeze(-1).unsqueeze(-1)
        cam1, cam_rv1, pred1 = self(img1)
        cam2, cam_rv2, pred2 = self(img2)
        
        label1 = F.adaptive_avg_pool2d(cam_rv1, (1,1))
        label2 = F.adaptive_avg_pool2d(cam_rv2, (1,1))
        #loss_msml
        loss_msml1 = F.multilabel_soft_margin_loss(label1[:,:14,:,:], labels[:,:14,:,:])
        loss_msml2 = F.multilabel_soft_margin_loss(label2[:,:14,:,:], labels[:,:14,:,:])
        
        cam1 = self.affine_transform(utils.max_norm(cam1), a)*labels
        cam_rv1 = self.affine_transform(utils.max_norm(cam_rv1), a)*labels 
   
        ns,cs,hs,ws = cam2.size()

        cam2 = utils.max_norm(cam2)*labels 
        cam_rv2 = utils.max_norm(cam_rv2)*labels 
        
        cam1[:,14,:,:] = 1-torch.max(cam1[:,:14,:,:],dim=1)[0]
        cam2[:,14,:,:] = 1-torch.max(cam2[:,:14,:,:],dim=1)[0] 
        
        # loss_ecr
        tensor_ecr1 = torch.abs(utils.max_onehot(cam2.detach()) - cam_rv1)
        tensor_ecr2 = torch.abs(utils.max_onehot(cam1.detach()) - cam_rv2)
        
        loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns,-1), k=(int)(cs*hs*ws*self.mask_scale), dim=-1)[0])
        loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns,-1), k=(int)(cs*hs*ws*self.mask_scale), dim=-1)[0])

        loss_ecr = (loss_ecr1 + loss_ecr2)
   
        loss_msml = (loss_msml1 + loss_msml2)

        loss = loss_msml*self.lambda_msml + self.lambda_ecr * loss_ecr 
        loss = loss/ACCUMULATION    
        loss.backward()
        
        if (batch_idx % ACCUMULATION == 0) or (batch_idx + 1== self.num_train_batches):
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) 

        return loss, pred1[:,:14], labels[:,:14,:,:].squeeze(-1).squeeze(-1)


    def eval_step(self, data, target, batch_idx, loss_fn):
        _,_,preds = self(data)
        loss,_,_ = loss_fn(preds[:, :14], target)
        return loss, preds[:,:14], target[:,:14]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 
                                    out_channels, 
                                    (kernel_size, kernel_size), 
                                    stride = stride, 
                                    padding = (padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x
    
        
class _upsample(nn.Module):
    def __init__(self, in_channels, out_channels, mode = 'bilinear'):
        super(_upsample, self).__init__()
        self.upsample_block = nn.Sequential(nn.Upsample(scale_factor=2, mode = mode, align_corners=True), 
                                            ConvBlock(in_channels, out_channels))
        
        
    def forward(self, x):
        return self.upsample_block(x)
               

class _Resnet50(nn.Module):
    def __init__(self):
        super(_Resnet50, self).__init__()
        backbone = torchvision.models.resnet50(pretrained = True)
        backbone.fc = torch.nn.Identity()
        backbone.avgpool = torch.nn.Identity() 
        self.backbone = backbone 

        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x1 = self.backbone.layer1(x)   #64
        x2 = self.backbone.layer2(x1)  #32
        x3 = self.backbone.layer3(x2)  #16
        x4 = self.backbone.layer4(x3)  #8
        return x4    #no sigmoid   
    
    
class Resnet50(ClsModel):
    def __init__(self, 
                 num_classes = 14,  
                 gamma = 10,
                 pooling_type = 'GAP'):
        super().__init__() 
        self.gamma = gamma
        self.num_classes = num_classes   
        self.feature_extractor = _Resnet50()
        self.pooling_type = pooling_type
        
        if self.pooling_type == 'GAP':
            self.pooling = nn.AdaptiveAvgPool2d((1,1))
        elif self.pooling_type == 'LSE':
            self.pooling = utils.LogSumExpPool(self.gamma)  
        else:
            raise ValueError('Unknown pooling type') 
             
        self.fc = nn.Conv2d(2048, num_classes, (1,1))
    
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pooling(x)
        x = self.fc(x).squeeze(-1).squeeze(-1)  
        return x    
    
    
    @torch.no_grad()
    def predict(self,x):
        n,c,h,w = x.size()
        x_maps = self.feature_extractor(x)
        x = self.fc(x_maps)
        x = F.interpolate(x, size = (h,w), align_corners=False, mode= 'bilinear')
        x_feat = self.pooling(x_maps)
        preds = self.fc(x_feat).squeeze(-1).squeeze(-1)  
        return x, preds
              
              
class ConvMerge(nn.Module):
    def __init__(self, n, m, innerdim1 = None, innerdim2 = None, out_dim=None):
        super().__init__()
        innerdim1 = n//4 if innerdim1 is None else innerdim1
        innerdim2 = m//4 if innerdim2 is None else innerdim2
        out_dim = m if out_dim is None else out_dim
        self.layer_n = ConvBlock(n, innerdim1, kernel_size=1, padding=0)
        self.layer_m = ConvBlock(m, innerdim2, kernel_size=1, padding=0)
        self.layer_mn = ConvBlock(innerdim1 + innerdim2, out_dim, kernel_size=1, padding=0)
        
        
    def forward(self, nfeat, mfeat):
        mfeat = self.layer_m(mfeat)
        nfeat = self.layer_n(nfeat)
        nfeat = F.interpolate(nfeat, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.layer_mn(torch.cat((nfeat,mfeat), dim=1))
        return out
        

class _Resnet50_MS(nn.Module):
    def __init__(self, pretrained = True):
        super(_Resnet50_MS, self).__init__()
        backbone = torchvision.models.resnet50(pretrained = pretrained)###
        backbone.fc = torch.nn.Identity()
        backbone.avgpool = torch.nn.Identity() 
        self.backbone = backbone 
        
        self.rc43 = ConvMerge(2048, 1024)
        self.rc32 = ConvMerge(1024, 512)
        
    def forward(self, x):
        #initial layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        #feature layers
        x1 = self.backbone.layer1(x)   #64
        x2 = self.backbone.layer2(x1)  #32
        x3 = self.backbone.layer3(x2)  #16
        x4 = self.backbone.layer4(x3)  #8
        
        xp = self.rc43(x4,x3)
        xp = self.rc32(xp,x2)
        return xp
        
class Resnet50_MS(ClsModel):
    def __init__(self, 
                 num_classes = 14, 
                 gamma = 10,
                 pooling_type = 'GAP',
                 pretrained = True
                 ):
        super().__init__() 
        self.gamma = gamma
        self.num_classes = num_classes   
        self.feature_extractor = _Resnet50_MS(pretrained=pretrained)
        self.pooling_type = pooling_type
        
        if self.pooling_type == 'GAP':
            self.pooling = nn.AdaptiveAvgPool2d((1,1))
        elif self.pooling_type == 'LSE':
            self.pooling = utils.LogSumExpPool(self.gamma)  
        
        else:
            raise ValueError('Unknown pooling type') 
             
        self.fc = nn.Conv2d(512, num_classes, (1,1))
    
    
    def forward(self, x):
        x_maps = self.feature_extractor(x)
        preds = self.fc(self.pooling(x_maps)).squeeze(-1).squeeze(-1)
        return preds[:,:14]
        
        
    def get_cam_forward(self, x):
        x_maps = self.feature_extractor(x)
        preds = self.fc(self.pooling(x_maps)).squeeze(-1).squeeze(-1)  
        cam = self.fc(x_maps)
        return x_maps, cam , preds   
    
    
    @torch.no_grad()
    def predict(self,x):
        n,c,h,w = x.size()
        x_maps = self.feature_extractor(x)
        cam = self.fc(x_maps)
        cam = F.interpolate(cam, size = (h,w), align_corners=True, mode= 'bilinear')
        preds = self.fc(self.pooling(x_maps)).squeeze(-1).squeeze(-1)  
        return cam, preds


class Pcm(nn.Module):
    def __init__(self, dim_in, window_size=32):
        super().__init__()
        self.window_size = window_size
        self.to_qk = nn.Conv2d(dim_in, dim_in//2, kernel_size = (1,1),bias=False)


    def forward(self, x, v, mask=None):
        b, n, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p
        p = self.window_size
        x = rearrange(x, 'b n (n1 p1) (n2 p2) -> (b n1 n2) n p1 p2', p1=p,p2=p,n1=n1,n2=n2)
        v = rearrange(v, 'b n (n1 p1) (n2 p2) -> (b n1 n2) (p1 p2) n', p1=p,p2=p,n1=n1,n2=n2)
        q = self.to_qk(x)
        q = rearrange(q, 'b c p1 p2 -> b (p1 p2) c')
        q = q/(torch.norm(q, dim=-1, keepdim=True) + 1e-6)
        dots = torch.einsum('b i k, b j k -> b i j', q, q)
        dots = F.relu(dots)
        
        attn = dots/(torch.sum(dots, dim=-1, keepdim=True)+1e-6)
        out = torch.einsum('b i j, b j k -> b i k', attn, v)
        out = rearrange(out, '(b n1 n2) (p1 p2) n -> b n (n1 p1) (n2 p2)', p1=p, p2=p, n1=n1, n2=n2)
        return out
        

class Resnet50_PCM(PcmModel):
    def __init__(self, 
                 num_classes = 15, 
                 gamma = 10, 
                 scale_factor = None, 
                 mask_scale = None,
                 pooling_type = 'GAP',
                 rotate_angle=30,
                 lambda_ecr =1,
                 lambda_msml = 1,
                 pretrained=True):
        super().__init__() 
        self.lambda_ecr = lambda_ecr
        self.lambda_msml = lambda_msml
        self.rotate_angle = rotate_angle
        self.gamma = gamma
        self.scale_factor = scale_factor
        self.mask_scale = mask_scale
        self.pooling_type = pooling_type
        self.feature_extractor = Resnet50_MS(
                                             num_classes = num_classes, 
                                             pooling_type = pooling_type,
                                             gamma = gamma,
                                             pretrained=pretrained
                                             )
        

        self.pcm = Pcm(512, window_size=32)  


    def forward(self, x):
        N,C,H,W = x.size()
        x_maps, cam , preds  = self.feature_extractor.get_cam_forward(x)
        n,c,h,w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())######
            cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
            cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
            cam_d[:,-1,:,:] = 1-torch.max(cam_d_norm[:,:-1,:,:], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:,:-1,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,:-1,:,:][cam_d_norm[:,:-1,:,:] < cam_max] = 0
            
        cam_rv = F.interpolate(self.pcm(x_maps.detach(), cam_d_norm), (H,W), mode='bilinear', align_corners=True)
        cam = F.interpolate(cam, (H,W), mode='bilinear', align_corners=True)
        return cam, cam_rv, preds
    

class EfficientNetB4(ClsModel):
    def __init__(self, num_classes=14, pretrained = True, pooling_type='LSE', gamma=10) -> None:
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b4(pretrained=pretrained)
        
        self.backbone.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), 
                                              nn.Linear(in_features=1792, out_features=num_classes, bias=True))
        if pooling_type == 'GAP':
            self.backbone.avgpool = nn.AdaptiveAvgPool2d((1,1))
        elif pooling_type == 'LSE':
            self.backbone.avgpool = utils.LogSumExpPool(gamma)
            
            
    def forward(self, x):
        out = self.backbone(x)
        return out
    
    
    @torch.no_grad()
    def predict(self,x):
        n,c,h,w = x.size()
        x_maps = self.backbone.features(x)
        preds = self.backbone.classifier(self.backbone.avgpool(x_maps).squeeze(-1).squeeze(-1))
        x_maps= rearrange(x_maps, 'b c h w -> b h w c')
        cam = self.backbone.classifier(x_maps)
        cam = rearrange(cam, 'b h w c -> b c h w')
        cam = F.interpolate(cam, size = (h,w), align_corners=True, mode= 'bilinear')
        return cam, preds


class EfficientNetB4_MS(ClsModel):
    def __init__(self, num_classes=14, pretrained=True, pooling_type = 'LSE', gamma=10):
        super().__init__()
        self.gamma = gamma
        self.pooling_type = pooling_type
        self.backbone = torchvision.models.efficientnet_b4(pretrained=pretrained)
        self.backbone.avgpool = nn.Identity()
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), 
                                                nn.Conv2d((1792//4 + 160)//4 + 56, num_classes, kernel_size=(1,1),bias=True))
        
        self.rc43 = ConvMerge(1792, 160, innerdim1=1792//4, innerdim2=160, out_dim=(1792//4 + 160))
        self.rc32 = ConvMerge((1792//4 + 160), 56, innerdim1=None, innerdim2=56, out_dim=(1792//4 + 160)//4 + 56)
        
        if self.pooling_type == 'GAP':
            self.pooling = nn.AdaptiveAvgPool2d((1,1))
        elif self.pooling_type == 'LSE':
            self.pooling = utils.LogSumExpPool(self.gamma)  
        
        else:
            raise ValueError('Unknown pooling type')
            
            
    def forward(self, x):
        x = self.backbone.features[0:2](x)   #128
        x1 = self.backbone.features[2](x)    #64
        x2 = self.backbone.features[3](x1)   #32
        x3 = self.backbone.features[4:6](x2) #16
        x4 = self.backbone.features[6:](x3)  #8
        
        x_maps = self.rc43(x4,x3)
        x_maps = self.rc32(x_maps,x2)
        x_pred = self.pooling(x_maps)
        preds = self.classifier(x_pred).squeeze(-1).squeeze(-1)
        return preds[:,:14]
    
    
    def get_cam_forward(self, x):
        x = self.backbone.features[0:2](x)   #128
        x1 = self.backbone.features[2](x)    #64
        x2 = self.backbone.features[3](x1)   #32
        x3 = self.backbone.features[4:6](x2) #16
        x4 = self.backbone.features[6:](x3)  #8
        
        x_maps = self.rc43(x4,x3)
        x_maps = self.rc32(x_maps,x2)
        
        cam = self.classifier(x_maps)
        x_pred = self.pooling(x_maps)
        preds = self.classifier(x_pred).squeeze(-1).squeeze(-1)
        
        return x_maps, cam, preds
    
    
    @torch.no_grad()
    def predict(self,x):
        n,c,h,w = x.size()
        x = self.backbone.features[0:2](x)   #128
        x1 = self.backbone.features[2](x)    #64
        x2 = self.backbone.features[3](x1)  #32
        x3 = self.backbone.features[4:6](x2) #16
        x4 = self.backbone.features[6:](x3)  #8
        
        x_maps = self.rc43(x4,x3)
        x_maps = self.rc32(x_maps,x2)
        
        cam = self.classifier(x_maps)
        x_pred = self.pooling(x_maps)
        preds = self.classifier(x_pred).squeeze(-1).squeeze(-1)
        cam = F.interpolate(cam, size = (h,w), align_corners=True, mode= 'bilinear')
        return cam, preds
    
    
class EfficientNetB4_PCM(PcmModel):
    def __init__(self, 
                 num_classes = 15,
                 pooling_type = 'GAP', 
                 scale_factor = None, 
                 mask_scale = None,
                 rotate_angle=30,
                 lambda_ecr =1,
                 lambda_msml = 1,
                 pretrained = True,
                 gamma = 10):
        super().__init__() 
        self.lambda_ecr = lambda_ecr
        self.lambda_msml = lambda_msml
        self.rotate_angle = rotate_angle
        self.scale_factor = scale_factor
        self.mask_scale = mask_scale
        self.pooling_type = pooling_type
        self.feature_extractor = EfficientNetB4_MS(
                                             num_classes=num_classes,
                                             pretrained=pretrained,
                                             pooling_type=pooling_type,
                                             gamma=gamma
                                             )
        
        self.pcm = Pcm(208)   


    def forward(self, x):
        N,C,H,W = x.size()
        x_maps, cam , preds  = self.feature_extractor.get_cam_forward(x)

        n,c,h,w = cam.size()
        
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
            cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
            cam_d[:,-1,:,:] = 1-torch.max(cam_d_norm[:,:-1,:,:], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:,:-1,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,:-1,:,:][cam_d_norm[:,:-1,:,:] < cam_max] = 0

        cam_rv = F.interpolate(self.pcm(x_maps.detach(), cam_d_norm), (H,W), mode='bilinear', align_corners=True)
        cam = F.interpolate(cam, (H,W), mode='bilinear', align_corners=True)
        
        return cam, cam_rv, preds       
