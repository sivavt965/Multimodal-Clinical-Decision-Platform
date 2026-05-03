import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm


class DenseNet121Backbone(nn.Module):
    def __init__(self, dropout_p=0.3, pretrained=True):
        super().__init__()
        self.dropout_p = dropout_p
        self.m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = self.m.features

    def forward(self,x,mc_dropout=False):
        f=self.features; drop=self.training or mc_dropout
        x=f.conv0(x); x=f.norm0(x); x=f.relu0(x); x=f.pool0(x)
        x=f.denseblock1(x); x=f.transition1(x); x=F.dropout(x,self.dropout_p,drop)
        x=f.denseblock2(x); x=f.transition2(x); x=F.dropout(x,self.dropout_p,drop)
        x=f.denseblock3(x); x=f.transition3(x); x=F.dropout(x,self.dropout_p,drop)
        x=f.denseblock4(x); x=f.norm5(x); x=F.relu(x,inplace=True)
        x=F.adaptive_avg_pool2d(x,(1,1)).flatten(1)
        return F.dropout(x,self.dropout_p,drop)


class MetaMLP(nn.Module):
    def __init__(self,in_dim=11,out_dim=128,dropout_p=0.2):
        super().__init__()
        self.fc1=nn.Linear(in_dim,64)
        self.fc2=nn.Linear(64,out_dim)
        self.dropout_p=dropout_p
    def forward(self,x,mc_dropout=False):
        drop=self.training or mc_dropout
        x=F.relu(self.fc1(x)); x=F.dropout(x,self.dropout_p,drop)
        return F.relu(self.fc2(x))


class MultiModalFiLMGated(nn.Module):
    def __init__(self,num_labels=8):
        super().__init__()
        self.image_encoder=DenseNet121Backbone()
        self.meta_encoder=MetaMLP(in_dim=11)
        self.meta_to_img=nn.Linear(128,1024)
        self.gates=nn.ModuleList([nn.Linear(1152,1) for _ in range(num_labels)])
        self.heads=nn.ModuleList([nn.Linear(1024,1) for _ in range(num_labels)])

    def forward(self,x,meta,mc_dropout=False):
        hi=self.image_encoder(x,mc_dropout)
        hm=self.meta_encoder(meta,mc_dropout)
        hgate=torch.cat([hi,hm],1)
        meta_img=self.meta_to_img(hm)
        logits=[]; gates=[]
        for k in range(len(self.heads)):
            g=torch.sigmoid(self.gates[k](hgate))
            z=g*hi+(1-g)*meta_img
            logits.append(self.heads[k](z)); gates.append(g)
        return torch.cat(logits,1), torch.cat(gates,1)
