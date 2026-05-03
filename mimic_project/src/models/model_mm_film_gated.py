# model_mm_film_gated.py  (MULTIMODAL, 3CH)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class DenseNet121Backbone(nn.Module):
    """
    3-channel DenseNet121 backbone.
    Output embedding: [B,1024]
    Functional dropout supports MC-dropout without changing BN mode:
      model.eval(); forward(mc_dropout=True)
    """
    def __init__(self, dropout_p: float = 0.30, pretrained: bool = True):
        super().__init__()
        self.dropout_p = float(dropout_p)
        self.m = tvm.densenet121(weights=(tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None))
        self.features = self.m.features  # conv0 expects 3-ch input

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        f = self.features
        drop_on = bool(self.training or mc_dropout)

        x = f.conv0(x); x = f.norm0(x); x = f.relu0(x); x = f.pool0(x)

        x = f.denseblock1(x); x = f.transition1(x)
        x = F.dropout(x, p=self.dropout_p, training=drop_on)

        x = f.denseblock2(x); x = f.transition2(x)
        x = F.dropout(x, p=self.dropout_p, training=drop_on)

        x = f.denseblock3(x); x = f.transition3(x)
        x = F.dropout(x, p=self.dropout_p, training=drop_on)

        x = f.denseblock4(x); x = f.norm5(x); x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # [B,1024]
        x = F.dropout(x, p=self.dropout_p, training=drop_on)
        return x


class MetaMLP(nn.Module):
    """7 -> 128 MLP; functional dropout supports MC-dropout."""
    def __init__(self, in_dim: int = 7, out_dim: int = 128, dropout_p: float = 0.20):
        super().__init__()
        self.dropout_p = float(dropout_p)
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        drop_on = bool(self.training or mc_dropout)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, p=self.dropout_p, training=drop_on)
        x = F.relu(self.fc2(x), inplace=True)
        return x


class FiLM(nn.Module):
    """Identity-init FiLM + LayerNorm."""
    def __init__(self, meta_dim: int = 128, img_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(meta_dim, 256)
        self.fc2 = nn.Linear(256, 2 * img_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.norm = nn.LayerNorm(img_dim)

    def forward(self, h_meta: torch.Tensor, h_img: torch.Tensor) -> torch.Tensor:
        t = F.relu(self.fc1(h_meta), inplace=True)
        gamma, beta = self.fc2(t).chunk(2, dim=1)
        mod = (1.0 + gamma) * h_img + beta
        return self.norm(mod)


class MultiModalFiLMGated(nn.Module):
    """
    Inputs:
      x_img  : [B,3,H,W]
      x_meta : [B,7]
    Outputs:
      logits : [B,K]
      gates  : [B,K]
    """
    def __init__(
        self,
        num_labels: int = 8,
        dropout_img: float = 0.30,
        dropout_meta: float = 0.20,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_labels = int(num_labels)

        self.image_encoder = DenseNet121Backbone(dropout_p=dropout_img, pretrained=pretrained_backbone)
        self.meta_encoder = MetaMLP(in_dim=7, out_dim=128, dropout_p=dropout_meta)

        self.meta_to_img = nn.Linear(128, 1024)
        self.film = FiLM(meta_dim=128, img_dim=1024)

        self.gates = nn.ModuleList([nn.Linear(1152, 1) for _ in range(self.num_labels)])
        for g in self.gates:
            nn.init.xavier_uniform_(g.weight)
            nn.init.zeros_(g.bias)  # sigmoid(0)=0.5

        self.heads = nn.ModuleList([nn.Linear(1024, 1) for _ in range(self.num_labels)])

    def forward(self, x_img: torch.Tensor, x_meta7: torch.Tensor, mc_dropout: bool = False):
        x_meta7 = x_meta7.float()

        h_img = self.image_encoder(x_img, mc_dropout=mc_dropout)      # [B,1024]
        h_meta = self.meta_encoder(x_meta7, mc_dropout=mc_dropout)    # [B,128]

        h_img_f = self.film(h_meta, h_img)                            # [B,1024]
        h_gate = torch.cat([h_img_f, h_meta], dim=1)                  # [B,1152]
        meta_img = self.meta_to_img(h_meta)                           # [B,1024]

        logits, gates_out = [], []
        for k in range(self.num_labels):
            gk = torch.sigmoid(self.gates[k](h_gate))                 # [B,1]
            zk = gk * h_img_f + (1.0 - gk) * meta_img                 # [B,1024]
            logits.append(self.heads[k](zk))
            gates_out.append(gk)

        return torch.cat(logits, dim=1), torch.cat(gates_out, dim=1)

    @torch.no_grad()
    def mc_predict(self, x_img: torch.Tensor, x_meta7: torch.Tensor, T: int = 30):
        device = next(self.parameters()).device
        x_img = x_img.to(device, non_blocking=True)
        x_meta7 = x_meta7.to(device, non_blocking=True)

        self.eval()
        logits_list, gates_list = [], []
        for _ in range(int(T)):
            lg, gt = self.forward(x_img, x_meta7, mc_dropout=True)
            logits_list.append(lg)
            gates_list.append(gt)
        return torch.stack(logits_list, dim=0), torch.stack(gates_list, dim=0)
