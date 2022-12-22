import math
import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange

from .build import MODEL_REGISTRY
      
class LearnedPosMapT(nn.Module):
    def __init__(self, win_size,gamma=1):
        super().__init__()
        self.gamma = gamma
        self.win_size =win_size
        self.token_proj_n_bias = nn.Parameter(torch.zeros(1 ,self.win_size,1))
        self.rel_locl_init(self.win_size,register_name='window')
        self.init_bias_table()

    def rel_locl_init(self,win_size, register_name='window'):

        h= win_size
        w= win_size
        self.register_parameter(f'{register_name}_relative_position_bias_table' ,nn.Parameter(
            torch.zeros(2*h-1, self.gamma)))

        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        relative_coords = coords_h.unsqueeze(1) - coords_w.unsqueeze(0)  
        relative_coords = relative_coords.unsqueeze(0).permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += h - 1
        relative_position_index = relative_coords
        self.register_buffer(f"{register_name}_relative_position_index", relative_position_index)
 
    def init_bias_table(self):
        for k,v in self.named_modules():
            if 'relative_position_bias_table' in k:
                trunc_normal_(v.weight, std=.02) 

    def forward_pos(self):
        posmap = self.window_relative_position_bias_table[self.window_relative_position_index.view(-1)].view(
            self.win_size, self.win_size, -1) 
        posmap =  posmap.permute(2, 0, 1).contiguous()  #weizhi seq seq
        return posmap

    def forward(self,x,weight=None,mask=None):
        posmap = self.forward_pos()
        x = rearrange(x,'b w n (v s) -> b w n v s', s = self.gamma) 
        win_weight = rearrange(posmap,'(s b) m n  -> b m n s', s = self.gamma)  #m,b都是seq_len = win_size*wen_size

        x = torch.einsum('wmns,bwnvs->bwmvs',win_weight,x) + self.token_proj_n_bias.unsqueeze(0).unsqueeze(-1)
        x = rearrange(x,'b w n v s -> b w n (v s)') 
        
        return x

class PoSGUT(nn.Module):
    
    def __init__(self, dim, win_size,chunks=2, norm_layer=nn.LayerNorm,quadratic=True, gamma=16,pos_only=True):
        super().__init__()
        self.chunks = chunks
        self.gate_dim = dim // chunks
        self.pos=LearnedPosMapT(win_size,gamma=gamma)

    def forward(self, x,mask = None):
           # B W N C
        if self.chunks==1:
            u = x
            v = x
        else:
            x_chunks = x.chunk(2, dim=-1)
            u = x_chunks[0]
            v = x_chunks[1]
        u = self.pos(u)
        u = u * v
        return u

class PosMLPLayerT(nn.Module):
    def __init__(self, dim, win_size,shift = False, gate_unit=PoSGUT, #num_blocks=1,
    chunks = 2,  drop=0., act_layer=nn.GELU,norm_layer=nn.LayerNorm,gamma=8):
        super().__init__()  
        chunks = chunks  
        self.dim =dim
        self.norm = norm_layer(dim)
        self.gate_unit = gate_unit(dim, win_size = win_size,chunks=chunks, norm_layer=norm_layer,gamma=gamma)
        self.act = act_layer()
        self.fc2 = nn.Linear(dim,dim)
        self.fc1 = nn.Linear(dim , dim*2)
        self.drop = nn.Dropout(drop)

    def forward_gate(self,x,mask= None):
        x = self.act(self.fc1(self.norm(x)))
        # 1 c b n 
        x = self.gate_unit(x,mask=mask)
        x = self.drop(self.fc2(x))
        return x

        
    def forward_noshift(self,x):
                # Input : x (1,b,n,c)
        return self.forward_gate(x)

    def forward(self,x):
        x = self.forward_noshift(x)
        return x

class LearnedPosMap(nn.Module):
    def __init__(self, win_size,gamma=1):
        super().__init__()
        self.gamma = gamma
        self.win_size =win_size
        self.seq_len = win_size*win_size
        self.token_proj_n_bias = nn.Parameter(torch.zeros(1 ,self.seq_len,1))
        self.rel_locl_init(self.win_size,register_name='window')
        self.init_bias_table()

    def rel_locl_init(self,win_size, register_name='window'):

        h= win_size
        w= win_size
        self.register_parameter(f'{register_name}_relative_position_bias_table' ,nn.Parameter(
            torch.zeros((2 * h - 1) * (2 * w - 1), self.gamma)))

        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.  unsqueeze(1)  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += h - 1 
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1

        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer(f"{register_name}_relative_position_index", relative_position_index)
 
    def init_bias_table(self):
        for k,v in self.named_modules():
            if 'relative_position_bias_table' in k:
                trunc_normal_(v.weight, std=.02) 

    def forward_pos(self):
        posmap = self.window_relative_position_bias_table[self.window_relative_position_index.view(-1)].view(
            self.seq_len, self.seq_len, -1) 
        posmap =  posmap.permute(2, 0, 1).contiguous()
        return posmap

    def forward(self,x,weight=None,mask=None):
        posmap = self.forward_pos()
        x = rearrange(x,'b w n (v s) -> b w n v s', s = self.gamma)             #x是b,win_size,win_size,v,s
        win_weight = rearrange(posmap,'(s b) m n  -> b m n s', s = self.gamma)  #m,b都是seq_len = win_size*wen_size
        if weight is not None:
            win_weight = win_weight + weight.unsqueeze(-1)
        else:
            win_weight = win_weight
        x = torch.einsum('wmns,bwnvs->bwmvs',win_weight,x) + self.token_proj_n_bias.unsqueeze(0).unsqueeze(-1)
        x = rearrange(x,'b w n v s -> b w n (v s)') 
       
        return x

class PoSGU(nn.Module):
    
    def __init__(self, dim, win_size,chunks=2, norm_layer=nn.LayerNorm,quadratic=True, gamma=16,pos_only=True):
        super().__init__()
        self.chunks = chunks
        self.gate_dim = dim // chunks
        self.seq_len=win_size*win_size
        self.quadratic = quadratic
        self.pos_only=pos_only
        self.pos=LearnedPosMap(win_size,gamma=gamma)

        if not self.pos_only:
            self.token_proj_n_weight = nn.Parameter(torch.zeros(1, self.seq_len, self.seq_len))
            trunc_normal_(self.token_proj_n_weight,std=1e-6)

    def forward(self, x,mask = None):
        if self.chunks==1:
            u = x
            v = x
        else:
            x_chunks = x.chunk(2, dim=-1)
            u = x_chunks[0]
            v = x_chunks[1]
        if not self.pos_only and not self.quadratic:
            u = self.pos(u,self.token_proj_n_weight)
        else:
            u =self.pos(u,mask=mask)
        
        u = u * v
        return u

class PosMLPLayer(nn.Module):
    def __init__(self, dim, win_size,shift = False, gate_unit=PoSGU, num_blocks=1,
    chunks = 2, drop=0., act_layer=nn.GELU,norm_layer=nn.LayerNorm,gamma=8):
        super().__init__()  
        chunks = chunks  
        shift_size = 0
        self.shift = shift
        # if shift and num_blocks != 1 and layer_idx %2 ==1:
        #     shift_size = win_size//2
        #     _logger.info(f"shifted layer {layer_idx}")
        self.shift_size = shift_size
        self.dim =dim
        self.norm = norm_layer(dim)
        self.window_size = win_size
        self.gate_unit = gate_unit(dim, win_size = win_size,chunks=chunks, norm_layer=norm_layer,gamma=gamma)
        self.act = act_layer()
        self.fc2 = nn.Linear(dim , dim)
        self.fc1 = nn.Linear(dim , dim*2)
        self.drop = nn.Dropout(drop)

    def forward_gate(self,x,mask= None):
        x = self.act(self.fc1(self.norm(x)))
        # 1 c b n 
        x = self.gate_unit(x,mask=mask)
        x = self.drop(self.fc2(x))

        return x
        
    def forward_noshift(self,x):
                # Input : x (1,b,n,c)
        return self.forward_gate(x)

    def forward_shift(self,x):
        # @swin transformer
        # @http://arxiv.org/abs/2103.14030

        _, H, W, C = x.size()

        if self.shift_size > 0:
            # print("shift,here")
            shifted_x= torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # 1,nW, window_size*window_size, 1
            mask_windows = self.window_partition(img_mask)
            attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-200.0)).masked_fill(
                                                  attn_mask == 0, float(0.0)).squeeze()
        else:
            shifted_x = x
            attn_mask =None
        # _,NM,N,C
        shifted_x = window_partition(shifted_x,self.window_size,self.window_size)   
        out = self.forward_gate(shifted_x,mask = attn_mask)
        shifted_x = window_reverse(out, H, W)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x
        return x

    def forward(self,x):
        if self.shift:
            x = self.forward_shift(x)
        else:
            x = self.forward_noshift(x)
        return x

def window_reverse(x, H, W):

    B,_,N,C=x.size()
    Wh= int(math.sqrt(N))
    Ww = Wh
    x= x.view(B, -1, Wh,  Ww, C)   
    x = x.view(B, H // Wh, W // Ww, Wh, Ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_partition( x,Wh,Ww):

    B, H, W, C = x.shape
    x = x.view(B, H //Wh, Wh, W // Ww,Ww, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(B,-1, Wh*Ww, C)
    return windows

class PosMLPLevelS(nn.Module):
    """ Single hierarchical level of a Nested Transformer
    """
    def __init__(
            self,gate_unit, win_size,  embed_dim, gate_layer=PosMLPLayer,
            drop_rate=0., gamma = 8,
            norm_layer=None, act_layer=None,  shift = False):

        super().__init__()
        self.win_size=win_size
        self.shift = shift
        self.encoder = gate_layer(dim= embed_dim,win_size=win_size,gate_unit=gate_unit,
                                drop = drop_rate,shift =shift ,gamma = gamma,
                                norm_layer=norm_layer,act_layer=act_layer)
    def forward(self, x):
        """
        expects x as (1, C, H, W)
        """
        B,T,H,W,C = x.shape
        bt = B*T
        x = x.reshape(bt,H,W,C) #BT,H,W,Cs
        if self.shift :
            x = self.encoder(x)  # (1, H, W, C)
        else:
            # M,H,W,C = x.shape
            x = window_partition(x,self.win_size,self.win_size)  # (1, T, N, C)
            x = self.encoder(x)  # (1, T, N, C)
            x = window_reverse(x, H, W)  # (1, H, W, C)
        x = x.reshape(B,T,H,W,C)
        return x

class PosMLPLevelT(nn.Module):
    """ Single hierarchical level of a Nested Transformer
    """
    def __init__(
            self,gate_unit, win_size_t, embed_dim, gate_layer=PosMLPLayerT,
             drop_rate=0., gamma = 8,
            norm_layer=None, act_layer=None,  shift = False):

        super().__init__()
        self.win_size_t=win_size_t
        self.shift = shift
        self.encoder = gate_layer(dim= embed_dim,win_size=win_size_t,gate_unit=gate_unit,
                                 drop = drop_rate,shift = shift ,gamma = gamma,
                                 norm_layer=norm_layer,act_layer=act_layer)
    def forward(self, x):
        """
        expects x as (B,T,H,W,C)
        """
        B, T, H, W, C = x.shape
        hw = H*W
        x = x.reshape(B,T,hw,C).permute(0,2,1,3)
        x = self.encoder(x)
        x = x.permute(0,2,1,3).reshape(B,T,H,W,C)
        return x


def conv_3xnxn(inp, oup, kernel_size=3, stride=3):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 1, 1))


def conv_1xnxn(inp, oup, kernel_size=3, stride=3):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 1, 1))


class PermutatorBlock(nn.Module):
    def __init__(self, dim, 
                 drop_path=0.,act_layer=None, norm_layer=None, gate_unit_t = PoSGUT,
                 gate_unit=PoSGU, win_size=None,win_size_t = None, gate_layer=PosMLPLayer,gate_layer_t = PosMLPLayerT,
                 drop_rate=0., gamma = 1,
                 shift = False):
        super().__init__()

        self.win_size=win_size
        self.shift = shift
        self.t_fc = PosMLPLevelT(gate_unit_t,win_size_t,embed_dim=dim, gate_layer=gate_layer_t,gamma=gamma,
                                drop_rate=drop_rate, norm_layer=norm_layer, act_layer=act_layer)

        self.fc = PosMLPLevelS(gate_unit,win_size, embed_dim=dim,gate_layer=gate_layer,gamma=gamma,shift = self.shift,
                         drop_rate=drop_rate, norm_layer=norm_layer, act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.fc(x) + self.t_fc(x))

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj1 = conv_3xnxn(in_chans, embed_dim//2, kernel_size=3, stride=2)
        self.norm1= nn.BatchNorm3d(embed_dim//2)
        self.act=nn.GELU()
        self.proj2 = conv_1xnxn(embed_dim//2, embed_dim, kernel_size=3, stride=2)
        self.norm2 = nn.BatchNorm3d(embed_dim)

    def forward(self, x):
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.norm2(x)
        return x

class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = conv_1xnxn(in_embed_dim, out_embed_dim, kernel_size=3, stride=2)
        self.norm=nn.LayerNorm(out_embed_dim)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x)  # B, C, T, H, W
        x = x.permute(0, 2, 3, 4, 1)
        x=self.norm(x)
        return x

@MODEL_REGISTRY.register()
class PosMLP_Video(nn.Module):
    """ PosMLP_Video
    """

    def __init__(self, cfg):
        super().__init__()

        win_size = cfg.WIN_SIZE
        win_size_t = cfg.WIN_SIZE_T
        gamma = cfg.POSGAMMA
        shift = cfg.POSMLP_VIDEO.SHIFT
        num_classes = cfg.MODEL.NUM_CLASSES
        img_size = cfg.DATA.TRAIN_CROP_SIZE
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        layers = cfg.POSMLP_VIDEO.LAYERS
        transitions = cfg.POSMLP_VIDEO.TRANSITIONS
        embed_dims = cfg.POSMLP_VIDEO.EMBED_DIMS
        patch_size = cfg.POSMLP_VIDEO.PATCH_SIZE
        drop_path_rate = cfg.POSMLP_VIDEO.DROP_DEPTH_RATE
        norm_layer = nn.LayerNorm

        self.num_classes = num_classes

        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        # stage1
        self.blocks1 = nn.ModuleList([])
        for i in range(layers[0]):
            self.blocks1.append(
                PermutatorBlock(embed_dims[0], drop_path=dpr[i],shift = shift,gamma = gamma[0],
                                act_layer=nn.GELU,norm_layer=nn.LayerNorm, win_size=win_size[0], win_size_t=win_size_t[0],
                                )
            )
        if transitions[0] or embed_dims[0] != embed_dims[1]:
            patch_size = 2 if transitions[0] else 1
            self.patch_embed2 = Downsample(embed_dims[0], embed_dims[1], patch_size)
        else:
            self.patch_embed2 = nn.Identity()
        # stage2
        self.blocks2 = nn.ModuleList([])
        for i in range(layers[1]):
            self.blocks2.append(
                PermutatorBlock(embed_dims[1], drop_path=dpr[i + layers[0]], shift = shift,gamma = gamma[1],
                                 act_layer=nn.GELU,norm_layer=nn.LayerNorm, win_size=win_size[1], win_size_t=win_size_t[1],
                                 )
            )
        if transitions[1] or embed_dims[1] != embed_dims[2]:
            patch_size = 2 if transitions[1] else 1
            self.patch_embed3 = Downsample(embed_dims[1], embed_dims[2], patch_size)
        else:
            self.patch_embed3 = nn.Identity()
        # stage3
        self.blocks3 = nn.ModuleList([])
        for i in range(layers[2]):
            self.blocks3.append(
               PermutatorBlock(embed_dims[2], 
                                drop_path=dpr[i + layers[0] + layers[1]],
                                shift = shift,
                                act_layer=nn.GELU,
                                norm_layer=nn.LayerNorm, 
                                win_size=win_size[2], 
                                win_size_t=win_size_t[2],
                                gamma = gamma[2]
                                )
            )
        if transitions[2] or embed_dims[2] != embed_dims[3]:
            patch_size = 2 if transitions[2] else 1
            self.patch_embed4 = Downsample(embed_dims[2], embed_dims[3], patch_size)
        else:
            self.patch_embed4 = nn.Identity()
        
        # stage4
        self.blocks4 = nn.ModuleList([])
        for i in range(layers[3]):
            self.blocks4.append(
                PermutatorBlock(embed_dims[3], gamma = gamma[3],drop_path=dpr[i + layers[0]+layers[1]+layers[2]], shift = shift,
                                 act_layer=nn.GELU,norm_layer=nn.LayerNorm, win_size=win_size[3], win_size_t=win_size_t[3],
                                 )
            )

        self.norm = norm_layer(embed_dims[-1])

        # Classifier head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def get_pretrained_model(self, cfg):
        if cfg.POSMLP_VIDEO.PRETRAIN_PATH:
            checkpoint = torch.load(cfg.POSMLP_VIDEO.PRETRAIN_PATH, map_location='cpu')
            if self.num_classes != 1000:
                del checkpoint['head.weight']
                del checkpoint['head.bias']
            return checkpoint
        else:
            return None

    def forward_features(self, x):
        x = self.patch_embed1(x[0])
        # B,C,T,H,W -> B,T,H,W,C
        x = x.permute(0, 2, 3,4, 1)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        B, T,H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        return self.head(x.mean(1))