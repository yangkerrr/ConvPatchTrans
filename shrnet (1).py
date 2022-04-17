# -*- encoding: utf-8 -*-
"""Networks based on vgg"""

import paddle
from paddle.fluid.initializer import NormalInitializer
import paddle.nn as nn
import paddle.nn.functional as F
import copy
from paddle.vision.models import vgg16
from network.transformer import Block,ConvTransBlock
import numpy as np
trunc_normal_ = nn.initializer.TruncatedNormal(std=.02)
zeros_ = nn.initializer.Constant(value=0.)
ones_ = nn.initializer.Constant(value=1.)
kaiming_normal_ = nn.initializer.KaimingNormal()

def param_init(net, param_file):
    # use pretrained model to the new model initialization
    param_pre = paddle.load(param_file)
    param_new = net.state_dict()
    for key in param_new.keys():
        if key in param_pre.keys() and param_new[key].size() == param_pre[key].size():
            param_new[key] = param_pre[key]
    return param_new
class TimeDistributed(nn.Layer):
  def __init__(self, module, timesteps = 4):
      super(TimeDistributed, self).__init__()
      self.module = module
      self.timesteps = timesteps

  def forward(self, input_seq):

      reshaped_input = input_seq.reshape(shape = [-1, input_seq.shape[-3],input_seq.shape[-2],input_seq.shape[-1]])

      output = self.module(reshaped_input)
      # We have to reshape Y
      output = output.reshape(shape = [-1,self.timesteps,output.shape[-3],output.shape[-2],output.shape[-1]])
      return output
class attention(nn.Layer):
    def __init__(self, feature_size=512, add_scaling_factor=True, return_attention_weights= False):
        super(attention, self).__init__()
        self.att_len = nn.Linear(feature_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)
        self.add_scaling_factor = add_scaling_factor
        self.return_attention_weights = return_attention_weights
        self.atten = []

    def forward(self, patches):
        #patch-> (N, num_patches, feature_size)
        atten_weights = self.att_len(patches).squeeze(2)#(N,num_patches)
        softmax_weights = self.softmax(atten_weights)#(N,num_patches)
        if (self.add_scaling_factor):
            softmax_weights /= np.sqrt(512) 
        #self.atten = np.append(self.atten, torch.argmax(softmax_weights, 1).cpu().data.numpy())
        patch_attention_encoding = (patches * softmax_weights.unsqueeze(2)).sum(1)#(N, feature_size)
        if (self.return_attention_weights):
            return patch_attention_encoding, softmax_weights
        return patch_attention_encoding

class SHR(nn.Layer):
    def __init__(self, num_classes):
        super(SHR, self).__init__()
        features = vgg16(pretrained=True).features

        self.quan = features[:21]
        self.features = features[:24]   # [512, 4, w/16], shared part
        # GS
        self.embedding = features[24:]   # [512, 2, w/32]


        self.classifier = nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(512, num_classes))
        # PA

        patch_size=16
        trans_dw_stride = patch_size // 8
        self.cls_token = self.create_parameter(shape=[1,1,768],default_initializer=nn.initializer.Constant(value = 0.0))
        self.trans_patch_conv = nn.Conv2D(512, 768, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)

        self.trans_1 = Block(dim = 768,num_heads=6, mlp_ratio=4, qkv_bias= True,
                             qk_scale=None, drop=0., attn_drop=0)

        self.convtrans1 = ConvTransBlock(512, 512, False, 1, dw_stride=trans_dw_stride, embed_dim=768,
                        num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                        drop_rate=0.2, attn_drop_rate=0,
                        num_med_block=0)

        self.convtrans2 = ConvTransBlock(512, 512, False, 1, dw_stride=trans_dw_stride, embed_dim=768,
                        num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                        drop_rate=0.2, attn_drop_rate=0,
                        num_med_block=0)
        
        self.convtrans3 = ConvTransBlock(512, 512, False, 1, dw_stride=trans_dw_stride, embed_dim=768,
                        num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                        drop_rate=0.2, attn_drop_rate=0,
                        num_med_block=0)

        self.trans_norm = nn.LayerNorm(768)
        self.trans_cls_head = nn.Linear(768, num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2D(1)
        self.conv_cls_head = nn.Linear(512, num_classes)

        self.trans_norm
        self.special = nn.Sequential(
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2D(512),
            nn.ReLU(),
        )   # [512, 2, w']
        self.cls1 = nn.Sequential(
             nn.Conv2D(512, 128, kernel_size=1, bias_attr=False),
             nn.BatchNorm2D(128),
             nn.ReLU(),
             nn.Conv2D(128, num_classes, kernel_size=1))
        self.cls2 = nn.Sequential(nn.Linear(num_classes, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, num_classes)
                                  )
        self.fusion = nn.Sequential(nn.Conv2D(512, 256, kernel_size=1, bias_attr=False),
                                    nn.BatchNorm2D(256),
                                    nn.ReLU(),
                                    nn.Conv2D(256, 1, kernel_size=1),
                                    nn.BatchNorm2D(1),
                                    nn.AdaptiveAvgPool2D((1, 1)),
                                    nn.Sigmoid(),
                                    )

    def forward(self, x):

        x = self.features(x)
        x_g = self.embedding(x)

        b = x_g.shape[0]


        out1 = self.classifier(F.adaptive_avg_pool2d(x_g, output_size=[1, 1]).reshape([b, -1])) 
        
        # L

        cls_tokens = self.cls_token.expand(shape = [b, -1, -1])

        x_t = self.trans_patch_conv(x).flatten(2).transpose(perm = [0, 2, 1])


        x_t = paddle.concat([cls_tokens, x_t], axis=1)

        x_t = self.trans_1(x_t)

        x_c,x_t = self.convtrans1(x,x_t)
        x_c,x_t = self.convtrans2(x_c,x_t)
        x_c,x_t = self.convtrans3(x_c,x_t)


        x_c = self.pooling(x_c).flatten(1)
        conv_cls = self.conv_cls_head(x_c)

        x_t = self.trans_norm(x_t)
        out3 = self.trans_cls_head(x_t[:, 0])     

        # Dynamic weighting
        weight = self.fusion(x_g).reshape(shape = [b, -1])    # [b, 1]
        #weight = paddle.full(shape=[b,1],dtype='float32', fill_value= 0.5)
        weight = paddle.concat([weight, 1-weight], axis=1)    # [b, 2]
        y_all = paddle.stack([out1,out3],axis=2)
        out = y_all.matmul(weight.unsqueeze(2)).squeeze(2)

        if self.training:
            return conv_cls, out1, out3, out
        else:
            return out


if __name__ == "__main__":
    net = SHR(7)
    print(net)
    im = paddle.randn(shape=[1, 3, 64, 128])
    out = net(im)
    print(out[0].shape)
