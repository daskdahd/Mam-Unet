import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import resnet50
from nets.vgg import VGG16


from atention import CAA, EMA, EfficientAdditiveAttnetion, AFGCAttention, DualDomainSelectionMechanism, AttentionTSSA
from module.ECA import ECA_layer
from atention import C2f_IEL


from simplified_block import CAA_HSFPN


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, attention_type='none'):
        super(unetUp, self).__init__()
        
  
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        
 
        if in_size == 3072:  # up_concat4
            skip_channels = 1024  # feat4
            up_channels = 2048    # feat5_up
        elif in_size == 1024:  # up_concat3
            skip_channels = 512   # feat3
            up_channels = 512     # up4
        elif in_size == 512:   # up_concat2
            skip_channels = 256   # feat2
            up_channels = 256     # up3
        elif in_size == 192:   # up_concat1
            skip_channels = 64    # feat1
            up_channels = 128     # up2
        else:

            up_channels = out_size * 2
            skip_channels = in_size - up_channels
            
            if skip_channels <= 0:
                skip_channels = out_size
                up_channels = in_size - skip_channels
        
        if attention_type != 'none':
            self.attention = DecoderAttentionModule(
                skip_channels=skip_channels,
                up_channels=up_channels,
                attention_type=attention_type
            )
        else:
            self.attention = None
            
        print(f"🔧 unetUp模块: in_size={in_size}, out_size={out_size}, skip_channels={skip_channels}, up_channels={up_channels}, attention={attention_type}")

    def forward(self, inputs1, inputs2):

        

        up_feat = self.up(inputs2)
        

        if self.attention is not None:
            skip_enhanced, up_enhanced = self.attention(inputs1, up_feat)
        else:
            skip_enhanced = inputs1
            up_enhanced = up_feat
        

        outputs = torch.cat([skip_enhanced, up_enhanced], 1)
        

        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        
        return outputs


class EncoderDecoderBridge(nn.Module):

    
    def __init__(self, backbone='resnet50', use_caa_hsfpn=True):
        super(EncoderDecoderBridge, self).__init__()
        
        self.use_caa_hsfpn = use_caa_hsfpn
        self.backbone = backbone
        
        if use_caa_hsfpn:
            if backbone == 'resnet50':

                self.caa_hsfpn_feat1 = CAA_HSFPN(ch=64, flag=True)    #
                self.caa_hsfpn_feat2 = CAA_HSFPN(ch=256, flag=True)   # 
                self.caa_hsfpn_feat3 = CAA_HSFPN(ch=512, flag=True)   #
                self.caa_hsfpn_feat4 = CAA_HSFPN(ch=1024, flag=True)  # 
                self.caa_hsfpn_feat5 = CAA_HSFPN(ch=2048, flag=True)  # 
                
            elif backbone == 'vgg':
           
                self.caa_hsfpn_feat1 = CAA_HSFPN(ch=64, flag=True)
                self.caa_hsfpn_feat2 = CAA_HSFPN(ch=128, flag=True)
                self.caa_hsfpn_feat3 = CAA_HSFPN(ch=256, flag=True)
                self.caa_hsfpn_feat4 = CAA_HSFPN(ch=512, flag=True)
                self.caa_hsfpn_feat5 = CAA_HSFPN(ch=512, flag=True)
            
          
      
         
    
    def forward(self, encoder_features):

        feat1, feat2, feat3, feat4, feat5 = encoder_features
        
        if self.use_caa_hsfpn:
            
            feat1_enhanced = self.caa_hsfpn_feat1(feat1) 
            feat2_enhanced = self.caa_hsfpn_feat2(feat2)  
            feat3_enhanced = self.caa_hsfpn_feat3(feat3)  # 
            feat4_enhanced = self.caa_hsfpn_feat4(feat4)  # 
            feat5_enhanced = self.caa_hsfpn_feat5(feat5)  # 
            
            return [feat1_enhanced, feat2_enhanced, feat3_enhanced, feat4_enhanced, feat5_enhanced]
        else:
            
            return [feat1, feat2, feat3, feat4, feat5]


class Unet(nn.Module):
    def __init__(self, num_classes=9, pretrained=False, backbone='resnet50', 
                 attention_type='caa', layer_attentions=None, use_c2f_iel=True, 
                 use_caa_hsfpn=True, use_transmamba=False):  
        super(Unet, self).__init__()
        

        self.use_caa_hsfpn = use_caa_hsfpn
        self.use_c2f_iel = use_c2f_iel
        self.use_transmamba = use_transmamba
   
        if layer_attentions is None:
            self.layer_attentions = {
                'up_concat4': 'caa',
                'up_concat3': 'eca',
                'up_concat2': 'none',
                'up_concat1': 'none'
            }
        else:
            self.layer_attentions = {}
            for layer in ['up_concat4', 'up_concat3', 'up_concat2', 'up_concat1']:
                self.layer_attentions[layer] = layer_attentions.get(layer, attention_type)
        
        print(f"\n🔥 构建增强版多层注意力UNet:")
        print(f"   骨干网络: {backbone}")
        print(f"   CAA_HSFPN桥接: {'启用' if use_caa_hsfpn else '禁用'}")
        print(f"   C2f_IEL增强: {'启用' if use_c2f_iel else '禁用'}")
        print(f"   TransMamba处理: {'启用' if use_transmamba else '禁用'}")  # 🔥 新增
        print(f"   解码器注意力配置:")
        for layer, att_type in self.layer_attentions.items():
            print(f"     {layer}: {att_type}")
        
        # 选择backbone
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]  
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained, use_transmamba=use_transmamba)  
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        
        out_filters = [64, 128, 256, 512] 

      
        self.encoder_decoder_bridge = EncoderDecoderBridge(
            backbone=backbone, 
            use_caa_hsfpn=use_caa_hsfpn
        )

    
        self.up_concat4 = unetUp(
            in_filters[3], out_filters[3], 
            attention_type=self.layer_attentions['up_concat4']
        )
        self.up_concat3 = unetUp(
            in_filters[2], out_filters[2], 
            attention_type=self.layer_attentions['up_concat3']
        )
        self.up_concat2 = unetUp(
            in_filters[1], out_filters[1], 
            attention_type=self.layer_attentions['up_concat2']
        )
        self.up_concat1 = unetUp(
            in_filters[0], out_filters[0], 
            attention_type=self.layer_attentions['up_concat1']
        )

 
        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None


        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone


        if use_c2f_iel: 
            print("🔧 初始化C2f_IEL模块...")
            if backbone == "resnet50":

                self.c2f_iel_feat1 = C2f_IEL(c1=64, c2=64, n=1, shortcut=False, e=0.5)
                self.c2f_iel_feat2 = C2f_IEL(c1=256, c2=256, n=2, shortcut=False, e=0.5)  
                self.c2f_iel_feat3 = C2f_IEL(c1=512, c2=512, n=3, shortcut=False, e=0.5)
                self.c2f_iel_feat4 = C2f_IEL(c1=1024, c2=1024, n=4, shortcut=False, e=0.5)
            elif backbone == "vgg":
                self.c2f_iel_feat1 = C2f_IEL(c1=64, c2=64, n=1, shortcut=False, e=0.5)
                self.c2f_iel_feat2 = C2f_IEL(c1=128, c2=128, n=1, shortcut=False, e=0.5)  
                self.c2f_iel_feat3 = C2f_IEL(c1=256, c2=256, n=1, shortcut=False, e=0.5)
                self.c2f_iel_feat4 = C2f_IEL(c1=512, c2=512, n=1, shortcut=False, e=0.5)
            print("✅ C2f_IEL模块初始化完成")
        else:
            print("⚠️ C2f_IEL模块已禁用")
            # 🔥 设置为None避免调用错误
            self.c2f_iel_feat1 = None
            self.c2f_iel_feat2 = None
            self.c2f_iel_feat3 = None
            self.c2f_iel_feat4 = None
        
        self.use_c2f_iel = use_c2f_iel
    
    def forward(self, inputs):

        if self.backbone == "vgg":
            encoder_features = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            encoder_features = self.resnet.forward(inputs)


        enhanced_features = self.encoder_decoder_bridge(encoder_features)
        feat1_bridge, feat2_bridge, feat3_bridge, feat4_bridge, feat5_bridge = enhanced_features


        if self.use_c2f_iel:
            feat1_enhanced = self.c2f_iel_feat1(feat1_bridge)  
            feat2_enhanced = self.c2f_iel_feat2(feat2_bridge) 
            feat3_enhanced = self.c2f_iel_feat3(feat3_bridge) 
            feat4_enhanced = self.c2f_iel_feat4(feat4_bridge)  
            feat5_final = feat5_bridge  
        else:

            feat1_enhanced = feat1_bridge
            feat2_enhanced = feat2_bridge
            feat3_enhanced = feat3_bridge
            feat4_enhanced = feat4_bridge
            feat5_final = feat5_bridge

        up4 = self.up_concat4(feat4_enhanced, feat5_final)  
        up3 = self.up_concat3(feat3_enhanced, up4)         
        up2 = self.up_concat2(feat2_enhanced, up3)         
        up1 = self.up_concat1(feat1_enhanced, up2)        

    
        if self.up_conv != None:
            up1 = self.up_conv(up1)

   
        final = self.final(up1)
        
        return final

 
    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

 
    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
    
    def get_attention_summary(self):

        summary = self.layer_attentions.copy()
        summary['caa_hsfpn_bridge'] = hasattr(self, 'encoder_decoder_bridge') and self.encoder_decoder_bridge.use_caa_hsfpn
        summary['c2f_iel_enhancement'] = self.use_c2f_iel
        return summary


class DecoderAttentionModule(nn.Module):

    
    def __init__(self, skip_channels, up_channels, attention_type='caa'):
        super(DecoderAttentionModule, self).__init__()
        
        self.attention_type = attention_type
        self.skip_channels = skip_channels
        self.up_channels = up_channels
        
        if attention_type == 'caa':
            self.skip_attention = CAA(ch=skip_channels)
            self.up_attention = CAA(ch=up_channels)
            
        elif attention_type == 'eca':
            self.skip_attention = ECA_layer(skip_channels)
            self.up_attention = ECA_layer(up_channels)
            
        elif attention_type == 'ema':
            self.skip_attention = EMA(channels=skip_channels)
            self.up_attention = EMA(channels=up_channels)
            
        elif attention_type == 'spatial':
            self.skip_spatial_att = nn.Sequential(
                nn.Conv2d(skip_channels, max(skip_channels//8, 1), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(skip_channels//8, 1), 1, 1),
                nn.Sigmoid()
            )
            self.up_spatial_att = nn.Sequential(
                nn.Conv2d(up_channels, max(up_channels//8, 1), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(up_channels//8, 1), 1, 1),
                nn.Sigmoid()
            )
            
        elif attention_type == 'channel':
            self.skip_channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(skip_channels, max(skip_channels//16, 1), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(skip_channels//16, 1), skip_channels, 1),
                nn.Sigmoid()
            )
            self.up_channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(up_channels, max(up_channels//16, 1), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(up_channels//16, 1), up_channels, 1),
                nn.Sigmoid()
            )
            
        elif attention_type == 'none':
            pass
            
        else:
            print(f"⚠️ 警告: 不支持的注意力类型 '{attention_type}'，将使用无注意力模式")
            self.attention_type = 'none'
        
        if attention_type != 'none':
            print(f"✅ 解码器注意力模块: {attention_type} (skip:{skip_channels}, up:{up_channels})")
    
    def forward(self, skip_feat, up_feat):
        if self.attention_type == 'caa':
            skip_enhanced = self.skip_attention(skip_feat)
            up_enhanced = self.up_attention(up_feat)
            
        elif self.attention_type == 'eca':
            skip_enhanced = self.skip_attention(skip_feat)
            up_enhanced = self.up_attention(up_feat)
            
        elif self.attention_type == 'ema':
            skip_enhanced = self.skip_attention(skip_feat)
            up_enhanced = self.up_attention(up_feat)
            
        elif self.attention_type == 'spatial':
            skip_att = self.skip_spatial_att(skip_feat)
            up_att = self.up_spatial_att(up_feat)
            skip_enhanced = skip_feat * skip_att
            up_enhanced = up_feat * up_att
            
        elif self.attention_type == 'channel':
            skip_att = self.skip_channel_att(skip_feat)
            up_att = self.up_channel_att(up_feat)
            skip_enhanced = skip_feat * skip_att
            up_enhanced = up_feat * up_att
            
        else:
            skip_enhanced = skip_feat
            up_enhanced = up_feat
        
        return skip_enhanced, up_enhanced
