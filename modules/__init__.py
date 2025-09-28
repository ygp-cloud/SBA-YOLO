# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3,
                    PatchEmbed, PatchMerging, SwinStage,
                    SPPFCSPC,
                    BEM, BFM,
                    C2f_PConv, C2f_GSConv, GSConv)
from .conv import (Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus, GhostConv, LightConv, RepConv,
                   SimFusion_4in, SimFusion_3in, IFM, InjectionMultiSum_Auto_pool, PyramidPoolAgg, AdvPoolFusion, TopBasicLayer,
                   BasicRFB,
                   ChannelAttention,  SpatialAttention, SE,  CBAM, CoordAtt,  GAM_Attention, ECA, EMA, MAM)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'Concat', 'TransformerLayer', 'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock',
           'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck',
           'BottleneckCSP', 'Proto', 'Detect', 'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'MLP',
           'RTDETRDecoder', 'AIFI', 'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn',

           'PatchEmbed', 'PatchMerging', 'SwinStage',
           'SimFusion_4in', 'SimFusion_3in', 'IFM', 'InjectionMultiSum_Auto_pool', 'PyramidPoolAgg', 'AdvPoolFusion','TopBasicLayer',
           'BasicRFB',

           'SPPFCSPC',
           'BEM', 'BFM',
           'ChannelAttention', 'SpatialAttention', 'SE',  'CBAM', 'CoordAtt', 'GAM_Attention', 'ECA', 'EMA', 'MAM',

           'C2f_PConv', 'C2f_GSConv', 'GSConv')
