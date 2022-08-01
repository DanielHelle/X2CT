# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import functools
from lib.model.nets.generator.encoder_decoder_utils import *


def UNetLike_DownStep5(input_shape, encoder_input_channels, decoder_output_channels, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out=False):
  # 64, 32, 16, 8, 4
  encoder_block_list = [6, 12, 24, 16, 6]
  decoder_block_list = [1, 2, 2, 2, 2, 0]
  growth_rate = 32
  encoder_channel_list = [64]
  decoder_channel_list = [16, 16, 32, 64, 128, 256]
  decoder_begin_size = input_shape // pow(2, len(encoder_block_list))
  return UNetLike_DenseDimensionNet(encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out)

def UNetLike_DownStep5_3(input_shape, encoder_input_channels, decoder_output_channels, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out=False):
  # 64, 32, 16, 8, 4
  encoder_block_list = [6, 12, 32, 32, 12]
  decoder_block_list = [3, 3, 3, 3, 3, 1]
  growth_rate = 32
  encoder_channel_list = [64]
  decoder_channel_list = [16, 32, 64, 64, 128, 256]
  decoder_begin_size = input_shape // pow(2, len(encoder_block_list))
  return UNetLike_DenseDimensionNet(encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out)

class UNetLike_DenseDimensionNet(nn.Module):
  def __init__(self, encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer=nn.BatchNorm2d, decoder_norm_layer=nn.BatchNorm3d, upsample_mode='nearest', decoder_feature_out=False):
    super(UNetLike_DenseDimensionNet, self).__init__()

    self.decoder_channel_list = decoder_channel_list
    self.decoder_block_list = decoder_block_list
    self.n_downsampling = len(encoder_block_list)
    self.decoder_begin_size = decoder_begin_size
    self.decoder_feature_out = decoder_feature_out

    activation = nn.ReLU(True)
    bn_size = 4

    ##############
    # Encoder
    ##############
    if type(encoder_norm_layer) == functools.partial:
      use_bias = encoder_norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = encoder_norm_layer == nn.InstanceNorm2d

    encoder_layers0 = [
      nn.ReflectionPad2d(3),
      nn.Conv2d(encoder_input_channels, encoder_channel_list[0], kernel_size=7, padding=0, bias=use_bias),
      encoder_norm_layer(encoder_channel_list[0]),
      activation
    ]
    self.encoder_layer = nn.Sequential(*encoder_layers0)

    num_input_channels = encoder_channel_list[0]
    for index, channel in enumerate(encoder_block_list):
      # pooling
      down_layers = [
        encoder_norm_layer(num_input_channels),
        activation,
        nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, stride=2, padding=1, bias=use_bias),

      ]
      down_layers += [
        Dense_2DBlock(encoder_block_list[index], num_input_channels, bn_size, growth_rate, encoder_norm_layer, activation, use_bias),
      ]
      num_input_channels = num_input_channels + encoder_block_list[index] * growth_rate

      # feature maps are compressed into 1 after the lastest downsample layers
      if index == (self.n_downsampling-1):
        down_layers += [
          nn.AdaptiveAvgPool2d(1)
        ]
      else:
        num_out_channels = num_input_channels // 2
        down_layers += [
          encoder_norm_layer(num_input_channels),
          activation,
          nn.Conv2d(num_input_channels, num_out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]
        num_input_channels = num_out_channels
      encoder_channel_list.append(num_input_channels)
      setattr(self, 'encoder_layer' + str(index), nn.Sequential(*down_layers))

    ##############
    # Linker
    ##############
    if type(decoder_norm_layer) == functools.partial:
      use_bias = decoder_norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = decoder_norm_layer == nn.InstanceNorm3d

    # linker FC
    # apply fc to link 2d and 3d
    self.base_link = nn.Sequential(*[
      nn.Linear(encoder_channel_list[-1], decoder_begin_size**3*decoder_channel_list[-1]),
      nn.Dropout(0.5),
      activation
    ])

    for index, channel in enumerate(encoder_channel_list[:-1]):
      in_channels = channel
      out_channels = decoder_channel_list[index]
      link_layers = [
        Dimension_UpsampleCutBlock(in_channels, out_channels, encoder_norm_layer, decoder_norm_layer, activation, use_bias)
      ]
      setattr(self, 'linker_layer' + str(index), nn.Sequential(*link_layers))

    ##############
    # Decoder
    ##############
    for index, channel in enumerate(decoder_channel_list[:-1]):
      out_channels = channel
      in_channels = decoder_channel_list[index+1]
      decoder_layers = []
      decoder_compress_layers = []
      if index != (len(decoder_channel_list) - 2):
        decoder_compress_layers += [
          nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=use_bias),
          decoder_norm_layer(in_channels),
          activation
        ]
        for _ in range(decoder_block_list[index+1]):
          decoder_layers += [
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=use_bias),
            decoder_norm_layer(in_channels),
            activation
          ]
      decoder_layers += [
        Upsample_3DUnit(3, in_channels, out_channels, decoder_norm_layer, scale_factor=2, upsample_mode=upsample_mode, activation=activation, use_bias=use_bias)
      ]
      # If decoder_feature_out is True, compressed feature after upsampling and concatenation
      # can be obtained.
      if decoder_feature_out:
        setattr(self, 'decoder_compress_layer' + str(index), nn.Sequential(*decoder_compress_layers))
        setattr(self, 'decoder_layer' + str(index), nn.Sequential(*decoder_layers))
      else:
        setattr(self, 'decoder_layer' + str(index), nn.Sequential(*(decoder_compress_layers+decoder_layers)))
    # last decode
    decoder_layers = []
    decoder_compress_layers = [
      nn.Conv3d(decoder_channel_list[0] * 2, decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
      decoder_norm_layer(decoder_channel_list[0]),
      activation
    ]
    for _ in range(decoder_block_list[0]):
      decoder_layers += [
        nn.Conv3d(decoder_channel_list[0], decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
        decoder_norm_layer(decoder_channel_list[0]),
        activation
      ]
    if decoder_feature_out:
      setattr(self, 'decoder_compress_layer' + str(-1), nn.Sequential(*decoder_compress_layers))
      setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*decoder_layers))
    else:
      setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*(decoder_compress_layers + decoder_layers)))

    self.decoder_layer = nn.Sequential(*[
      nn.Conv3d(decoder_channel_list[0], decoder_output_channels, kernel_size=7, padding=3, bias=use_bias),
      decoder_out_activation()
    ])

  def forward(self, input):
    encoder_feature = self.encoder_layer(input)
    next_input = encoder_feature
    for i in range(self.n_downsampling):
      setattr(self, 'feature_linker' + str(i), getattr(self, 'linker_layer' + str(i))(next_input))
      next_input = getattr(self, 'encoder_layer'+str(i))(next_input)

    next_input = self.base_link(next_input.view(next_input.size(0), -1))
    next_input = next_input.view(next_input.size(0), self.decoder_channel_list[-1], self.decoder_begin_size, self.decoder_begin_size, self.decoder_begin_size)

    for i in range(self.n_downsampling - 1, -2, -1):
      if i == (self.n_downsampling - 1):
        if self.decoder_feature_out:
          next_input = getattr(self, 'decoder_layer' + str(i))(getattr(self, 'decoder_compress_layer' + str(i))(next_input))
        else:
          next_input = getattr(self, 'decoder_layer' + str(i))(next_input)

      else:
        if self.decoder_feature_out:
          next_input = getattr(self, 'decoder_layer' + str(i))(getattr(self, 'decoder_compress_layer' + str(i))(torch.cat((next_input, getattr(self, 'feature_linker'+str(i+1))), dim=1)))
        else:
          next_input = getattr(self, 'decoder_layer' + str(i))(torch.cat((next_input, getattr(self, 'feature_linker'+str(i+1))), dim=1))

    return self.decoder_layer(next_input)


class MultiView_UNetLike_DenseDimensionNet(nn.Module):
  def __init__(self, view1Model, view2Model, view1Order, view2Order, backToSub, decoder_output_channels, decoder_out_activation, decoder_block_list=None, decoder_norm_layer=nn.BatchNorm3d, upsample_mode='nearest',useConnectionModules=True):
    super(MultiView_UNetLike_DenseDimensionNet, self).__init__()
    #Could have to pass enc and dec fmaps to forward
    self.useConnectionModules = useConnectionModules
    self.view1Model = view1Model
    self.view2Model = view2Model
    self.view1Order = view1Order
    self.view2Order = view2Order
    self.backToSub = backToSub
    self.n_downsampling = view2Model.n_downsampling
    self.decoder_channel_list = view2Model.decoder_channel_list
    if decoder_block_list is None:
      self.decoder_block_list = view2Model.decoder_block_list
    else:
      self.decoder_block_list = decoder_block_list

    activation = nn.ReLU(True)
    if type(decoder_norm_layer) == functools.partial:
      use_bias = decoder_norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = decoder_norm_layer == nn.InstanceNorm3d

    self.batch_norm_enc1 = nn.BatchNorm3d(16)
    self.batch_norm_enc2 = nn.BatchNorm3d(32)
    self.batch_norm_enc3 = nn.BatchNorm3d(64)
    self.batch_norm_enc4 = nn.BatchNorm3d(128)
    self.batch_norm_enc5 = nn.BatchNorm3d(256)

    self.batch_norm_dec1 = nn.BatchNorm3d(256)
    self.batch_norm_dec2 = nn.BatchNorm3d(256)
    self.batch_norm_dec3 = nn.BatchNorm3d(128)
    self.batch_norm_dec4 = nn.BatchNorm3d(64)
    self.batch_norm_dec5 = nn.BatchNorm3d(32)


    self.fmap_fusion_layer0 = nn.Sequential(
      #enc1 channels + dec5 channels + next_input channels
      #conv so C is divided by 2 and then equal output channel
      nn.Conv3d(16+32+16,32,kernel_size=3,stride=1,padding=1),
      nn.PReLU(32),
      nn.BatchNorm3d(32),
      nn.Conv3d(32,32,kernel_size=3,stride=1,padding=1),
      nn.PReLU(32),
      nn.BatchNorm3d(32),
      nn.Conv3d(32,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      nn.BatchNorm3d(16),
      nn.Conv3d(16,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      
    )
    self.fmap_fusion_layer1 = nn.Sequential(
      #enc2 channels + dec4 channels + next_input channels
      nn.Conv3d(32+64+16,56,kernel_size=3,stride=1,padding=1),
      nn.PReLU(56),
      nn.BatchNorm3d(56),
      nn.Conv3d(56,56,kernel_size=3,stride=1,padding=1),
      nn.PReLU(56),
      nn.BatchNorm3d(56),
      nn.Conv3d(56,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      nn.BatchNorm3d(16),
      nn.Conv3d(16,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
    )

    self.fmap_fusion_layer2 = nn.Sequential(
      nn.Conv3d(64+128+32,112,kernel_size=3,stride=1,padding=1),
      nn.PReLU(112),
      nn.BatchNorm3d(112),
      nn.Conv3d(112,112,kernel_size=3,stride=1,padding=1),
      nn.PReLU(112),
      nn.BatchNorm3d(112),
      nn.Conv3d(112,32,kernel_size=3,stride=1,padding=1),
      nn.PReLU(32),
      nn.BatchNorm3d(32),
      nn.Conv3d(32,32,kernel_size=3,stride=1,padding=1),
      nn.PReLU(32),
    )

    self.fmap_fusion_layer3 = nn.Sequential(
      nn.Conv3d(128+256+64,224,kernel_size=3,stride=1,padding=1),
      nn.PReLU(224),
      nn.BatchNorm3d(224),
      nn.Conv3d(224,224,kernel_size=3,stride=1,padding=1),
      nn.PReLU(224),
      nn.BatchNorm3d(224),
      nn.Conv3d(224,64,kernel_size=3,stride=1,padding=1),
      nn.PReLU(64),
      nn.BatchNorm3d(64),
      nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
      nn.PReLU(64),
    )

    self.fmap_fusion_layer4 = nn.Sequential(
      nn.Conv3d(256+256+128,320,kernel_size=3,stride=1,padding=1),
      nn.PReLU(320),
      nn.BatchNorm3d(320),
      nn.Conv3d(320,320,kernel_size=3,stride=1,padding=1),
      nn.PReLU(320),
      nn.BatchNorm3d(320),
      nn.Conv3d(320,128,kernel_size=3,stride=1,padding=1),
      nn.PReLU(128),
      nn.BatchNorm3d(128),
      nn.Conv3d(128,128,kernel_size=3,stride=1,padding=1),
      nn.PReLU(128),

    )

    self.enc_fmap1_conv = nn.Sequential(
      nn.Conv3d(16,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      nn.BatchNorm3d(16),
      nn.Conv3d(16,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      nn.BatchNorm3d(16),
      nn.Conv3d(16,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      
    )
    self.enc_fmap2_conv = nn.Sequential(
      nn.Conv3d(32,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      nn.BatchNorm3d(16),
      nn.Conv3d(16,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      nn.BatchNorm3d(16),
      nn.Conv3d(16,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      
    )
    self.enc_fmap3_conv = nn.Sequential(
      nn.Conv3d(64,32,kernel_size=3,stride=1,padding=1),
      nn.PReLU(32),
      nn.BatchNorm3d(32),
      nn.Conv3d(32,32,kernel_size=3,stride=1,padding=1),
      nn.PReLU(32),
      nn.BatchNorm3d(32),
      nn.Conv3d(32,32,kernel_size=3,stride=1,padding=1),
      nn.PReLU(32),
      
    )
    self.enc_fmap4_conv = nn.Sequential(
      nn.Conv3d(128,64,kernel_size=3,stride=1,padding=1),
      nn.PReLU(64),
      nn.BatchNorm3d(64),
      nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
      nn.PReLU(64),
      nn.BatchNorm3d(64),
      nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
      nn.PReLU(64),
      
    )
    self.enc_fmap5_conv = nn.Sequential(
      nn.Conv3d(256,128,kernel_size=3,stride=1,padding=1),
      nn.PReLU(128),
      nn.BatchNorm3d(128),
      nn.Conv3d(128,128,kernel_size=3,stride=1,padding=1),
      nn.PReLU(128),
      nn.BatchNorm3d(128),
      nn.Conv3d(128,128,kernel_size=3,stride=1,padding=1),
      nn.PReLU(128),
    
    )
    self.dec_fmap1_conv = nn.Sequential(
      nn.Conv3d(256,128,kernel_size=3,stride=1,padding=1),
      nn.PReLU(128),
      nn.BatchNorm3d(128),
      nn.Conv3d(128,128,kernel_size=3,stride=1,padding=1),
      nn.PReLU(128),
      nn.BatchNorm3d(128),
      nn.Conv3d(128,128,kernel_size=3,stride=1,padding=1),
      nn.PReLU(128),
      
    )
    self.dec_fmap2_conv = nn.Sequential(
      nn.Conv3d(256,64,kernel_size=3,stride=1,padding=1),
      nn.PReLU(64),
      nn.BatchNorm3d(64),
      nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
      nn.PReLU(64),
      nn.BatchNorm3d(64),
      nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
      nn.PReLU(64),
      
    )
    self.dec_fmap3_conv = nn.Sequential(
      nn.Conv3d(128,32,kernel_size=3,stride=1,padding=1),
      nn.PReLU(32),
      nn.BatchNorm3d(32),
      nn.Conv3d(32,32,kernel_size=3,stride=1,padding=1),
      nn.PReLU(32),
      nn.BatchNorm3d(32),
      nn.Conv3d(32,32,kernel_size=3,stride=1,padding=1),
      nn.PReLU(32),
    )
    self.dec_fmap4_conv = nn.Sequential(
      nn.Conv3d(64,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      nn.BatchNorm3d(16),
      nn.Conv3d(16,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      nn.BatchNorm3d(16),
      nn.Conv3d(16,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
    )
    self.dec_fmap5_conv = nn.Sequential(
      nn.Conv3d(32,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      nn.BatchNorm3d(16),
      nn.Conv3d(16,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
      nn.BatchNorm3d(16),
      nn.Conv3d(16,16,kernel_size=3,stride=1,padding=1),
      nn.PReLU(16),
    )

    ##############
    # Decoder
    ##############
    for index, channel in enumerate(self.decoder_channel_list[:-1]):
      out_channels = channel
      in_channels = self.decoder_channel_list[index + 1]
      decoder_layers = []
      decoder_compress_layers = []
      if index != (len(self.decoder_channel_list) - 2):
        decoder_compress_layers += [
          nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=use_bias),
          decoder_norm_layer(in_channels),
          activation
        ]
        for _ in range(self.decoder_block_list[index+1]):
          decoder_layers += [
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=use_bias),
            decoder_norm_layer(in_channels),
            activation
          ]
      decoder_layers += [
        Upsample_3DUnit(3, in_channels, out_channels, decoder_norm_layer, scale_factor=2, upsample_mode=upsample_mode,
                        activation=activation, use_bias=use_bias)
      ]

      setattr(self, 'decoder_layer' + str(index), nn.Sequential(*(decoder_compress_layers + decoder_layers)))
    # last decode
    decoder_layers = []
    decoder_compress_layers = [
      nn.Conv3d(self.decoder_channel_list[0] * 2, self.decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
      decoder_norm_layer(self.decoder_channel_list[0]),
      activation
    ]
    for _ in range(self.decoder_block_list[0]):
      decoder_layers += [
        nn.Conv3d(self.decoder_channel_list[0], self.decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
        decoder_norm_layer(self.decoder_channel_list[0]),
        activation
      ]
    setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*(decoder_compress_layers + decoder_layers)))
    self.decoder_layer = nn.Sequential(*[
      nn.Conv3d(self.decoder_channel_list[0], decoder_output_channels, kernel_size=7, padding=3, bias=use_bias),
      decoder_out_activation()
    ])

    self.transposed_layer = Transposed_And_Add(view1Order, view2Order)

  
  def forward(self, input,enc_fmaps,dec_fmaps):
    # only support two views 
    assert len(input) == 2
    # View 1 encoding process
    view1_encoder_feature = self.view1Model.encoder_layer(input[0])
    view1_next_input = view1_encoder_feature
    for i in range(self.view1Model.n_downsampling):
      setattr(self.view1Model, 'feature_linker' + str(i), getattr(self.view1Model, 'linker_layer' + str(i))(view1_next_input))
      view1_next_input = getattr(self.view1Model, 'encoder_layer'+str(i))(view1_next_input)
    # View 2 encoding process
    view2_encoder_feature = self.view2Model.encoder_layer(input[1])
    view2_next_input = view2_encoder_feature
    for i in range(self.view2Model.n_downsampling):
      setattr(self.view2Model, 'feature_linker' + str(i),
              getattr(self.view2Model, 'linker_layer' + str(i))(view2_next_input))
      view2_next_input = getattr(self.view2Model, 'encoder_layer' + str(i))(view2_next_input)
    # View 1 decoding process Part1
    view1_next_input = self.view1Model.base_link(view1_next_input.view(view1_next_input.size(0), -1))
    view1_next_input = view1_next_input.view(view1_next_input.size(0), self.view1Model.decoder_channel_list[-1], self.view1Model.decoder_begin_size,
                                             self.view1Model.decoder_begin_size, self.view1Model.decoder_begin_size)
    # View 2 decoding process Part1
    view2_next_input = self.view2Model.base_link(view2_next_input.view(view2_next_input.size(0), -1))
    view2_next_input = view2_next_input.view(view2_next_input.size(0), self.view2Model.decoder_channel_list[-1], self.view2Model.decoder_begin_size,
                                             self.view2Model.decoder_begin_size, self.view2Model.decoder_begin_size)

    view_next_input = None
    # View 1 and 2 decoding process Part2
    for i in range(self.n_downsampling - 1, -2, -1):
      if i == (self.n_downsampling - 1):
        view1_next_input = getattr(self.view1Model, 'decoder_compress_layer' + str(i))(view1_next_input)
        view2_next_input = getattr(self.view2Model, 'decoder_compress_layer' + str(i))(view2_next_input)
        ########### MultiView Fusion
        # Method One: Fused feature back to sub-branch
        if self.backToSub:
          view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
          view1_next_input = view_avg.permute(*self.view1Order)
          view2_next_input = view_avg.permute(*self.view2Order)
          #print(" first IF i: {}, size: {}".format(i,view_avg.size()))
          #for j in range(self.n_downsampling - 1, -2, -1):
            #print(j)
          #print(" first IF i: {}, size: {}".format(i,view_avg.size()))
          view_next_input = getattr(self, 'decoder_layer' + str(i))(view_avg)
          """
          print("view_next_input: {}".format(view_next_input.size()))
          print("i: {}".format(i))
          
          print("enc_fmaps[{}]: {}".format(4,self.enc_fmap5_conv(enc_fmaps[4]).size()))
          print("dec_fmaps[{}]: {}".format(4-4,self.dec_fmap1_conv(dec_fmaps[4-4]).size()))

          print("enc_fmaps[{}]: {}".format(3,self.enc_fmap4_conv(enc_fmaps[3]).size()))
          print("dec_fmaps[{}]: {}".format(4-3,self.dec_fmap2_conv(dec_fmaps[4-3]).size()))

          print("enc_fmaps[{}]: {}".format(2,self.enc_fmap3_conv(enc_fmaps[2]).size()))
          print("dec_fmaps[{}]: {}".format(4-2,self.dec_fmap3_conv(dec_fmaps[4-2]).size()))

          print("enc_fmaps[{}]: {}".format(1,self.enc_fmap2_conv(enc_fmaps[1]).size()))
          print("dec_fmaps[{}]: {}".format(4-1,self.dec_fmap4_conv(dec_fmaps[4-1]).size()))

          print("enc_fmaps[{}]: {}".format(0,self.enc_fmap1_conv(enc_fmaps[0]).size()))
          print("dec_fmaps[{}]: {}".format(4-0,self.dec_fmap5_conv(dec_fmaps[4-0]).size()))
          """
          if self.useConnectionModules and  i >0: #should be i >-1

            #print(view_next_input.size())
            #print(enc_fmaps[i].size())
            #print(dec_fmaps[4-i].size())

            view_next_input = getattr(self, 'fmap_fusion_layer' + str(i)) (torch.cat((view_next_input,getattr(self,"batch_norm_enc"+str(i+1))(enc_fmaps[i]),getattr(self,"batch_norm_dec"+str(5-i))(dec_fmaps[4-i])), dim=1))


          """
          if self.useConnectionModules:
            if i == 0:
              view_next_input = view_next_input + self.enc_fmap1_conv(enc_fmaps[i]) + self.dec_fmap5_conv(dec_fmaps[4-i])
            elif i == 1:
              view_next_input = view_next_input + self.enc_fmap2_conv(enc_fmaps[i]) + self.dec_fmap4_conv(dec_fmaps[4-i])
            elif i == 2:
              view_next_input = view_next_input + self.enc_fmap3_conv(enc_fmaps[i]) + self.dec_fmap3_conv(dec_fmaps[4-i])
            elif i == 3:
              view_next_input = view_next_input + self.enc_fmap4_conv(enc_fmaps[i]) + self.dec_fmap2_conv(dec_fmaps[4-i])
            elif i == 4:
              view_next_input = view_next_input + self.enc_fmap5_conv(enc_fmaps[i]) + self.dec_fmap1_conv(dec_fmaps[4-i])

              """
          """
          else:
            if i == 0:
              view_avg = view_avg + enc_fmaps[i] + dec_fmaps[i]
            elif i == 1:
              view_avg = view_avg + enc_fmaps[i] + dec_fmaps[i]
            elif i == 2:
              view_avg = view_avg + enc_fmaps[i] + dec_fmaps[i]
            elif i == 3:
              view_avg = view_avg + enc_fmaps[i] + dec_fmaps[i]
            elif i == 4:
              view_avg = view_avg + enc_fmaps[i] + dec_fmaps[i]
          """   

          
          #print("\n first IF idx: {}, pass 1, shape: {} \n".format(i,view_next_input.size()))

        # Method Two: Fused feature only used in main-branch
        else:
          view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
          view_next_input = getattr(self, 'decoder_layer' + str(i))(view_avg)
        ###########
        view1_next_input = getattr(self.view1Model, 'decoder_layer' + str(i))(view1_next_input)
        view2_next_input = getattr(self.view2Model, 'decoder_layer' + str(i))(view2_next_input)
      else:
        view1_next_input = getattr(self.view1Model, 'decoder_compress_layer' + str(i))(torch.cat((view1_next_input, getattr(self.view1Model, 'feature_linker' + str(i + 1))), dim=1))
        view2_next_input = getattr(self.view2Model, 'decoder_compress_layer' + str(i))(torch.cat((view2_next_input, getattr(self.view2Model, 'feature_linker' + str(i + 1))), dim=1))
        ########### MultiView Fusion
        # Method One: Fused feature back to sub-branch
        if self.backToSub:
          view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
          view1_next_input = view_avg.permute(*self.view1Order)
          view2_next_input = view_avg.permute(*self.view2Order)
          #print("second IF i: {}, size: {}".format(i,view_avg.size()))
          view_next_input = getattr(self, 'decoder_layer' + str(i))(torch.cat((view_avg, view_next_input), dim=1))

          if self.useConnectionModules and i>0: #should be i>-1
    
            
            view_next_input = getattr(self, 'fmap_fusion_layer' + str(i)) (torch.cat((view_next_input,getattr(self,"batch_norm_enc"+str(i+1))(enc_fmaps[i]),getattr(self,"batch_norm_dec"+str(5-i))(dec_fmaps[4-i])), dim=1))
          """
          if self.useConnectionModules:
            if i == 0:
              view_next_input = view_next_input + self.enc_fmap1_conv(enc_fmaps[i]) + self.dec_fmap5_conv(dec_fmaps[4-i])
            elif i == 1:
              view_next_input = view_next_input + self.enc_fmap2_conv(enc_fmaps[i]) + self.dec_fmap4_conv(dec_fmaps[4-i])
            elif i == 2:
              view_next_input = view_next_input + self.enc_fmap3_conv(enc_fmaps[i]) + self.dec_fmap3_conv(dec_fmaps[4-i])
            elif i == 3:
              view_next_input = view_next_input + self.enc_fmap4_conv(enc_fmaps[i]) + self.dec_fmap2_conv(dec_fmaps[4-i])
            elif i == 4:
              view_next_input = view_next_input + self.enc_fmap5_conv(enc_fmaps[i]) + self.dec_fmap1_conv(dec_fmaps[4-i])
          """
          """
          else:
            if i == 0:
              view_next_input = view_next_input + self.enc_fmap1_conv(enc_fmaps[i]) + self.dec_fmap5_conv(dec_fmaps[4-i])
            elif i == 1:
              view_next_input = view_next_input + self.enc_fmap2_conv(enc_fmaps[i]) + self.dec_fmap4_conv(dec_fmaps[4-i])
            elif i == 2:
              view_next_input = view_next_input + self.enc_fmap3_conv(enc_fmaps[i]) + self.dec_fmap3_conv(dec_fmaps[4-i])
            elif i == 3:
              view_next_input = view_next_input + self.enc_fmap4_conv(enc_fmaps[i]) + self.dec_fmap2_conv(dec_fmaps[4-i])
            elif i == 4:
              view_next_input = view_next_input + self.enc_fmap5_conv(enc_fmaps[i]) + self.dec_fmap1_conv(dec_fmaps[4-i])
          """
          #print("\n second IF idx: {}, pass 2, shape: {} \n".format(i,view_next_input.size()))
        # Method Two: Fused feature only used in main-branch
        else:
          view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
          view_next_input = getattr(self, 'decoder_layer' + str(i))(torch.cat((view_avg, view_next_input), dim=1))
        ###########
        view1_next_input = getattr(self.view1Model, 'decoder_layer' + str(i))(view1_next_input)
        view2_next_input = getattr(self.view2Model, 'decoder_layer' + str(i))(view2_next_input)


    return self.view1Model.decoder_layer(view1_next_input), self.view2Model.decoder_layer(view2_next_input), self.decoder_layer(view_next_input)