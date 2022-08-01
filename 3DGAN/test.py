# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------
import os

from sqlalchemy import false 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
from lib.utils.visualizer import tensor_back_to_unnormalization, tensor_back_to_unMinMax
from lib.utils.metrics_np import MAE, MSE, Peak_Signal_to_Noise_Rate, Structural_Similarity, Cosine_Similarity, \
  Peak_Signal_to_Noise_Rate_3D
import copy
import tqdm
import torch
import numpy as np
from logger import Logger
from lib.model.multiView_AutoEncoder import ResUNet2
import torch.optim as optim
import kornia




def parse_args():
  parse = argparse.ArgumentParser(description='CTGAN')
  parse.add_argument('--data', type=str, default='', dest='data',
                     help='input data ')
  parse.add_argument('--tag', type=str, default='', dest='tag',
                     help='distinct from other try')
  parse.add_argument('--dataroot', type=str, default='', dest='dataroot',
                     help='input data root')
  parse.add_argument('--dataset', type=str, default='', dest='dataset',
                     help='Train or test or valid')
  parse.add_argument('--datasetfile', type=str, default='', dest='datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--ymlpath', type=str, default=None, dest='ymlpath',
                     help='config have been modified')
  parse.add_argument('--gpu', type=str, default='0,1', dest='gpuid',
                     help='gpu is split by ,')
  parse.add_argument('--dataset_class', type=str, default='unalign', dest='dataset_class',
                     help='Dataset class should select from unalign /')
  parse.add_argument('--model_class', type=str, default='cyclegan', dest='model_class',
                     help='Model class should select from cyclegan / ')
  parse.add_argument('--check_point', type=str, default=None, dest='check_point',
                     help='which epoch to load? ')
  parse.add_argument('--latest', action='store_true', dest='latest',
                     help='set to latest to use latest cached model')
  parse.add_argument('--verbose', action='store_true', dest='verbose',
                     help='if specified, print more debugging information')
  parse.add_argument('--load_path', type=str, default=None, dest='load_path',
                     help='if load_path is not None, model will load from load_path')
  parse.add_argument('--how_many', type=int, dest='how_many', default=50,
                     help='if specified, only run this number of test samples for visualization')
  parse.add_argument('--resultdir', type=str, default='', dest='resultdir',
                     help='dir to save result')
  parse.add_argument('--useConnectionModules',default="1", dest='useConnectionModules', type=str ,
                      help='Select if you want to conv skip connections from feature_extractor to X2CT')
  parse.add_argument('--useConstFeatureMaps',default="0", dest='useConstFeatureMaps', type=str ,
                      help='Select if you want to you constant feature maps')
  args = parse.parse_args()
  return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# map function
def output_map(v, dim):
    '''
    :param v: tensor
    :param dim:  dimension be reduced
    :return:
      N1HW
    '''
    ori_dim = v.dim()
    # tensor [NDHW]
    if ori_dim == 4:
      map = torch.mean(torch.abs(v), dim=dim)
      # [NHW] => [NCHW]
      return map.unsqueeze(1)
    # tensor [NCDHW] and c==1
    elif ori_dim == 5:
      # [NCHW]
      map = torch.mean(torch.abs(v), dim=dim)
      return map
    else:
      raise NotImplementedError()

def transition(predict):
    p_max, p_min = predict.max(), predict.min()
    new_predict = (predict - p_min) / (p_max - p_min)
    return new_predict

def ct_unGaussian(opt, value):
    return value * opt.CT_MEAN_STD[1] + opt.CT_MEAN_STD[0]


def projection_visual(opt,pred):
    # map F is projected in dimension of H
    x_ray_fake_F = transition(output_map(ct_unGaussian(opt,pred), 2))
    #map S is projected in dimension of W
    x_ray_fake_S = transition(output_map(ct_unGaussian(opt,pred), 3))
    return x_ray_fake_F, x_ray_fake_S

def evaluate(args):
  # check gpu

  args.useConnectionModules = str2bool(args.useConnectionModules)
  args.useConstFeatureMaps = str2bool(args.useConstFeatureMaps)

  if args.gpuid == '':
    args.gpu_ids = []
  else:
    if torch.cuda.is_available():
      split_gpu = str(args.gpuid).split(',')
      args.gpu_ids = [int(i) for i in split_gpu]
    else:
      print('There is no gpu!')
      exit(0)

  # check point
  if args.check_point is None:
    args.epoch_count = 1
  else:
    args.epoch_count = int(args.check_point)

  # merge config with yaml
  if args.ymlpath is not None:
    cfg_from_yaml(args.ymlpath)
  # merge config with argparse
  opt = copy.deepcopy(cfg)
  opt = merge_dict_and_yaml(args.__dict__, opt)
  print_easy_dict(opt)

  opt.serial_batches = True

  # add data_augmentation
  datasetClass, _, dataTestClass, collateClass = get_dataset(opt.dataset_class)
  opt.data_augmentation = dataTestClass

  # get dataset
  dataset = datasetClass(opt)
  
  print('DataSet is {}'.format(dataset.name))
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=int(opt.nThreads),
    collate_fn=collateClass)

  dataset_size = len(dataloader)
  print('#Test images = %d' % dataset_size)

  # get model
  gan_model = get_model(opt.model_class)()
  print('Model --{}-- will be Used'.format(gan_model.name))

  # set to test
  gan_model.eval()

  gan_model.init_process(opt)
  total_steps, epoch_count = gan_model.setup(opt)

  # must set to test Mode again, due to  omission of assigning mode to network layers
  # model.training is test, but BN.training is training
  if opt.verbose:
    print('## Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)

  if 'batch' in opt.norm_G:
    gan_model.eval()
  elif 'instance' in opt.norm_G:
    gan_model.eval()
    # instance norm in training mode is better
    for name, m in gan_model.named_modules():
      if m.__class__.__name__.startswith('InstanceNorm'):
        m.train()
  else:
    raise NotImplementedError()

  if opt.verbose:
    print('## Change to Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)

  result_dir = os.path.join(opt.resultdir, opt.data, '%s_%s' % (opt.dataset, opt.check_point))
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  template_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"data", "template-data","models","template.pt"))
  template = torch.load(template_path).to(device)
       
  template = torch.unsqueeze(template,dim=0)
  template = torch.unsqueeze(template,dim=0)
  feature_extractor_path = os.path.join(opt.MODEL_SAVE_PATH,"feature_extractor.pt")
        
  feature_extractor = ResUNet2(in_channel=1,out_channel=1, training=True, out_fmap = False).to(device)
  feature_extractor.load_state_dict(torch.load(feature_extractor_path))
  #feature_extractor2 uses recurrently finetuned weights
  feature_extractor2 = ResUNet2(in_channel=1,out_channel=1, training=False, out_fmap = True).to(device)
  #Probably unecessary to freeze encoder again
  for name, param in feature_extractor.named_parameters():
    if param.requires_grad and "down_conv" in name:
        param.requires_grad = False
    if param.requires_grad and "encoder_stage" in name:
        param.requires_grad = False
    if param.requires_grad and "batch_norm" in name:
        param.requires_grad = False

  feature_map_path = os.path.join(opt.MODEL_SAVE_PATH,"feature_map")

  enc_fmaps = []
  dec_fmaps = []
  temp_efmaps = []
  temp_dfmaps = []

  if opt.useConstFeatureMaps:

    for i in range(5):
        enc_fmaps.append(torch.load(os.path.join(feature_map_path, "enc_fmap{}.pt".format(i+1))))
        #print(enc_fmaps[len(enc_fmaps)-1].size())
    for i in range(5):
        dec_fmaps.append(torch.load(os.path.join(feature_map_path, "dec_fmap{}.pt".format(i+1))))
        #print(dec_fmaps[len(dec_fmaps)-1].size())
    temp_efmaps = copy.deepcopy(enc_fmaps)
    temp_dfmaps = copy.deepcopy(dec_fmaps)


  recurrences = 0
  lr = 0.005 #May need to be lower than during pretraining
  init_loss = 1000000
  loss_res= 0
  alpha = 0.4
  gamma= 0.3
  loss = kornia.losses.MS_SSIMLoss().to(device)

  optimizer = optim.Adam(feature_extractor.parameters(),lr)
  #scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma, verbose=True)
  init_dict = feature_extractor.state_dict()

  avg_dict = dict()

  #FINETUNE ONLY WITH BATCH_SIZE = 1, we cannot update weights based on several ct scans
  for epoch_i, data in tqdm.tqdm(enumerate(dataloader)):
    #preform recurrent feature extraction
    if not opt.useConstFeatureMaps:
      loss_res = 0
      init_loss = 1000000
      recurrences = 0
      feature_extractor = ResUNet2(in_channel=1,out_channel=1, training=True, out_fmap = False).to(device)
      feature_extractor.load_state_dict(torch.load(feature_extractor_path))
      optimizer = optim.Adam(feature_extractor.parameters(),0.005) #0.005 is lr dont want to use pointer to lr
      scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma, verbose=True)
      feature_extractor.train()
      #reload weights
      
      while recurrences <= 100: #add or abs(init_loss - loss_res) < 0.0005 and
        if recurrences > 0 and recurrences  % 25 == 0: 
            print("rec: {},init_loss: {}, loss_res: {}".format(recurrences,init_loss,loss_res, abs(init_loss - loss_res)))
        if recurrences > 50 and recurrences < 100 and recurrences % 10 == 0:
          scheduler.step()
        optimizer.zero_grad()
        x_ray_S= data[1][1].to(device)
        x_ray_F = data[1][0].to(device)
        temp  = template.repeat(data[0].size()[0],1,1,1,1).to(device)
        predicts = feature_extractor(temp)
        pred1 = predicts[0]
        pred2 = predicts[1]
        pred3 = predicts[2]
        pred4 = predicts[3]
        pred1_F, pred1_S = projection_visual(opt,pred1) 
        pred2_F, pred2_S = projection_visual(opt,pred2)  
        pred3_F, pred3_S = projection_visual(opt,pred3) 
        pred4_F, pred4_S = projection_visual(opt,pred4) 
        pred1_F = pred1_F.to(device)
        pred1_S = pred1_S.to(device)
        pred2_F = pred2_F.to(device)
        pred2_S = pred2_S.to(device)
        pred3_F = pred3_F.to(device)
        pred3_S = pred3_S.to(device)
        pred4_F = pred4_F.to(device)
        pred4_S = pred4_S.to(device)
        cost1 = loss(pred1_F,x_ray_F)
        cost2 = loss(pred1_S,x_ray_S)
        loss1 = cost1 + cost2
        cost1 = loss(pred2_F,x_ray_F)
        cost2 = loss(pred2_S,x_ray_S)
        loss2 = cost1 + cost2
        cost1 = loss(pred3_F,x_ray_F)
        cost2 = loss(pred3_S,x_ray_S)
        loss3 = cost1 + cost2
        cost1 = loss(pred4_F,x_ray_F)
        cost2 = loss(pred4_S,x_ray_S)
        loss4 = cost1 + cost2
        loss_res = loss4 + (alpha *(loss1 + loss2 + loss3))
        if recurrences == 0:
          init_loss = loss_res
        loss_res.backward()
        optimizer.step()
        recurrences = recurrences + 1
        
      feature_extractor2.load_state_dict(feature_extractor.state_dict())
     
      feature_extractor2.eval()
      with torch.no_grad():
        res, temp_efmaps, temp_dfmaps = feature_extractor2(template)
    else:
    
      temp_efmaps[0]= enc_fmaps[0].repeat(data[0].size()[0],1,1,1,1)
      temp_efmaps[1]= enc_fmaps[1].repeat(data[0].size()[0],1,1,1,1)
      temp_efmaps[2]= enc_fmaps[2].repeat(data[0].size()[0],1,1,1,1)
      temp_efmaps[3]= enc_fmaps[3].repeat(data[0].size()[0],1,1,1,1)
      temp_efmaps[4]= enc_fmaps[4].repeat(data[0].size()[0],1,1,1,1)
      #print(data[0].size()[0])
      #print(enc_fmaps[len(enc_fmaps)-1].size())
      
      
      temp_dfmaps[0]= dec_fmaps[0].repeat(data[0].size()[0],1,1,1,1)
      temp_dfmaps[1]= dec_fmaps[1].repeat(data[0].size()[0],1,1,1,1)
      temp_dfmaps[2]= dec_fmaps[2].repeat(data[0].size()[0],1,1,1,1)
      temp_dfmaps[3]= dec_fmaps[3].repeat(data[0].size()[0],1,1,1,1)
      temp_dfmaps[4]= dec_fmaps[4].repeat(data[0].size()[0],1,1,1,1)

    gan_model.set_input(input=data,enc_fmaps=temp_efmaps,dec_fmaps=temp_dfmaps)
    gan_model.test()

    visuals = gan_model.get_current_visuals()
    img_path = gan_model.get_image_paths()

    #
    # Evaluate Part
    #
    generate_CT = visuals['G_fake'].data.clone().cpu().numpy()
    real_CT = visuals['G_real'].data.clone().cpu().numpy()
    # To [0, 1]
    # To NDHW
    if 'std' in opt.dataset_class or 'baseline' in opt.dataset_class:
      generate_CT_transpose = generate_CT
      real_CT_transpose = real_CT
    else:
      generate_CT_transpose = np.transpose(generate_CT, (0, 2, 1, 3))
      real_CT_transpose = np.transpose(real_CT, (0, 2, 1, 3))
    generate_CT_transpose = tensor_back_to_unnormalization(generate_CT_transpose, opt.CT_MEAN_STD[0],
                                                           opt.CT_MEAN_STD[1])
    real_CT_transpose = tensor_back_to_unnormalization(real_CT_transpose, opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1])
    # clip generate_CT
    generate_CT_transpose = np.clip(generate_CT_transpose, 0, 1)

    # CT range 0-1
    mae0 = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
    mse0 = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)
    cosinesimilarity = Cosine_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False)
    ssim = Structural_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=1.0)
    # CT range 0-4096
    generate_CT_transpose = tensor_back_to_unMinMax(generate_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(
      np.int32)
    real_CT_transpose = tensor_back_to_unMinMax(real_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(
      np.int32)
    psnr_3d = Peak_Signal_to_Noise_Rate_3D(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=4095)
    psnr = Peak_Signal_to_Noise_Rate(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=4095)
    mae = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
    mse = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)

    name1 = os.path.splitext(os.path.basename(img_path[0][0]))[0]
    name2 = os.path.split(os.path.dirname(img_path[0][0]))[-1]
    name = name2 + '_' + name1
    print(cosinesimilarity, name)
    if cosinesimilarity is np.nan or cosinesimilarity > 1:
      print(os.path.splitext(os.path.basename(gan_model.get_image_paths()[0][0]))[0])
      continue

    metrics_list = [('MAE0', mae0), ('MSE0', mse0), ('MAE', mae), ('MSE', mse), ('CosineSimilarity', cosinesimilarity),
                    ('psnr-3d', psnr_3d), ('PSNR-1', psnr[0]),
                    ('PSNR-2', psnr[1]), ('PSNR-3', psnr[2]), ('PSNR-avg', psnr[3]),
                    ('SSIM-1', ssim[0]), ('SSIM-2', ssim[1]), ('SSIM-3', ssim[2]), ('SSIM-avg', ssim[3])]

    for key, value in metrics_list:
      if avg_dict.get(key) is None:
        avg_dict[key] = [] + value.tolist()
      else:
        avg_dict[key].extend(value.tolist())

    del visuals, img_path

  for key, value in avg_dict.items():
    print('### --{}-- total: {}; avg: {} '.format(key, len(value), np.round(np.mean(value), 7)))
    avg_dict[key] = np.mean(value)

  return avg_dict


if __name__ == '__main__':
  args = parse_args()
  evaluate(args)
