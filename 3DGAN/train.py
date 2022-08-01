# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
import copy
import torch
import time
import kornia
import torch.optim as optim
from lib.model.multiView_AutoEncoder import ResUNet2
import kornia.enhance.normalize as normalize
import torch.nn.functional as F



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
  parse.add_argument('--valid_dataset', type=str, default=None, dest='valid_dataset',
                     help='Train or test or valid')
  parse.add_argument('--datasetfile', type=str, default='', dest='datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--valid_datasetfile', type=str, default='', dest='valid_datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--ymlpath', type=str, default=None, dest='ymlpath',
                     help='config have been modified')
  parse.add_argument('--gpu', type=str, default='0,1', dest='gpuid',
                     help='gpu is split by ,')
  parse.add_argument('--dataset_class', type=str, default='align', dest='dataset_class',
                     help='Dataset class should select from align /')
  parse.add_argument('--model_class', type=str, default='simpleGan', dest='model_class',
                     help='Model class should select from simpleGan / ')
  parse.add_argument('--check_point', type=str, default=None, dest='check_point',
                     help='which epoch to load? ')
  parse.add_argument('--load_path', type=str, default=None, dest='load_path',
                     help='if load_path is not None, model will load from load_path')
  parse.add_argument('--latest', action='store_true', dest='latest',
                     help='set to latest to use latest cached model')
  parse.add_argument('--verbose', action='store_true', dest='verbose',
                     help='if specified, print more debugging information')
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


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  args = parse_args()
  args.useConnectionModules = str2bool(args.useConnectionModules)
  args.useConstFeatureMaps = str2bool(args.useConstFeatureMaps)

  # check gpu
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
    args.epoch_count = int(args.check_point) + 1

  # merge config with yaml
  if args.ymlpath is not None:
    cfg_from_yaml(args.ymlpath)
  # merge config with argparse
  opt = copy.deepcopy(cfg)
  opt = merge_dict_and_yaml(args.__dict__, opt)
  print_easy_dict(opt)

  # add data_augmentation
  datasetClass, augmentationClass, dataTestClass, collateClass = get_dataset(opt.dataset_class)
  opt.data_augmentation = augmentationClass

  # valid dataset
  if args.valid_dataset is not None:
    valid_opt = copy.deepcopy(opt)
    valid_opt.data_augmentation = dataTestClass
    valid_opt.datasetfile = opt.valid_datasetfile


    valid_dataset = datasetClass(valid_opt)
    print('Valid DataSet is {}'.format(valid_dataset.name))
    valid_dataloader = torch.utils.data.DataLoader(
      valid_dataset,
      batch_size=1,
      shuffle=False,
      num_workers=int(valid_opt.nThreads),
      collate_fn=collateClass)
    valid_dataset_size = len(valid_dataloader)
    print('#validation images = %d' % valid_dataset_size)
  else:
    valid_dataloader = None

  # get dataset
  batch_size = 2
  dataset = datasetClass(opt)
  print('DataSet is {}'.format(dataset.name))
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,#opt.batch_size
    shuffle=True,
    num_workers=int(opt.nThreads),
    collate_fn=collateClass)

  dataset_size = len(dataloader)
  print('#training images = %d' % dataset_size)

  # get model
  gan_model = get_model(opt.model_class)()
  print('Model --{}-- will be Used'.format(gan_model.name))
  gan_model.init_process(opt)
  total_steps, epoch_count = gan_model.setup(opt)

  # set to train
  gan_model.train()

  # visualizer
  from lib.utils.visualizer import Visualizer
  visualizer = Visualizer(log_dir=os.path.join(gan_model.save_root, 'train_log'))

  total_steps = total_steps

  # train discriminator more
  dataloader_iter_for_discriminator = iter(dataloader)
  feature_map_path = os.path.join(opt.MODEL_SAVE_PATH,"feature_map")

  template_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"data", "template-data","models","template.pt"))
  template = torch.load(template_path).to(device)
       
  template = torch.unsqueeze(template,dim=0)
  template = torch.unsqueeze(template,dim=0)

  enc_fmaps = []
  dec_fmaps = []
  temp_efmaps = []
  temp_dfmaps = []
  print("opt.useConstFeatureMaps")
  print(opt.useConstFeatureMaps)
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
  #0.0005  conv to 122
  lr = 0.005 #May need to be lower than during pretraining 
  init_loss = 1000000
  loss_res= 0
  alpha = 0.5
  gamma = 0.3
  loss = kornia.losses.MS_SSIMLoss().to(device)
  feature_extractor_path = os.path.join(opt.MODEL_SAVE_PATH,"feature_extractor.pt")

  feature_extractor = ResUNet2(in_channel=1,out_channel=1, training=True, out_fmap = False).to(device)
  
  

  feature_extractor.load_state_dict(torch.load(feature_extractor_path))
  for name, param in feature_extractor.named_parameters():
    if param.requires_grad and "down_conv" in name:
        param.requires_grad = False
    if param.requires_grad and "encoder_stage" in name:
        param.requires_grad = False
    if param.requires_grad and "batch_norm" in name:
        param.requires_grad = False

  feature_extractor2 =ResUNet2(in_channel=1,out_channel=1, training=False, out_fmap = True).to(device)
  feature_extractor2.load_state_dict(feature_extractor.state_dict())

  feature_extractor.train()

  valid_GAN = True
 
  

  optimizer = optim.Adam(feature_extractor.parameters(),lr)
  #scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma, verbose=True)
  init_dict = feature_extractor.state_dict()

  
  
  # train
  for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    iter_data_time = time.time()

      



    for epoch_i, data in enumerate(dataloader):
      
     
      if not opt.useConstFeatureMaps:
         for b_i in range(data[0].size()[0]):



          loss_res = 0
          init_loss = 1000000
          feature_extractor = ResUNet2(in_channel=1,out_channel=1, training=True, out_fmap = False).to(device)
          feature_extractor.load_state_dict(torch.load(feature_extractor_path))
          optimizer = optim.Adam(feature_extractor.parameters(),0.005) #0.005 is lr
          #scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma, verbose=True)
          recurrences = 0
          feature_extractor.train()

          while recurrences <= 13: #25 recurrences
          # if  (abs(init_loss - loss_res)<= 1.2 and recurrences >= 7) or (loss_res > init_loss and (abs(init_loss - loss_res) >= 1.2) and recurrences >=6)  :
            #  break
            if recurrences > 0 and recurrences % 6 == 0:
              print("rec: {},init_loss: {}, loss_res: {}".format(recurrences,init_loss,loss_res, abs(init_loss - loss_res)))
            optimizer.zero_grad()
            x_ray_S= torch.unsqueeze(data[1][1][b_i],dim=0).to(device)
           
            
            x_ray_F = torch.unsqueeze(data[1][0][b_i],dim=0).to(device)
            
            #temp  = template.repeat(data[0].size()[0],1,1,1,1).to(device)
          
            predicts = feature_extractor(template)
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
            if b_i == 0:
              res, temp_efmaps, temp_dfmaps = feature_extractor2(template)
            else:
              res, a, b = feature_extractor2(template)
              for j in range(5):
                temp_efmaps[j] = torch.concat((temp_efmaps[j],a[j]),dim=0)
              for j in range(5):
                temp_dfmaps[j] = torch.concat((temp_dfmaps[j],b[j]),dim=0)



         
          
        
            


       


            
            
        


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

      
      
      iter_start_time = time.time()

      total_steps += 1
    
    
      gan_model.set_input(input=data,enc_fmaps=temp_efmaps,dec_fmaps=temp_dfmaps)
      t0 = time.time()
      gan_model.optimize_parameters()
      t1 = time.time()

      

      # if total_steps == 1:
      #   visualizer.add_graph(model=gan_model, input=gan_model.forward())

      # # visual gradient
      # if opt.verbose and total_steps % opt.print_freq == 0:
      #   for name, para in gan_model.named_parameters():
      #     visualizer.add_histogram('Grad_' + name, para.grad.data.clone().cpu().numpy(), step=total_steps)
      #     visualizer.add_histogram('Weight_' + name, para.data.clone().cpu().numpy(), step=total_steps)
      #   for name in gan_model.model_names:
      #     net = getattr(gan_model, 'net' + name)
      #     if hasattr(net, 'output_dict'):
      #       for name, out in net.output_dict.items():
      #         visualizer.add_histogram(name, out.numpy(), step=total_steps)

      # loss
      loss_dict = gan_model.get_current_losses()
      # visualizer.add_scalars('Train_Loss', loss_dict, step=total_steps)
      total_loss = visualizer.add_total_scalar('Total loss', loss_dict, step=total_steps)
      # visualizer.add_average_scalers('Epoch Loss', loss_dict, step=total_steps, write=False)
      # visualizer.add_average_scalar('Epoch total Loss', total_loss)

      # metrics
      # metrics_dict = gan_model.get_current_metrics()
      # visualizer.add_scalars('Train_Metrics', metrics_dict, step=total_steps)
      # visualizer.add_average_scalers('Epoch Metrics', metrics_dict, step=total_steps, write=False)
     
      if total_steps % opt.print_freq == 0:
        print('total step: {} timer: {:.4f} sec.'.format(total_steps, t1 - t0))
        print('epoch {}/{}, step{}:{} || total loss:{:.4f}'.format(epoch, opt.niter + opt.niter_decay,
                                                                   epoch_i, dataset_size, total_loss))
        print('||'.join(['{}: {:.4f}'.format(k, v) for k, v in loss_dict.items()]))
        # print('||'.join(['{}: {:.4f}'.format(k, v) for k, v in metrics_dict.items()]))
        print('')

      # if total_steps % opt.print_img_freq == 0:
      #   visualizer.add_image('Image', gan_model.get_current_visuals(), gan_model.get_normalization_list(), total_steps)

      '''
      WGAN
      '''
      if (opt.critic_times - 1) > 0:
        for critic_i in range(opt.critic_times - 1):
          try:
            data = next(dataloader_iter_for_discriminator)
            gan_model.set_input(data)
            gan_model.optimize_D()
          except:
            dataloader_iter_for_discriminator = iter(dataloader)
      del(loss_dict)
      

    # # save model every epoch
    # print('saving the latest model (epoch %d, total_steps %d)' %
    #       (epoch, total_steps))
    # gan_model.save_networks(epoch, total_steps, True)

    # save model several epoch
    if epoch % opt.save_epoch_freq == 0 and epoch >= opt.begin_save_epoch:
      print('saving the model at the end of epoch %d, iters %d' %
            (epoch, total_steps))
      gan_model.save_networks(epoch, total_steps)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    ##########
    # For speed
    ##########
    # visualizer.add_image('Image_Epoch', gan_model.get_current_visuals(), gan_model.get_normalization_list(), epoch)
    # visualizer.add_average_scalers('Epoch Loss', None, step=epoch, write=True)
    # visualizer.add_average_scalar('Epoch total Loss', None, step=epoch, write=True)

    # visualizer.add_average_scalers('Epoch Metrics', None, step=epoch, write=True)

    # visualizer.add_scalar('Learning rate', gan_model.optimizers[0].param_groups[0]['lr'], epoch)
    gan_model.update_learning_rate(epoch)

    # # Test
    
    args.valid_dataset = None
    if args.valid_dataset is not None:
      
      if epoch % 10 == 0 and epoch > 1:
        gan_model.eval()
        iter_valid_dataloader = iter(valid_dataloader)
        for v_i in range(len(valid_dataloader)):

        
          data = next(iter_valid_dataloader)
          

          gan_model.set_input(input=data,enc_fmaps=temp_efmaps,dec_fmaps=temp_dfmaps)
          gan_model.test()
  
          if v_i < opt.howmany_in_train:
            visualizer.add_image('Test_Image', gan_model.get_current_visuals(), gan_model.get_normalization_list(), epoch*10+v_i, max_image=25)
  
          # metrics
          metrics_dict = gan_model.get_current_metrics()
          
          visualizer.add_average_scalers('Epoch Test_Metrics', metrics_dict, step=total_steps, write=False)
        visualizer.add_average_scalers('Epoch Test_Metrics', None, step=epoch, write=True)
  
        gan_model.train()