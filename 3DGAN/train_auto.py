
import os
from sre_constants import JUMP

from tabnanny import verbose
from xmlrpc.client import Boolean 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
from lib.model.multiView_AutoEncoder import ResUNet
from lib.model.multiView_AutoEncoder import ResUNet2
from lib.model.multiView_AutoEncoder import ResUNet_Down
from lib.model.multiView_AutoEncoder import ResUNet_up
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import copy
import torch
import time
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    parse.add_argument('--resultdir', type=str, default='', dest='resultdir',
                     help='dir to save result')
    #parse.add_argument('--pretrain', action=argparse.BooleanOptionalAction)
    parse.add_argument('--pretrain',default="0", dest='pretrain', type=str ,
                        help='if specified, pretrains autoencoder. If not trains pretrained models')
    #parse.add_argument('--model_to_train',default="0", dest='model_to_train', type=str ,
     #                   help='select model to train, decodeCorrection or decodeGroundTruth')
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

if __name__ == '__main__':

    args = parse_args()

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
    

    opt.pretrain = str2bool(opt.pretrain)
    print("Pretraining: {} ".format(opt.pretrain))
    #if opt.model_to_train == "0" and opt.pretrain == False:
    #    print("Please select model to train: decodeCorrection or decodeGroundTruth")
    #    exit(0)
    #opt.model_to_train = opt.model_to_train.lower()

    

    
    datasetClass, augmentationClass, dataTestClass, collateClass = get_dataset(opt.dataset_class)
    opt.data_augmentation = augmentationClass
    dataset = datasetClass(opt)


    
    

    
    
    #print(feature_map_path)
    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = ResUNet2(in_channel=1,out_channel=1,training=True).to(device)

  

    #auto_down = ResUNet_Down(in_channel = 1, out_channel=256).to(device)
    
    #print(autoencoder)
    #print(auto_down)
    

    autoencoder.train()

    # set to train
    
    #for l1 loss: lr = 0.00075, alpha=0.4, batch_size=5, lr=lr=0.00075, gamma=0.6
    pretrain_auto = {}
    pretrain_auto["alpha"] = 0.4
    pretrain_auto["epoch"] =150
    #first batch size was 30
    pretrain_auto["batch-print"] = 35
    pretrain_auto["batch_size"] = 5
    #pretrain_auto["loss"] = torch.nn.L1Loss().to(device)
    pretrain_auto["loss"] = torch.nn.MSELoss(reduction='mean')

    pretrain_auto["optimizer"] = optim.Adam(autoencoder.parameters(),lr=0.000075)

    pretrain_auto["scheduler"] = optim.lr_scheduler.ExponentialLR(pretrain_auto["optimizer"],gamma=0.5, verbose=True)
    #pretrain_auto["scheduler"] = optim.lr_scheduler.LamdaLR(pretrain_auto["optimizer"],batch_learn)
    if opt.pretrain == True:
        print(autoencoder)

    print("figs path: {}".format(os.path.join(opt.MODEL_SAVE_PATH,"figs","autoencoder","train")))

    dataloader_auto = torch.utils.data.DataLoader(
        dataset,
        batch_size= pretrain_auto['batch_size'],
        shuffle=True,
        num_workers=int(opt.nThreads),
        collate_fn=collateClass)
    avg_losses = []
    lr_list = []
    predicts = None

   
    template = None
    pretrain_auto["running_loss"] = 0.0
    curr_lr = 0
    #if pretrian != None
    if opt.pretrain == True:
    #pretraining of autoencoder
        for epoch in range(pretrain_auto["epoch"]):
            pretrain_auto["running_loss"] = 0.0
            correct = 0
            for i, data in enumerate(dataloader_auto):
                X = data[0]
                    
                X = torch.unsqueeze(X,1)
                    
                X = X.to(device)

                pretrain_auto["optimizer"].zero_grad()
                
                predicts = autoencoder(X)

                loss0 = pretrain_auto["loss"](predicts[0],X)
                loss1 = pretrain_auto["loss"](predicts[1],X)
                loss2 = pretrain_auto["loss"](predicts[2],X)
                loss3 = pretrain_auto["loss"](predicts[3],X)
                #print("\n loss0: {}, loss1: {}, loss2: {}, loss3: {} \n".format(loss0.item(),loss1.item(),loss2.item(),loss3.item()))
                loss = loss3 + (pretrain_auto["alpha"] *(loss0 + loss1 + loss2))
                loss.backward()
                pretrain_auto["optimizer"].step()
                pretrain_auto["running_loss"] = pretrain_auto["running_loss"] + loss.item()
                if i % pretrain_auto["batch-print"] == 0:
                    print("\n Epoch: {}, Loss: {}, sample {}\n ".format(epoch, loss.item(),i*pretrain_auto['batch_size']))

            curr_lr = pretrain_auto["scheduler"].optimizer.param_groups[0]['lr']
            if epoch % 50 == 0 and epoch > 0 and epoch < pretrain_auto["epoch"]:
                pretrain_auto["scheduler"].step()
            avg_loss = pretrain_auto["running_loss"]/len(dataloader_auto)
            avg_losses.append(avg_loss)
            lr_list.append(curr_lr)

            

 
        

        
        
        #template = template.to(device)

        #template = torch.unsqueeze(template,dim=0)
        #template = torch.unsqueeze(template,dim=0)
        #keys = set(auto_down.state_dict().keys())
        #auto_down.load_state_dict({k:v for k,v in autoencoder.state_dict().items() if k in keys})
        #print(auto_down)
        #feature_map,long_range1, long_range2, long_range3, long_range4 = auto_down(template)
        #dim = feature_map.size()[2] 
        #print(feature_map.size())
        #print(dim)
        #file_path = os.path.join(feature_map_path,"feature_map.pt")
        #long_range1_path = os.path.join(feature_map_path,"long_range1.pt")
        ##long_range2_path = os.path.join(feature_map_path,"long_range2.pt")
        #long_range3_path = os.path.join(feature_map_path,"long_range3.pt")
        #long_range4_path = os.path.join(feature_map_path,"long_range4.pt")

        autoencoder_figs_path = os.path.join(opt.MODEL_SAVE_PATH,"figs","autoencoder","train")
        avg_loss_path = os.path.join(autoencoder_figs_path,"avg-loss.png")
        lr_path = os.path.join(autoencoder_figs_path,"lr.png")
        
        
        autoencoder_path = os.path.join(opt.MODEL_SAVE_PATH,"autoencoder.pt")
        #auto_down_path = os.path.join(opt.MODEL_SAVE_PATH,"auto_down.pt")
        """
        if os.path.isfile(file_path):
            os.remove(file_path)
        if os.path.isfile(long_range1_path):
            os.remove(long_range1_path)
        if os.path.isfile(long_range2_path):
            os.remove(long_range2_path)
        if os.path.isfile(long_range3_path):
            os.remove(long_range3_path)
        if os.path.isfile(long_range4_path):
            os.remove(long_range4_path)

        """
        if os.path.isfile(avg_loss_path):
            os.remove(avg_loss_path)
        if os.path.isfile(lr_path):
            os.remove(lr_path)
        if os.path.isfile(autoencoder_path):
            os.remove(autoencoder_path)
        #if os.path.isfile(auto_down_path):
            #os.remove(auto_down_path)
        

        

        

        #torch.save(feature_map,file_path)
        #torch.save(long_range1,long_range1_path)
        #torch.save(long_range2,long_range2_path)
        #torch.save(long_range3,long_range3_path)
        #torch.save(long_range4,long_range4_path)
        torch.save(autoencoder.state_dict(),autoencoder_path)
        #torch.save(auto_down.state_dict(),auto_down_path)

        fg_loss = Figure()
        ax_loss = fg_loss.gca()
        ax_loss.plot(avg_losses)
        ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_loss.set_xlabel('epochs', fontsize=10)
        ax_loss.set_ylabel('avg-loss', fontsize='medium')
        
        fg_loss.savefig(avg_loss_path)
        
        fg_lr = Figure()
        ax_lr = fg_lr.gca()
        ax_lr.plot(lr_list)
        ax_lr.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_lr.set_xlabel('epochs', fontsize=10)
        ax_lr.set_ylabel('lr', fontsize='medium')
        
        fg_lr.savefig(lr_path)
        
        exit()
    else:
        
        #create Template
        #auto_down_path = os.path.join(opt.MODEL_SAVE_PATH,"auto_down.pt")
        #auto_down = ResUNet_Down(in_channel = 1, out_channel=256).to(device)
        #auto_down_weights = torch.load(auto_down_path)
        #auto_down.load_state_dict(auto_down_weights)

        #feature_map = torch.load(feature_map_path +"\\feature_map.pt")
        #long_range1 = torch.load(feature_map_path +"\\long_range1.pt")
        #long_range2 = torch.load(feature_map_path +"\\long_range2.pt")
        #long_range3 = torch.load(feature_map_path +"\\long_range3.pt")
        #long_range4 = torch.load(feature_map_path +"\\long_range4.pt")
        #w = torch.tensor([0.5,0.5],requires_grad=False).to(device)
        autoencoder_path = os.path.join(opt.MODEL_SAVE_PATH,"autoencoder.pt")
        feature_extractor = ResUNet2(in_channel=1,out_channel=1, training=True).to(device)
        feature_extractor.load_state_dict(torch.load(autoencoder_path))
        #Freezes encoder weights
        

        for name, param in feature_extractor.named_parameters():
            if param.requires_grad and "down_conv" in name:
                param.requires_grad = False
            if param.requires_grad and "encoder_stage" in name:
                param.requires_grad = False
            if param.requires_grad and "batch_norm" in name:
                param.requires_grad = False

        #for name, param in feature_extractor.named_parameters():
        #    if param.requires_grad:print(name)

       
        
        template_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"data", "template-data","models","template.pt"))
        template = torch.load(template_path).to(device)
       
        template = torch.unsqueeze(template,dim=0)
        template = torch.unsqueeze(template,dim=0)
        #print(template.size())
        

       
            
        print(feature_extractor)
        
          
        """
        gan_model = get_model(opt.model_class)()
        print('Model --{}-- will be Used'.format(gan_model.name))

        

        # set to test
        gan_model.eval()

        gan_model.init_process(opt)
                #total_steps, epoch_count = gan_model.setup(opt)
                
                #New training loop for both X2CT and auto_up

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
        
        """
        avg_dict = dict()
        epochs = 210
        batch_size =6
        lr = 0.005
        gamma = 0.4
        alpha = 0.4
        running_loss = 0.0
        curr_lr = 0.0
        avg_losses = []
        lr_list = []
        batch_print = 35
        
        
        
        #loss = torch.nn.L1Loss().to(device)
        loss = kornia.losses.MS_SSIMLoss().to(device)
        optimizer = optim.Adam(feature_extractor.parameters(),lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma, verbose=True)
        
        dataloader_decoder = torch.utils.data.DataLoader(
        dataset,
        batch_size= batch_size,
        shuffle=True,
        num_workers=int(opt.nThreads),
        collate_fn=collateClass)

        #VALIDATION DATA
        valid_feature_extractor = True
        opt_valid = copy.deepcopy(opt)
        opt_valid.dataset = "test"
        opt_valid.datasetfile="./data/test.txt"
        batch_size_valid = 6
        valid_running_loss = 0.0
        valid_avg_loss = 0.0
        valid_avg_losses = []

        datasetClassValid, _, dataTestClassValid, collateClassValid = get_dataset(opt_valid.dataset_class)
        opt_valid.data_augmentation = dataTestClassValid
        dataset_valid = datasetClassValid(opt_valid)

        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size= batch_size_valid,
            shuffle=True,
            num_workers=int(opt_valid.nThreads),
        collate_fn=collateClassValid)
        #print(dataloader_decoder.__len__())
        #print(dataloader_valid.__len__())
        
        feature_extractor.train()
        #feature_extractor.train()
        

        #Takes x-rays as input and calculates loss based on the l1 diff between projected prediction and x-rays
        #x_ray 1 is from side and x_ray2 is from front
        for epoch in range(epochs):
            running_loss = 0.0
            counter = 0

            for i,data in enumerate(dataloader_decoder):   
                
                
                x_ray_S= data[1][1].to(device) #x_ray_1
                x_ray_F = data[1][0].to(device) # x_ray_2

                optimizer.zero_grad()

                #print(template.size())
                temp = template.repeat(data[0].size()[0],1,1,1,1)

                predicts = feature_extractor(temp)
               
                pred1 = predicts[0]
                pred2 = predicts[1]
                pred3 = predicts[2]
                pred4 = predicts[3]

                
                #F stands for Front and S stands for Side
                pred1_F, pred1_S = projection_visual(opt,pred1) #Size [1, 1, 128, 128] 
                pred2_F, pred2_S = projection_visual(opt,pred2) #Size [1, 1, 128, 128] 
                pred3_F, pred3_S = projection_visual(opt,pred3) #Size [1, 1, 128, 128] 
                pred4_F, pred4_S = projection_visual(opt,pred4) #Size [1, 1, 128, 128] 
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

                loss_res.backward()
                optimizer.step()
                if i % batch_print == 0:
                    print("\n Epoch: {}, Loss: {}, sample {}\n ".format(epoch, loss_res.item(),i*batch_size))
                running_loss = running_loss + loss_res.item()
                
                
            curr_lr = scheduler.optimizer.param_groups[0]['lr']
            if epoch % 35 == 0 and epoch > 0 and epoch < epochs:
                scheduler.step()
            avg_loss = running_loss/len(dataloader_auto)
            avg_losses.append(avg_loss)
            lr_list.append(curr_lr)

            if valid_feature_extractor:
                valid_running_loss = 0.0
                counter2 = 0
                with torch.no_grad():
                    for i_v, data_v in enumerate(dataloader_valid):
                        x_ray_S= data_v[1][1].to(device)
                        x_ray_F = data_v[1][0].to(device)
                        temp  = template.repeat(data_v[0].size()[0],1,1,1,1)
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
                        #print(pred1_F.size())
                        #print(x_ray_F.size())
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
                        valid_running_loss = valid_running_loss + loss_res.item()
                    valid_avg_loss = valid_running_loss/len(dataloader_valid)
                    valid_avg_losses.append(valid_avg_loss)
            if epoch % 2 == 0:
                torch.save(feature_extractor.state_dict(), os.path.join(opt.MODEL_SAVE_PATH,"saved_states","feature_extractor{}.pt".format(epoch)))
            print("\n valid-loss: {} \n".format(valid_avg_losses[len(valid_avg_losses)-1]))

        
        feature_extractor_path = os.path.join(opt.MODEL_SAVE_PATH,"feature_extractor.pt")
        feature_extractor_figs_path = os.path.join(opt.MODEL_SAVE_PATH,"figs","feature_extractor","train")
        feature_map_path = os.path.join(opt.MODEL_SAVE_PATH,"feature_map")
        

        avg_loss_path = os.path.join(feature_extractor_figs_path,"avg-loss.png")
        avg_loss_valid_path = os.path.join(feature_extractor_figs_path,"avg-valid-loss.png")
        lr_path = os.path.join(feature_extractor_figs_path,"lr.png")


        if os.path.isfile(avg_loss_path):
            os.remove(avg_loss_path)
        if os.path.isfile(feature_extractor_path):
            os.remove(feature_extractor_path)
        if os.path.isfile(lr_path):
            os.remove(lr_path)
        for f in os.listdir(feature_map_path):
            os.remove(os.path.join(feature_map_path, f))

        for f in os.listdir(feature_extractor_figs_path):
            os.remove(os.path.join(feature_extractor_figs_path, f))

        torch.save(feature_extractor.state_dict(),feature_extractor_path)

     
        fg_loss = Figure()
        ax_loss = fg_loss.gca()
        ax_loss.plot(avg_losses)
        ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_loss.set_xlabel('epochs', fontsize=10)
        ax_loss.set_ylabel('avg-loss', fontsize='medium')
        
        fg_loss.savefig(avg_loss_path)

        fg_lr = Figure()
        ax_lr = fg_lr.gca()
        ax_lr.plot(lr_list)
        ax_lr.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_lr.set_xlabel('epochs', fontsize=10)
        ax_lr.set_ylabel('lr', fontsize='medium')
        
        fg_lr.savefig(lr_path)

        if valid_feature_extractor:
            fg_loss = Figure()
            ax_loss = fg_loss.gca()
            ax_loss.plot(valid_avg_losses)
            ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_loss.set_xlabel('epochs', fontsize=10)
            ax_loss.set_ylabel('avg-valid-loss', fontsize='medium')
        
            fg_loss.savefig(avg_loss_valid_path)



        #if os.path.isfile(enc_fmaps_path):
            #os.remove(enc_fmaps_path)
        #if os.path.isfile(dec_fmaps_path):
            #os.remove(dec_fmaps_path)
        #for f in os.listdir(feature_map_path):
            #os.remove(os.path.join(feature_map_path, f))
        #torch.load(decoder_fmap,os.path.join(feature_map_path,"dec_fmaps.pt"))
        #torch.load(decoder_fmap,os.path.join(feature_map_path,"enc_fmaps.pt"))
        

        #Saves enc_fmaps and dec_fmaps
        feature_extractor2 = feature_extractor = ResUNet2(in_channel=1,out_channel=1, training=False, out_fmap = True).to(device)
        feature_extractor2.load_state_dict(torch.load(feature_extractor_path))
        with torch.no_grad():
            pred, enc_fmaps, dec_fmaps = feature_extractor2(template)

        for i in range(5):
            torch.save(enc_fmaps[i],os.path.join(feature_map_path, "enc_fmap{}.pt".format(i+1)))
        for i in range(5):
            torch.save(dec_fmaps[i],os.path.join(feature_map_path, "dec_fmap{}.pt".format(i+1)))
        
      





    




    
    #print(feature_map)
    #print("SUM of feature map: {}".format(torch.sum(feature_map)))
   
