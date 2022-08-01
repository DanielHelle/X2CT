import os
from re import template
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from lib.dataset.factory import get_dataset
import argparse
import h5py
import numpy as np
import nibabel as nib

#You might have to run this script in a separate environment because of incompatible dependencies

class optDict():
    def __init__(self, dataset_class):
        self.dataset_class = dataset_class
        self.datasetfile = "./data/train.txt"
        self.dataroot = "./data/LIDC-HDF5-256"
        self.ct_channel = 128
        self.fine_size = 128
        self.resize_size= 150
        self.xray_channel= 1
        self.CT_MIN_MAX= [0, 2500]
        self.XRAY1_MIN_MAX= [0, 255]
        self.XRAY2_MIN_MAX= [0, 255]
        self.CT_MEAN_STD= [0.0, 1.0]
        self.XRAY1_MEAN_STD= [0.0, 1.0]
        self.XRAY2_MEAN_STD= [0.0, 1.0]
        self.nThreads= 5                                                       #3DGAN
        self.TEMPLATE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),"data", "template-data"))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--createTemplateData",
        help="Choose if you want to create data for template registration in X2CT\3DGAN\data\template-data",
        dest="createTemplateData",
        default="0",
        type=str,
)
    parser.add_argument(
        "--convertToTensor",
        help="Choose if you want to convert template.nii.gz to pytorch tensor and save it in X2CT\3DGAN\data\template-data",
        dest="convertToTensor",
        default="0",
        type=str,
)
    args = parser.parse_args()
    args.createTemplateData = str2bool(args.createTemplateData)
    args.convertToTensor = str2bool(args.convertToTensor)
    opt = optDict("align_ct_xray_views_std")
    

    if args.createTemplateData:

        data_path = opt.TEMPLATE_DATA_PATH
        print(data_path)
        
        print(opt.dataset_class)

        datasetClass, augmentationClass, dataTestClass, collateClass = get_dataset(opt.dataset_class)
        opt.data_augmentation = augmentationClass

    
        
        for f in os.listdir(data_path):
            os.remove(os.path.join(data_path, f))


        dataset = datasetClass(opt)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1 ,#916 total
            shuffle=False,
            num_workers=int(opt.nThreads),
            collate_fn=collateClass)
        vol_names = []

        for idx, data in enumerate(dataloader):
            data = torch.squeeze(data[0],dim=0).numpy()
            #open(os.path.join(data_path,"vol{}.npy".format(idx)))
            #f = open(os.path.join(data_path,"vol{}.npy".format(idx)),"x")
            np.save(os.path.join(data_path,"vol{}.npy".format(idx)), data)
            vol_names.append("vol{}.npy".format(idx))
            
            
        f_txt = open(os.path.join(data_path,"data.txt"),"w")
        for elem in vol_names:
            f_txt.write(elem+"\n")
        f_txt.close()
    


    if args.convertToTensor:
        data_path = opt.TEMPLATE_DATA_PATH
        template_path = os.path.join(data_path,"models","template.nii.gz")
        template = nib.load(template_path)
        template = np.array(template.dataobj)
        template = torch.from_numpy(template)
        torch.save(template, os.path.join(data_path,"models","template.pt"))
        





