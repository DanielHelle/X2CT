       
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


MODEL_SAVE_PATH= os.path.abspath(os.path.join(os.path.dirname(__file__),"save_models"))





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

template_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"data", "template-data","models","template.pt"))
template = torch.load(template_path).to(device)
       
template = torch.unsqueeze(template,dim=0)
template = torch.unsqueeze(template,dim=0)
        

feature_extractor_path = os.path.join(MODEL_SAVE_PATH,"feature_extractor.pt")
feature_map_path = os.path.join(MODEL_SAVE_PATH,"feature_map")
        
feature_extractor2 = feature_extractor = ResUNet2(in_channel=1,out_channel=1, training=False, out_fmap = True).to(device)
feature_extractor2.load_state_dict(torch.load(feature_extractor_path))
with torch.no_grad():
    pred, enc_fmaps, dec_fmaps = feature_extractor2(template)

for i in range(5):
    torch.save(enc_fmaps[i],os.path.join(feature_map_path, "enc_fmap{}.pt".format(i+1)))
for i in range(5):
    torch.save(dec_fmaps[i],os.path.join(feature_map_path, "dec_fmap{}.pt".format(i+1)))