import os
import math
import torch
import numpy as np


f = open('./task/classification/resnet_18/voc2010_crop/bird/0/net-1.pkl','rb')

mdl = torch.load(f)

print(mdl)


