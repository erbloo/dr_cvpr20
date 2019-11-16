import torch
from torch.autograd import Variable
import numpy as np

def numpy_to_variable(image, device=torch.device('cuda:0')):
    x_image = np.expand_dims(image, axis=0)
    x_image = Variable(torch.tensor(x_image))
    x_image = x_image.to(device)
    x_image.retain_grad()
    return x_image

def variable_to_numpy(variable):
    return np.squeeze(variable.cpu().detach().numpy())
