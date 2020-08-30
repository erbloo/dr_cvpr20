""" Adapted VGG pytorch model that used as surrogate. """
import torchvision.models as models
import torch

import pdb


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.model = models.vgg16(pretrained=True).cuda().eval()
        features = list(self.model.features)
        self.features = torch.nn.ModuleList(features).cuda().eval()

    def prediction(self, x, internal=[]):
        pred = self.model(x)
        if len(internal) == 0:
            return pred
        
        layers = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if(ii in internal):
            # if isinstance(model, torch.nn.modules.conv.Conv2d):
                layers.append(x)
        return layers, pred

if __name__ == "__main__":
    Vgg16()
