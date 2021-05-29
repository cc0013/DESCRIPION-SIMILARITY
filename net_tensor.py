import torchvision.models as models
import torch


class VGGNet:
    def __init__(self):
        self.model = models.vgg16(pretrained=True)

  
        self.model = torch.nn.Sequential(
                    self.model.features, 
                    torch.nn.AdaptiveMaxPool2d(1), 
                    torch.nn.Flatten() # 
        )

        for param in self.model.parameters():
            param.requires_grad = False

       self.model.cuda() # Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

    # def extract_features(self, x):
    #     features = self.model(x) # tensor
    #     for i in range(len(features)):
    #         features[i] = features[i]/features[i].norm(2) 
    #     return features # type(features): tensor(n, 512)

    def extract_features(self, x):
        features = self.model(x) # tensor

        flag = False
        for i in range(len(features)):
            feat_single = features[i]/features[i].norm(2)
            feat_single = feat_single.unsqueeze(0)

            if flag == False:
                feats = feat_single
                flag = True
                continue

            feats = torch.cat((feats, feat_single))

        return feats # type(feats): tensor(n, 512)
