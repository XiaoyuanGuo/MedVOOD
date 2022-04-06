import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, num_classes=14, capacity=16, channel=3, N = 256):
        super(ResNet, self).__init__()
        self.extractor = models.resnet18(pretrained=True)
        self.extractor_dim = self.extractor.fc.in_features
        self.out_classes = num_classes
        self.extractor.fc = nn.Identity()

        self.proj1 = nn.Linear(self.extractor_dim, N) 
        self.proj2 = nn.Linear(self.extractor_dim, N) 
        self.proj3 = nn.Linear(self.extractor_dim, N) 
        self.proj4 = nn.Linear(self.extractor_dim, N)
        self.proj5 = nn.Linear(self.extractor_dim, N) 
        self.proj6 = nn.Linear(self.extractor_dim, N) 
        self.proj7 = nn.Linear(self.extractor_dim, N) 
        self.proj8 = nn.Linear(self.extractor_dim, N) 
        self.proj9 = nn.Linear(self.extractor_dim, N) 
        self.proj10 = nn.Linear(self.extractor_dim, N) 
        self.proj11 = nn.Linear(self.extractor_dim, N) 
        self.proj12 = nn.Linear(self.extractor_dim, N) 
        self.proj13 = nn.Linear(self.extractor_dim, N) 
        self.proj14 = nn.Linear(self.extractor_dim, N)

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(N, 1) #class 1
        self.fc2 = nn.Linear(N, 1) #class 2
        self.fc3 = nn.Linear(N, 1) #class 3
        self.fc4 = nn.Linear(N, 1) #class 4
        self.fc5 = nn.Linear(N, 1) #class 5
        self.fc6 = nn.Linear(N, 1) #class 6
        self.fc7 = nn.Linear(N, 1) #class 7
        #reconstruction
#         self.decoder = aeDecoder(capacity, channel, num_classes)

        self.sigm = nn.Sigmoid()


    def forward(self, x):
        #first, process images    
        out = self.extractor(x)

        #out1 - class
        out1 = self.proj1(out)
        out_feat1 = out1
        
        #out2 - class
        out2 = self.proj2(out)
        out_feat2 = out2
        
        #out3 - class
        out3 = self.proj3(out)
        out_feat3 = out3
        
        #out4 - class
        out4 = self.proj4(out)
        out_feat4 = out4
        
        #out5 - class
        out5 = self.proj5(out)
        out_feat5 = out5
          
        #out6 - class
        out6 = self.proj6(out)
        out_feat6 = out6
        
        #out7 - class
        out7 = self.proj7(out)
        out_feat7 = out7
        
        out1 = self.sigm(self.fc1(out1))
        out2 = self.sigm(self.fc2(out2))
        out3 = self.sigm(self.fc3(out3))
        out4 = self.sigm(self.fc4(out4))
        out5 = self.sigm(self.fc5(out5))
        out6 = self.sigm(self.fc6(out6))
        out7 = self.sigm(self.fc7(out7))

        
            
        return (out1, out2, out3, out4, out5, out6, out7), (out_feat1, out_feat2, out_feat3, out_feat4, out_feat5, out_feat6, out_feat7)       
