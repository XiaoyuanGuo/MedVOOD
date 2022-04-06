import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models


class MyDenseNetConv(torch.nn.Module):
    def __init__(self, fixed_extractor = True):
        super(MyDenseNetConv,self).__init__()
        original_model = torchvision.models.densenet161(pretrained=True)
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])
        
        if fixed_extractor:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x
    
    
class DenseNet(nn.Module):
    def __init__(self, num_classes=7, capacity=16, channel=3, N = 256):
        super(DenseNet, self).__init__()
        self.out_classes = num_classes
        self.extractor = MyDenseNetConv()
        self.proj1 = nn.Linear(2208, N)
        self.proj2 = nn.Linear(2208, N) 
        self.proj3 = nn.Linear(2208, N)
        self.proj4 = nn.Linear(2208, N)
        self.proj5 = nn.Linear(2208, N)
        self.proj6 = nn.Linear(2208, N)
        self.proj7 = nn.Linear(2208, N)

        self.fc1 = nn.Linear(N, 1) #class 1
        self.fc2 = nn.Linear(N, 1) #class 2
        self.fc3 = nn.Linear(N, 1) #class 3
        self.fc4 = nn.Linear(N, 1) #class 4
        self.fc5 = nn.Linear(N, 1) #class 5
        self.fc6 = nn.Linear(N, 1) #class 6
        self.fc7 = nn.Linear(N, 1) #class 7
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
