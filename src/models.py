import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
import data.Dti_dg_lib.networks as networks
from copy import deepcopy


class Learner(nn.Module):
    def __init__(self, args, hid_dim = 128, weights = None):
        super(Learner, self).__init__()
        self.block_1 = nn.Sequential(nn.Linear(args.input_dim, hid_dim), nn.LeakyReLU(0.1))
        self.block_2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.LeakyReLU(0.1))
        self.fclayer = nn.Sequential(nn.Linear(hid_dim, 1))

        if weights != None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward_mixup(self, x1, x2, lam=None):
        x1 = self.block_1(x1)
        x2 = self.block_1(x2)
        x = lam * x1 + (1 - lam) * x2
        x = self.block_2(x)
        output = self.fclayer(x)
        return output

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        output = self.fclayer(x)
        return output

    def repr_forward(self, x):
        with torch.no_grad():
            x = self.block_1(x)
            repr = self.block_2(x)
            return repr
    
from torchvision import models

class Learner_RCF_MNIST(nn.Module):
    def __init__(self, args, weights = None):
        super(Learner_RCF_MNIST, self).__init__()
        self.args = args

        # get feature extractor from original model
        ori_model = models.resnet18(pretrained=True)
        #for param in model.parameters():
        #    param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = ori_model.fc.in_features
        # print(f'num_ftrs = {num_ftrs}')
        
        self.feature_extractor = torch.nn.Sequential(*list(ori_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        self.fc = nn.Linear(num_ftrs, 1)

        if weights != None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

    def forward_mixup(self, x1, x2, lam = None):
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)

        # mixup feature
        x = lam * x1 + (1 - lam) * x2
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


# ---> :https://github.com/laiguokun/LSTNet
class Learner_TimeSeries(nn.Module):
    def __init__(self, args, data, weights = None):
        super(Learner_TimeSeries, self).__init__()
        self.use_cuda = args.cuda
        self.P = int(args.window)
        self.m = int(data.m)
        self.hidR = int(args.hidRNN)
        self.hidC = int(args.hidCNN)
        self.hidS = int(args.hidSkip)
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = int((self.P - self.Ck)/self.skip)
        print(f'self.pt = {self.pt}')
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = args.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            print(self.hidR + self.skip * self.hidS, self.m)
            self.linear1 = nn.Linear(int(self.hidR + self.skip * self.hidS), self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0): #highway -> autoregressiion
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

        if weights != None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self, x):
        batch_size = x.size(0)
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN time number <-> layer number
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
            _, s = self.GRUskip(s)
            s = s.view(batch_size, int(self.skip * self.hidS))
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        # FC
        res = self.linear1(r)
        
        #highway auto-regression
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
            
        if (self.output):
            res = self.output(res)

        return res

    def repr_forward(self, x):
        batch_size = x.size(0)
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN time number <-> layer number
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
            _, s = self.GRUskip(s)
            s = s.view(batch_size, int(self.skip * self.hidS))
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        # FC
        return r
        res = self.linear1(r)
        
        #highway auto-regression
            

    def forward_mixup(self, x1, x2, lam):
        batch_size = x1.size(0)
        #CNN
        c1 = x1.view(-1, 1, self.P, self.m)
        c1 = F.relu(self.conv1(c1))
        c1 = self.dropout(c1)
        c1 = torch.squeeze(c1, 3)

        #CNN
        c2 = x2.view(-1, 1, self.P, self.m)
        c2 = F.relu(self.conv1(c2))
        c2 = self.dropout(c2)
        c2 = torch.squeeze(c2, 3)
        
        # just mixup after conv block
        c = lam * c1 + (1 - lam) * c2

        # RNN time number <-> layer number
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
            _, s = self.GRUskip(s)
            s = s.view(batch_size, int(self.skip * self.hidS))
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        # FC
        res = self.linear1(r)
        
        #highway auto-regression --> not mixup
        if (self.hw > 0):
            x = lam * x1 + (1 - lam) * x2
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
            
        if (self.output):
            res = self.output(res)
        return res

# ---> https://github.com/mims-harvard/TDC/tree/master/
class Learner_Dti_dg(nn.Module):
    def __init__(self, hparams = None, weights = None):
        super(Learner_Dti_dg, self).__init__()

        self.num_classes = 1
        self.input_shape = [(63, 100), (26, 1000)]
        self.num_domains = 6
        self.hparams = hparams

        self.featurizer = networks.DTI_Encoder()
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes,
            False)
            #self.hparams['nonlinear_classifier'])

        #self.network = mySequential(self.featurizer, self.classifier)

        if weights != None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self,x):
        drug_num = self.input_shape[0][0] * self.input_shape[0][1]
        x_drug = x[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
        x_protein = x[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])
        
        feature_out = self.featurizer.forward(x_drug,x_protein)
        linear_out = self.classifier(feature_out)
        return linear_out
    
    def repr_forward(self, x):
        with torch.no_grad():
            drug_num = self.input_shape[0][0] * self.input_shape[0][1]
            x_drug = x[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
            x_protein = x[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])
            
            repr = self.featurizer.forward(x_drug,x_protein)
            return repr

    def forward_mixup(self,x1, x2, lambd):
        drug_num = self.input_shape[0][0] * self.input_shape[0][1]
        x1_drug = x1[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
        x1_protein = x1[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])
        
        x2_drug = x2[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
        x2_protein = x2[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])

        feature_out = self.featurizer.forward_mixup(x1_drug,x1_protein,x2_drug,x2_protein,lambd)
        linear_out = self.classifier(feature_out)
        return linear_out
        #return self.network.forward_mixup(x1_drug, x1_protein,x2_drug, x2_protein,lambd)

