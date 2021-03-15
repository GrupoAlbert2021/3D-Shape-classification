import torch
import torch.nn as nn
import torch.nn.functional as F

# We will start by defining the transformation networks (input and feature transform)
# Note: The implementation given below can be used both for the input and feature transforms 
#       simply by specifying the expected output dimension (respectively 3 and 64).

class Tnet(nn.Module): 
   def __init__(self, k=3):
      super().__init__()
      self.k=k #input dimension
      
      self.conv1 = nn.Conv1d(k, 64, 1) 
      self.conv2 = nn.Conv1d(64, 128, 1)
      self.conv3 = nn.Conv1d(128, 1024, 1)
      self.fc1 = nn.Linear(1024, 512)
      self.fc2 = nn.Linear(512, 256)
      self.fc3 = nn.Linear(256, k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       
   def forward(self, input):
      # input.shape = [bs,3,n]
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      xb = nn.MaxPool1d(xb.size(-1))(xb) 
      # xb.shape = [bs, n, 1]
      xb = xb.view(bs, -1)  #Alternative: xb = nn.Flatten(1)(xb) 
      # xb.shape = [bs, n]
      xb = F.relu(self.bn4(self.fc1(xb)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #The output matrix is initialized as an identity matrix
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      #matrix0 = self.fc3(xb) -> matrix0.shape = [bs, 9] or [bs, 4096]
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init  #matrix.shape = [bs, 3, 3] or [bs, 64, 64]
      return matrix   #Return the transformation matrix (3x3 or 64x64)


# Implement the transformation part of the classification PoinNet 

class Transform(nn.Module): 
   def __init__(self):
        super().__init__()
        
        self.input_transform = Tnet(k=3)     # Input transformation matrix [bs, 3, 3]
        self.feature_transform = Tnet(k=64)  # Feature transformation matrix [bs, 64, 64]
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
       
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        # input.shape = [bs,3,n]
        bs = input.size(0)

        input_transform = self.input_transform(input)  #Obtain the input transform matrix 3x3
        
        xb = torch.bmm(torch.transpose(input,1,2), input_transform).transpose(1,2) # We multiply each point of the batch by the input transform matrix 

        xb = F.relu(self.bn1(self.conv1(xb))) #pass it through the 1st conv. block (conv+bn1+Relu)

        feature_transform = self.feature_transform(xb) #Obtain the feature transform matrix 64x64
          
        xb = torch.bmm(torch.transpose(xb,1,2), feature_transform).transpose(1,2) # We multiply each point of the batch by the feature transform matrix

        xb = F.relu(self.bn2(self.conv2(xb)))  #pass it through the 2nd conv. block (conv+bn2+Relu)
        
        xb = self.bn3(self.conv3(xb))  #pass it through the 3rd conv. block (conv+bn)
        
        xb = nn.MaxPool1d(xb.size(-1))(xb) #Apply max pooling

        global_feature = xb.view(bs, -1)  #Flatten the output in order to pass it through the last MLP network
        
        return global_feature  #Return the global feature vector


#Implement the last part (last MLP) of the classification PointNet Network
class ClassificationPointNet(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        
        self.transform = Transform()
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, input):
        # input shape [bs,3,n]
        global_feature = self.transform(input)
        
        #pass it throught the last MLP (512,256,k)
        xb = F.relu(self.bn1(self.fc1(global_feature))) #Pass it through the 1st FC layer
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb)))) #Pass it through the 2nd FC layer
        
        output = self.fc3(xb)  #Pass it through the 3rd FC layer
        
        return output
