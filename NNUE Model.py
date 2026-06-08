import torch
import torch.nn as nn
class NNUE(nn.Module):
    def __init__(self,input_size=41024):
      super().__init__()
      self.input_weights=nn.Parameter(torch.randn(input_size,512)*0.01)
      self.input_bias=nn.Parameter(torch.zeros(512))
      self.l2=nn.Linear(512,32)
      self.l3=nn.Linear(32,32)
      self.l4=nn.Linear(32,1)
    def clipped_relu(self,x):
        return torch.clamp(x,min=0,max=1)
    def refresh_accumulator(self,active_features):
        acc=self.input_bias.copy()
        for i in active_features:
          acc+= self.input_weights[i]
        return acc
    def update_accumulator(self,accumulator,added_features,removed_features):
        acc=accumulator.copy()
        for i in added_features:
          acc+=self.input_weights[i]
        for i in removed_features:
          acc-=self.input_weights[i]
        return acc
    def forward(self,accumulator):
        x=self.clipped_relu(accumulator)
        x=self.clipped_relu(self.l2(x))
        x=self.clipped_relu(self.l3(x))
        x=self.l4(x)
        return x
