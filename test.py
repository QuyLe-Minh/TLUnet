from architecture.TLUnet import TLUnet
from architecture.CNN3D import CNN3D
import torch

model = TLUnet()

# model.apply(init_weights)
def applyWeight(model, model_weight):
    weight = torch.load(model_weight)
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= 36: break
        
        param.data = weight[name]
        param.requires_grad = False
        
applyWeight(model, "cnn3d_2.pt")

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable parameters:", total_trainable_params)