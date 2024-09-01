import torch
from models import LR_Model

def convert_sklearn_to_torch_model(clf, num_input, device):
    model_temp = LR_Model(num_input).to(device)
    model_temp.linear.weight.data = torch.tensor(clf.coef_).float()
    model_temp.linear.bias.data = torch.tensor(clf.intercept_).float()
    return model_temp
    
def convert_sklearn_to_torch_model_v2(W, B, num_input, device):
    model_temp = LR_Model(num_input).to(device)
    model_temp.linear.weight.data = W.float()
    model_temp.linear.bias.data = B
    return model_temp
