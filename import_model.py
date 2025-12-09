import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
import timm
model = timm.create_model("resnet18", pretrained=True)
torch.save(model.state_dict(), "resnet18_pretrained.pth")
