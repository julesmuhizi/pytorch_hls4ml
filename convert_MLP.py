import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchinfo import summary
import tqdm.notebook as tqdm
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
import hls4ml


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # fully connected layer, output 10 classes
        self.flatten = nn.Flatten(1,3)
        self.out = nn.Linear(28 * 28, 10)
    def forward(self, x):
        # flatten the input to (batch_size, 1 * 28 * 28)
        x = self.flatten(x)     
        output = self.out(x)
        return output    

# load pytorch model 
PATH = './model/mnist_MLP.pth'
model = torch.load(PATH)

# load test dataset
root = './data'
if not os.path.exists(root):
    os.mkdir(root)
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=False)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=False)

test_batch_size = 1

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=test_batch_size,
                shuffle=False)

# Input to the model
x = next(iter(test_loader))[0]
torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "./model/onnx/mnist_MLP.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                 )

onnx_model = ModelWrapper("./model/onnx/mnist_MLP.onnx")
onnx_model = cleanup_model(onnx_model)
onnx_model = onnx_model.transform(GemmToMatMul())
onnx_model = cleanup_model(onnx_model)
onnx_model.save("./model/onnx/mnist_MLP_noGemm_clean.onnx") 

# convert ONNX model to HLS model
model_name = "./model/onnx/mnist_MLP_noGemm_clean.onnx"
config = hls4ml.utils.config_from_onnx_model(model_name, granularity='model')
hls_model = hls4ml.converters.convert_from_onnx_model(model_name, hls_config=config, output_dir='hls_project')
hls_model.compile()