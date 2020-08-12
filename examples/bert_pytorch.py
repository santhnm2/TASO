import numpy as np
import torch
import onnx
from onnx import numpy_helper
import re

np.random.seed(0)

def get_weights(path):
    model = onnx.load(path)
    weights = {}
    for value in model.graph.initializer:
        if 'kernel' in value.name:
            match = re.search('layer_(\d+)\/attention\/(\w+)\/(\w+)\/kernel', value.name)
            if match is not None:
                layer = int(match.group(1))
                weight = match.group(3)
                if layer not in weights:
                    weights[layer] = {}
                weights[layer][weight] = numpy_helper.to_array(value)
    return weights

class Attention(torch.nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.query = torch.nn.Linear(in_features=d_model, out_features=d_model,
                                     bias=False)
        self.key = torch.nn.Linear(in_features=d_model, out_features=d_model,
                                   bias=False)
        self.value = torch.nn.Linear(in_features=d_model, out_features=d_model,
                                     bias=False)
        self.dense = torch.nn.Linear(in_features=d_model, out_features=d_model,
                                     bias=False)

    def forward(self, x):
        assert(x.shape[-1] == self.d_model)
        seq_length = x.shape[0]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(seq_length, self.heads, -1)
        k = k.reshape(seq_length, self.heads, -1)
        v = v.reshape(seq_length, self.heads, -1)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1)

        logits = torch.matmul(q, k)
        logits = logits / np.sqrt(self.d_model // self.heads)
        logits = torch.softmax(logits, dim=-1)
        output = torch.matmul(logits, v)
        output = output.reshape((seq_length, self.d_model))
        output = self.dense(output)
        return output

class Bert(torch.nn.Module):
    def __init__(self, num_layers, d_model, heads):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.layers = [Attention(d_model, heads) for i in range(num_layers)]

    def forward(self, x):
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
        return x

model = Bert(1, 768, 12)
weights = get_weights('/lfs/1/keshav2/approximate_taso/onnx_models/bert/bertsquad10.onnx')
for i, layer in enumerate(model.layers):
    layer.query.weight = torch.nn.Parameter(torch.from_numpy(weights[i]['query']))
    layer.key.weight = torch.nn.Parameter(torch.from_numpy(weights[i]['key']))
    layer.value.weight = torch.nn.Parameter(torch.from_numpy(weights[i]['value']))
    layer.dense.weight = torch.nn.Parameter(torch.from_numpy(weights[i]['dense']))

with torch.no_grad():
    x = np.random.normal(0, 1, (512, 768))
    output = model(torch.from_numpy(x).float())
print('Min:', torch.min(output))
print('Median:', torch.median(output))
print('Max:', torch.max(output))
