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
            match = re.search('layer_(\d+)\/(\w+).*\/(\w+)\/kernel', value.name)
            if match is not None:
                layer = int(match.group(1))
                group = match.group(2)
                weight = match.group(3)
                if group != 'attention':
                    weight = group
                if layer not in weights:
                    weights[layer] = {}
                weights[layer][weight] = numpy_helper.to_array(value)
                if group != 'attention':
                    print('%d %s:' % (layer, weight), weights[layer][weight].shape)
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

    def forward(self, x, proj_k=None):
        assert(x.shape[-1] == self.d_model)
        seq_length = x.shape[0]
        d_k = self.d_model // self.heads

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        q = q.reshape(seq_length, self.heads, d_k)
        k = k.reshape(seq_length, self.heads, d_k)
        v = v.reshape(seq_length, self.heads, d_k)

        q = q.permute(1, 0, 2)
        k = k.permute(1, 2, 0)
        v = v.permute(1, 0, 2)
        
        if proj_k is not None:
            stddev = 1.0 / np.sqrt(proj_k)
            e = torch.FloatTensor(self.heads, seq_length, proj_k).uniform_(-stddev, stddev)
            f = torch.FloatTensor(self.heads, proj_k, seq_length).uniform_(-stddev, stddev)
            k = torch.matmul(k, e)
            v = torch.matmul(f, v)

        logits = torch.matmul(q, k)        
        logits = torch.div(logits, np.sqrt(d_k))
        logits = torch.softmax(logits, dim=-1)
        output = torch.matmul(logits, v)
        output = output.permute(1, 0, 2)
        output = output.reshape((seq_length, self.d_model))
        output = self.dense(output)
        return output

class FeedForward(torch.nn.Module):
    def __init__(self):
        pass

class Bert(torch.nn.Module):
    def __init__(self, num_layers, d_model, heads):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.layers = [Attention(d_model, heads) for i in range(num_layers)]

    def forward(self, x, proj_k=None):
        #x = self.relu(x)
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) // 2:
                x = layer(x, proj_k)
            else:
                x = layer(x)
        return x

model = Bert(12, 768, 12)
weights = get_weights('/lfs/1/keshav2/approximate_taso/onnx_models/bert/bertsquad10.onnx')
for i, layer in enumerate(model.layers):
    layer.query.weight = torch.nn.Parameter(torch.from_numpy(np.copy(weights[i]['query'])))
    layer.key.weight = torch.nn.Parameter(torch.from_numpy(np.copy(weights[i]['key'])))
    layer.value.weight = torch.nn.Parameter(torch.from_numpy(np.copy(weights[i]['value'])))
    layer.dense.weight = torch.nn.Parameter(torch.from_numpy(np.copy(weights[i]['dense'])))
model.eval()

x = np.random.normal(0, 1, (1024, 768))
with torch.no_grad():
    output = model(torch.from_numpy(x).float())
print('Baseline:')
print('Min:', torch.min(output))
print('Median:', torch.median(output))
print('Max:', torch.max(output))
with torch.no_grad():
    linformer_output = model(torch.from_numpy(x).float(), proj_k=512)
print('With Linformer (k=256):')
print('Min:', torch.min(linformer_output))
print('Median:', torch.median(linformer_output))
print('Max:', torch.max(linformer_output))

