import numpy as np
import onnx
from onnx import numpy_helper
import re
import taso as ts

np.random.seed(0)

seq_length = 2048
hidden_dims = 768
mul = 4
num_heads = 12
num_layers = 12
BERT_PATH = '/lfs/1/keshav2/approximate_taso/onnx_models/bert/bertsquad10.onnx'

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
    return weights

def attention(graph, input, heads, weight_data, scale_factor):
    seq_length = input.dim(0)
    d_model = input.dim(1)
    d_k = d_model // heads
    assert input.dim(1) % heads == 0
    weight_dims = (d_model, d_model)
    weights = {
        'query': graph.new_weight(dims=weight_dims, data=weight_data['query']),
        'key': graph.new_weight(dims=weight_dims, data=weight_data['key']),
        'value': graph.new_weight(dims=weight_dims, data=weight_data['value']),
        'dense': graph.new_weight(dims=weight_dims, data=weight_data['dense']),
    }
    
    # compute query, key, value tensors
    q = graph.matmul(input, weights['query'])
    k = graph.matmul(input, weights['key'])
    v = graph.matmul(input, weights['value'])
    
    # reshape query, key, value to multiple heads
    q = graph.reshape(q, shape=(seq_length,heads,d_k))
    k = graph.reshape(k, shape=(seq_length,heads,d_k))
    v = graph.reshape(v, shape=(seq_length,heads,d_k))
    
    # transpose query, key, value for batched matmul
    q = graph.transpose(q, perm=(1,0,2), shuffle=True)
    k = graph.transpose(k, perm=(1,2,0), shuffle=True)
    v = graph.transpose(v, perm=(1,0,2), shuffle=True)

    # perform matrix multiplications
    logits = graph.matmul(q, k)
    #logits = graph.div(x=logits, y=scale_factor) 
    #logits = graph.softmax(logits)
    scores = graph.matmul(logits, v)
    
    # transpose the scores back
    scores = graph.transpose(scores, perm=(1,0,2), shuffle=True)
    scores = graph.reshape(scores, shape=(seq_length, d_model))

    # a final linear layer
    dense = graph.matmul(scores, weights['dense'])
     
    return dense

def feed_forward(graph, input, mul, weight_data):
    seq_length = input.dim(0)
    d_model = input.dim(1)
    weights = {
        'intermediate': graph.new_weight(dims=(d_model, d_model * mul),
                                         data=weight_data['intermediate']),
        'output': graph.new_weight(dims=(d_model * mul, d_model),
                                   data=weight_data['output'])
    }
    
    intermediate = graph.matmul(input, weights['intermediate'])
    output = graph.matmul(intermediate, weights['output'])
    return output

def print_output_summary(output, preamble=None):
    if preamble:
        print(preamble)
    print('Min:', np.min(output))
    print('Median:', np.median(output))
    print('Max:', np.max(output))

def main():
    weights = get_weights(BERT_PATH)
    x = np.random.normal(0, 1, (seq_length, hidden_dims))
    graph = ts.new_graph()
    input = graph.new_input(dims=(seq_length, hidden_dims))
    graph.set_input_value(graph.get_input_list()[0], x)
    scale_factor = graph.new_weight(dims=(1,),
                                    data=np.asarray([np.sqrt(hidden_dims // num_heads)]))
    input = graph.relu(input)
    t = input
    for i in range(num_layers):
        t = attention(graph, t, num_heads, weights[i], scale_factor)
        # t = feed_forward(graph, t, mul, weights[i])

    #model = ts.export_onnx(graph)
    #onnx.save(model, '/lfs/1/keshav2/approximate_taso/onnx_models/bert/bert_taso.onnx')
    original_output = np.reshape(graph.evaluate(), (seq_length, hidden_dims))
  
    new_graph = ts.optimize(graph, alpha=1.0, budget=15, original_subst=False, print_subst=True)
    #model = ts.export_onnx(new_graph)
    #onnx.save(model, '/lfs/1/keshav2/approximate_taso/onnx_models/bert/bert_taso_optimized.onnx')
    optimized_output = np.reshape(new_graph.evaluate(), (seq_length, hidden_dims))
    print_output_summary(original_output, 'Before optimization:')
    print()
    print_output_summary(optimized_output, 'After optimization:')

if __name__=='__main__':
    main()
