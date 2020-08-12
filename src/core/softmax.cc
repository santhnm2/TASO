/* Copyright 2018 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "taso/ops.h"
using namespace taso;

TensorHandle Graph::softmax(const TensorHandle _input)
{
  Op op = model->get_or_create_softmax(*_input);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_softmax(const Tensor& _input)
{
  SoftmaxKey key(_input);
  Softmax* softmaxOp;
  if (softmax.find(key) != softmax.end()) {
    softmaxOp = softmax[key];
  } else {
    softmaxOp = new Softmax(this, _input);
    measure_softmax_cost(softmaxOp);
    softmax[key] = softmaxOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = softmaxOp;
  return ret;
}

Softmax::Softmax(Model* _model, const Tensor& _input)
: OpBase(_input, _model, OP_SOFTMAX)
{
  numOutputs = 1;
  outputs[0] = _input;
  outputs[0].idx = 0;
}

Softmax::~Softmax(void) {}

void Softmax::collect_costs(float& exe_time, float& flops,
                            float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  // cost metrics
  exe_time += runtime;
  flops += outputSize;
  mem_acc += inputSize;
  num_kernels += 1;
  printf("        cost[Softmax]: cost(%.4lf) total_cost(%.4lf)\n",
         runtime, exe_time);
}

// key is (_input)
SoftmaxKey::SoftmaxKey(const Tensor& _input)
{
  int idx = 0;
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(KEY_LENGTH == idx);
}

