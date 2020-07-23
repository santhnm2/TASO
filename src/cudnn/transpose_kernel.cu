/* Copyright 2019 Stanford
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
#include "taso/cuda_helper.h"
using namespace taso;

void Transpose::map(void)
{
  size_t outputSize = outputs[0].volume() * sizeof(DATATYPE);
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Transpose::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Transpose::forward(bool block)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int numDim = outputs[0].numDim;
  int m = inputs[0].dim[numDim-2];
  int n = inputs[0].dim[numDim-1];
  cublasOperation_t transA, transB;
  transA = CUBLAS_OP_T;
  transB = CUBLAS_OP_N;
  checkCUDA(cublasSgeam(model->blas, transA, transB, m, n, &alpha,
                        (float*)inputs[0].data_ptr, n, &beta,
                        (float*)inputs[0].data_ptr, m,
                        (float*)outputs[0].data_ptr, m));
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_transpose_cost(Transpose* transpose)
{
  // Transpose requires no kernel launch
  transpose->runtime = 0;
}
