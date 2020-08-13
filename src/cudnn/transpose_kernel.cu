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

void transpose3d_bac(Tensor& inputTensor, void* input,
                     Tensor& outputTensor, void* output)
{
  size_t size = sizeof(DATATYPE) * inputTensor.volume();
  DATATYPE* inputHost = (DATATYPE*) malloc(size);
  DATATYPE* outputHost = (DATATYPE*) malloc(size);
  checkCUDA(cudaMemcpy(inputHost, input, size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < inputTensor.dim[0]; i++) {
    for (int j = 0; j < inputTensor.dim[1]; j++) {
      for (int k = 0; k < inputTensor.dim[2]; k++) {
        int outputOffset = j * outputTensor.dim[1] * outputTensor.dim[2];
        outputOffset += i * outputTensor.dim[2];
        outputOffset += k;
        int inputOffset = i * inputTensor.dim[1] * inputTensor.dim[2];
        inputOffset += j * inputTensor.dim[2];
        inputOffset += k;
        outputHost[outputOffset] = inputHost[inputOffset];
      }
    }
  }
  checkCUDA(cudaMemcpy((void*) output, (void*) outputHost, size,
                       cudaMemcpyHostToDevice));
  free(inputHost);
  free(outputHost);
}

void transpose3d_acb(Tensor& inputTensor, void* input,
                     Tensor& outputTensor, void* output)
{
  size_t size = sizeof(DATATYPE) * inputTensor.volume();
  DATATYPE* inputHost = (DATATYPE*) malloc(size);
  DATATYPE* outputHost = (DATATYPE*) malloc(size);
  checkCUDA(cudaMemcpy(inputHost, input, size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < inputTensor.dim[0]; i++) {
    for (int j = 0; j < inputTensor.dim[1]; j++) {
      for (int k = 0; k < inputTensor.dim[2]; k++) {
        int outputOffset = i * outputTensor.dim[1] * outputTensor.dim[2];
        outputOffset += k * outputTensor.dim[2];
        outputOffset += j;
        int inputOffset = i * inputTensor.dim[1] * inputTensor.dim[2];
        inputOffset += j * inputTensor.dim[2];
        inputOffset += k;
        outputHost[outputOffset] = inputHost[inputOffset];
      }
    }
  }
  checkCUDA(cudaMemcpy((void*) output, (void*) outputHost, size,
                       cudaMemcpyHostToDevice));
  free(inputHost);
  free(outputHost);
}

void transpose3d_bca(Tensor& inputTensor, void* input,
                     Tensor& outputTensor, void* output)
{
  size_t size = sizeof(DATATYPE) * inputTensor.volume();
  DATATYPE* inputHost = (DATATYPE*) malloc(size);
  DATATYPE* outputHost = (DATATYPE*) malloc(size);
  checkCUDA(cudaMemcpy(inputHost, input, size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < inputTensor.dim[0]; i++) {
    for (int j = 0; j < inputTensor.dim[1]; j++) {
      for (int k = 0; k < inputTensor.dim[2]; k++) {
        int outputOffset = j * outputTensor.dim[1] * outputTensor.dim[2];
        outputOffset += k * outputTensor.dim[2];
        outputOffset += i;
        int inputOffset = i * inputTensor.dim[1] * inputTensor.dim[2];
        inputOffset += j * inputTensor.dim[2];
        inputOffset += k;
        outputHost[outputOffset] = inputHost[inputOffset];
      }
    }
  }
  checkCUDA(cudaMemcpy((void*) output, (void*) outputHost, size,
                       cudaMemcpyHostToDevice));
  free(inputHost);
  free(outputHost);
}

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
  if (inputs[0].numDim == 2) {
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
  } else if (inputs[0].numDim == 3) {
    if (inputs[0].dim[0] == outputs[0].dim[1] &&
        inputs[0].dim[1] == outputs[0].dim[0] &&
        inputs[0].dim[2] == outputs[0].dim[2]) {
      transpose3d_bac(inputs[0], inputs[0].data_ptr,
                      outputs[0], outputs[0].data_ptr);
    } else if (inputs[0].dim[0] == outputs[0].dim[0] &&
               inputs[0].dim[1] == outputs[0].dim[2] &&
               inputs[0].dim[2] == outputs[0].dim[1]) {
      transpose3d_acb(inputs[0], inputs[0].data_ptr,
                      outputs[0], outputs[0].data_ptr);
    } else if (inputs[0].dim[0] == outputs[0].dim[2] &&
               inputs[0].dim[1] == outputs[0].dim[0] &&
               inputs[0].dim[2] == outputs[0].dim[1]) {
      transpose3d_bca(inputs[0], inputs[0].data_ptr,
                      outputs[0], outputs[0].data_ptr);
    } else {
      printf("Unsupported transpose perm!\n");
      assert(false);
    }
  } else {
    printf("Unsupported transpose size (%d dims)!\n", inputs[0].numDim);
    assert(false);
  }
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_transpose_cost(Transpose* transpose)
{
  // Transpose requires no kernel launch
  transpose->runtime = 0;
}
