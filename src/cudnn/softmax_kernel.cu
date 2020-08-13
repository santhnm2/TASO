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

void softmax_kernel(Tensor& inputTensor, void* input,
                    Tensor& outputTensor, void* output, int axis)
{
  // TODO: Implement in CUDA
  // TODO: Support arbitrary axis
  assert(axis == 2);
  size_t size = sizeof(DATATYPE) * inputTensor.volume();
  DATATYPE* inputHost = (DATATYPE*) malloc(size);
  DATATYPE* outputHost = (DATATYPE*) malloc(size);
  checkCUDA(cudaMemcpy(inputHost, input, size, cudaMemcpyDeviceToHost));
  const float epsilon = 0.0001;
  int nHeads = inputTensor.dim[0];
  int nRows = inputTensor.dim[1];
  int nCols = inputTensor.dim[2];
  for (int h = 0; h < nHeads; h++) {
    int hOffset = h * nRows * nCols;
    for (int i = 0; i < nRows; i++) {
      float sum = 0;
      for (int j = hOffset + i * nCols; j < hOffset + (i + 1) * nCols; j++) {
        outputHost[j] = expf(inputHost[j]);
        sum += outputHost[j];
      }
      for (int j = hOffset + i * nCols; j < hOffset + (i + 1) * nCols; j++) {
        outputHost[j] /= (sum + epsilon);
      }
    }
  }
  checkCUDA(cudaMemcpy((void*) output, (void*) outputHost, size,
                       cudaMemcpyHostToDevice));
  free(inputHost);
  free(outputHost);
}

bool Softmax::use_kernel(void) const
{
  if (inputs[0].numDim == 3) {
    return true;
  } else if (inputs[0].numDim == 4) {
    return false;
  } else {
    printf("Softmax not implemented for input tensor with dimension %d\n",
           inputs[0].numDim);
    assert(false);
  }
  return false;
}

void Softmax::map(void)
{
  if (!use_kernel()) {
      checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
      helperSetTensorDescriptor(inputs[0], inputTensor);
      checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
      helperSetTensorDescriptor(outputs[0], outputTensor);
  }
  size_t outputSize = sizeof(DATATYPE);
  for (int i = 0; i < inputs[0].numDim; i++)
    outputSize *= inputs[0].dim[i];
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Softmax::unmap(void)
{
  if (!use_kernel()) {
    checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  }
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Softmax::forward(bool block)
{
  if (use_kernel()) {
      int axis = 2;
      softmax_kernel(inputs[0], inputs[0].data_ptr,
                     outputs[0], outputs[0].data_ptr, axis);
  } else if (inputs[0].numDim == 4) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_FAST;
    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    checkCUDNN(cudnnSoftmaxForward(model->dnn, algo, mode, &alpha, inputTensor,
                                   inputs[0].data_ptr, &beta, outputTensor,
                                   outputs[0].data_ptr));
  }
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_softmax_cost(Softmax* softmax)
{
  // TODO: Measure softmax cost when we have a CUDA implementation
  softmax->runtime = 0;
  /*
  float milliseconds;
  if (softmax->use_kernel()) {
      int axis = 2;
      checkCUDA(cudaDeviceSynchronize());
      checkCUDA(cudaEventRecord(startEvent));
      for (int i = 0; i < REPEAT_TIMES; i++) {
        softmax_kernel(softmax->inputs[0], inputPtr,
                       softmax->outputs[0], outputPtr, axis);
      }
  } else {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      helperSetTensorDescriptor(softmax->inputs[0], inputTensor);
      helperSetTensorDescriptor(softmax->outputs[0], outputTensor);
      cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_FAST;
      cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;
      assert(inputPtr != NULL);
      assert(outputPtr != NULL);
      checkCUDA(cudaDeviceSynchronize());
      checkCUDA(cudaEventRecord(startEvent));
      for (int i = 0; i < REPEAT_TIMES; i++) {
        checkCUDNN(cudnnSoftmaxForward(dnn, algo, mode, &alpha, inputTensor,
                                       inputPtr, &beta, outputTensor,
                                       outputPtr));
      }
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  softmax->runtime = milliseconds / REPEAT_TIMES;
  if (print_cost)
    printf("  measure[Softmax]: i(%d %d %d %d) cost(%.4lf)\n",
           softmax->inputs[0].dim[0], softmax->inputs[0].dim[1],
           softmax->inputs[0].dim[2], softmax->inputs[0].dim[3],
           softmax->runtime);
  */
}
