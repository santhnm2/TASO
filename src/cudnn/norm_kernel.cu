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

#include "taso/cuda_helper.h"
#include "taso/cuda_helper.h"
using namespace taso;

float Graph::norm(const float* x, const float* y, int volume)
{
  // Allocate memory and initialize parameters
  float result;
  float* z;
  int n = volume / sizeof(float);
  checkCUDA(cudaMalloc(&z, volume));
  float alpha = -1.0f;

  // Copy y into new memory as axpy is in-place
  checkCUDA(cudaMemcpy(z, y, volume, cudaMemcpyDeviceToDevice));

  // Compute difference between inputs
  checkCUDA(cublasSaxpy(model->blas, n, &alpha, x, 1, z, 1));

  // Compute norm of difference
  checkCUDA(cublasSnrm2(model->blas, n, z, 1, &result));

  // Wait for result and clean up
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaFree(z));
  return result;
}
