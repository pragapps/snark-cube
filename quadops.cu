#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <cassert>
#include "cgbn/cgbn.h"
#include "utility/support.h"

#define TPI 32
#define BITS 768 

#define TPB 128 

uint64_t INVERSE_64BIT_MNT4 = 0xf2044cfbe45e7fff;
uint64_t INVERSE_64BIT_MNT6 = 0xc90776e23fffffff;

typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> m;
  cgbn_mem_t<BITS> r_lo;
  cgbn_mem_t<BITS> r_hi;
} mul_t;

typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> m;
  cgbn_mem_t<BITS> r;
} simple_t;

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, 768> env96_t;

// See CGBN github for information on this code design.
__global__ void mul_const_kernel(simple_t *problem_instances, uint32_t instance_count, uint32_t constant) {
  context_t         bn_context;
  env96_t         bn96bytes_env(bn_context);
  env96_t::cgbn_t a, tmp_r, tmp_r1, tmp_r2, m;  
  env96_t::cgbn_t res, res1;
  
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid
  
  cgbn_load(bn96bytes_env, a, &(problem_instances[my_instance]).a);
  cgbn_load(bn96bytes_env, m, &(problem_instances[my_instance]).m);

  cgbn_set(bn96bytes_env, tmp_r, a); 
  for (int i = 0; i < 12; i ++) {
    cgbn_add(bn96bytes_env, tmp_r1, tmp_r, a);
    if (cgbn_compare(bn96bytes_env, tmp_r1, m) >= 0) {
       cgbn_sub(bn96bytes_env, tmp_r2, tmp_r1, m);
       cgbn_set(bn96bytes_env, tmp_r, tmp_r2); 
    } else {
       cgbn_set(bn96bytes_env, tmp_r, tmp_r1); 
    }
  }

  cgbn_store(bn96bytes_env, &(problem_instances[my_instance].r), tmp_r);
}

__global__ void add_kernel(simple_t *problem_instances, uint32_t instance_count) {
  context_t         bn_context;
  env96_t         bn96bytes_env(bn_context);
  env96_t::cgbn_t a, b, m;                 
  env96_t::cgbn_t res, res1;
  
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  
  if(my_instance>=instance_count) return;
  
  cgbn_load(bn96bytes_env, a, &(problem_instances[my_instance]).x);
  cgbn_load(bn96bytes_env, b, &(problem_instances[my_instance]).y);
  cgbn_load(bn96bytes_env, m, &(problem_instances[my_instance]).m);

  cgbn_add(bn96bytes_env, res1, a, b);
  if (cgbn_compare(bn96bytes_env, res1, m) >= 0) {
       cgbn_sub(bn96bytes_env, res, res1, m);
    } else {
       cgbn_set(bn96bytes_env, res, res1); 
    }

  cgbn_store(bn96bytes_env, &(problem_instances[my_instance].result), res);
}

__global__ void my_kernel(mul_t *problem_instances, uint32_t instance_count) {
  context_t         bn_context;        
  env96_t         bn96bytes_env(bn_context);  
  env96_t::cgbn_t a, b, m;                   
  env96_t::cgbn_wide_t mul_wide;
  
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI; 
  
  if(my_instance>=instance_count) return;
  
  cgbn_load(bn96bytes_env, a, &(problem_instances[my_instance]).x);
  cgbn_load(bn96bytes_env, b, &(problem_instances[my_instance]).y);
  cgbn_load(bn96bytes_env, m, &(problem_instances[my_instance]).m);

  cgbn_mul_wide(bn96bytes_env, mul_wide, a, b);

  cgbn_store(bn96bytes_env, &(problem_instances[my_instance].mul_lo), mul_wide._low);
  cgbn_store(bn96bytes_env, &(problem_instances[my_instance].mul_hi), mul_wide._high);
}


void print_uint8_array(uint8_t* array, int size) {
    for (int i = 0; i < size; i ++) {
        printf("%02x", array[i]);
    }
    printf("\n");
}

std::vector<uint8_t*>* mycgbn_mul_by13(std::vector<uint8_t*> a, uint8_t* input_m_base, int num_bytes) {
  int num_elements = a.size();

  simple_t *gpuInstances;
  simple_t* instance_array = (simple_t*) malloc(sizeof(simple_t) * num_elements);
  cgbn_error_report_t *report;

  // create a cgbn_error_report for CGBN to report back errors
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));
  for (int i = 0; i < num_elements; i ++) {
    std::memcpy((void*)instance_array[i].x._limbs, (const void*) a[i], num_bytes);
    std::memcpy((void*)instance_array[i].m._limbs, (const void*) input_m_base, num_bytes);
  }

  // printf("Copying instances to the GPU ...\n");
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(simple_t)*num_elements));
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instance_array, sizeof(simple_t)*num_elements, cudaMemcpyHostToDevice));
  
  int tpb = TPB;
  // printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;
  int tpi = TPI;
  // printf("\n Threads per instance = %d", tpi);
  // printf("\n Instances per block = %d", IPB);

  uint32_t num_blocks = (num_elements+IPB-1)/IPB;
  // printf("\n Number of blocks = %d", num_blocks);

  mul_by13_kernel<<<num_blocks, TPB>>>(gpuInstances, num_elements);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  // printf("Copying results back to CPU ...\n");
  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(simple_t)*num_elements, cudaMemcpyDeviceToHost));

  std::vector<uint8_t*>* res_vector = new std::vector<uint8_t*>();
  for (int i = 0; i < num_elements; i ++) {
     uint8_t* result = (uint8_t*) malloc(num_bytes * sizeof(uint8_t));
     std::memcpy((void*)result, (const void*)instance_array[i].result._limbs, num_bytes);
     res_vector->emplace_back(result);
  }

  free(instance_array);
  cudaFree(gpuInstances);
  return res_vector;
}

std::vector<uint8_t*>* mycgbn_add(std::vector<uint8_t*> a, std::vector<uint8_t*> b, uint8_t* input_m_base, int num_bytes) {
  int num_elements = a.size();

  simple_t *gpuInstances;
  simple_t* instance_array = (simple_t*) malloc(sizeof(simple_t) * num_elements);
  cgbn_error_report_t *report;

  // create a cgbn_error_report for CGBN to report back errors
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));
  for (int i = 0; i < num_elements; i ++) {
    std::memcpy((void*)instance_array[i].x._limbs, (const void*) a[i], num_bytes);
    std::memcpy((void*)instance_array[i].y._limbs, (const void*) b[i], num_bytes);
    std::memcpy((void*)instance_array[i].m._limbs, (const void*) input_m_base, num_bytes);
  }

  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(simple_t)*num_elements));
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instance_array, sizeof(simple_t)*num_elements, cudaMemcpyHostToDevice));
  
  int tpb = TPB;
  int IPB = TPB/TPI;
  int tpi = TPI;

  uint32_t num_blocks = (num_elements+IPB-1)/IPB;

  add_kernel<<<num_blocks, TPB>>>(gpuInstances, num_elements);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(simple_t)*num_elements, cudaMemcpyDeviceToHost));

  std::vector<uint8_t*>* res_vector = new std::vector<uint8_t*>();
  for (int i = 0; i < num_elements; i ++) {
     uint8_t* result = (uint8_t*) malloc(num_bytes * sizeof(uint8_t));
     std::memcpy((void*)result, (const void*)instance_array[i].result._limbs, num_bytes);
     res_vector->emplace_back(result);
  }

  free(instance_array);
  cudaFree(gpuInstances);
  return res_vector;
}

std::vector<uint8_t*>* mycgbn_montmul(std::vector<uint8_t*> a, std::vector<uint8_t*> b, uint8_t* input_m_base, int num_bytes, uint64_t inv) {
  int num_elements = a.size();

  mul_t *gpuInstances;
  mul_t* instance_array = (mul_t*) malloc(sizeof(mul_t) * num_elements);
  cgbn_error_report_t *report;

  // create a cgbn_error_report for CGBN to report back errors
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));
  for (int i = 0; i < num_elements; i ++) {
    std::memcpy((void*)instance_array[i].x._limbs, (const void*) a[i], num_bytes);
    std::memcpy((void*)instance_array[i].y._limbs, (const void*) b[i], num_bytes);
    std::memcpy((void*)instance_array[i].m._limbs, (const void*) input_m_base, num_bytes);
  }

  printf("Copying instances to the GPU ...\n");
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(mul_t)*num_elements));
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instance_array, sizeof(mul_t)*num_elements, cudaMemcpyHostToDevice));
  
  int tpb = TPB;
  printf("\n Threads per block =%d", tpb);
  int IPB = TPB/TPI;
  int tpi = TPI;
  printf("\n Threads per instance = %d", tpi);
  printf("\n Instances per block = %d", IPB);

  uint32_t num_blocks = (num_elements+IPB-1)/IPB;
  printf("\n Number of blocks = %d", num_blocks);

  my_kernel<<<num_blocks, TPB>>>(gpuInstances, num_elements);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(mul_t)*num_elements, cudaMemcpyDeviceToHost));


  int num_limbs = num_bytes / 8;
  printf("\n Setting num 64 limbs = %d", num_limbs);
  mp_limb_t* num = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs * 2);
  mp_limb_t* modulus = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs);
  std::memcpy((void*) modulus, (const void*) instance_array->m._limbs, num_bytes);

  std::vector<uint8_t*>* res_vector = new std::vector<uint8_t*>();
  for (int i = 0; i < num_elements; i ++) {
    // Reduce
    std::memcpy((void*)num, (const void*)instance_array[i].mul_lo._limbs, num_bytes);
    std::memcpy((void*) (num + num_limbs), (const void*)instance_array[i].mul_hi._limbs, num_bytes);
    mp_limb_t* fresult = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs);
 
    // printf("\n Dumping 64 byte limb wide num [%d]:", i);
    // gmp_printf("%Nx\n", num, num_limbs * 2); 

    reduce_wide(fresult, num, modulus, inv, num_limbs);

    // store the result.
    res_vector->emplace_back((uint8_t*)fresult);
  }
  free(num);
  free(modulus);
  free(instance_array);
  cudaFree(gpuInstances);
  return res_vector;
}

// Logic:  From the CODA webpage.
//  var x0_y0 = fq_mul(x.a0, y.a0);
//  var x1_y1 = fq_mul(x.a1, y.a1);
//  var x1_y0 = fq_mul(x.a1, y.a0);
//  var x0_y1 = fq_mul(x.a0, y.a1);
//  return {
//    a0: fq_add(a0_b0, fq_mul(a1_b1, alpha)),
//    a1: fq_add(a1_b0, a0_b1)
//  };
//

std::pair<std::vector<uint8_t*>, std::vector<uint8_t*> > 
compute_quadex_cuda(std::vector<uint8_t*> x0_a0,
                    std::vector<uint8_t*> x0_a1,
                    std::vector<uint8_t*> y0_a0,
                    std::vector<uint8_t*> y0_a1,
                    uint8_t* mnt_modulus, int num_bytes, uint64_t inv) {
  int num_elements = x0_a0.size();
  std::vector<uint8_t*>* x0_y0;
  std::vector<uint8_t*>* x0_y1;
  std::vector<uint8_t*>* x1_y0;
  std::vector<uint8_t*>* x1_y1;
  std::vector<uint8_t*>* res_a0;
  std::vector<uint8_t*>* res_a1;

  x0_y0 = mycgbn_montmul(x0_a0, y0_a0, mnt_modulus, num_bytes, inv);
  x0_y1 = mycgbn_montmul(x0_a0, y0_a1, mnt_modulus, num_bytes, inv);
  x1_y0 = mycgbn_montmul(x0_a1, y0_a0, mnt_modulus, num_bytes, inv);
  x1_y1 = mycgbn_montmul(x0_a1, y0_a1, mnt_modulus, num_bytes, inv);
  res_a1 = mycgbn_add(*x1_y0, *x0_y1, mnt_modulus, num_bytes);
  res_a0 = mycgbn_mul_by13(*x1_y1, mnt_modulus, num_bytes);
  res_a0 = mycgbn_add(*x0_y0, *res_a0, mnt_modulus, num_bytes);
  std::pair<std::vector<uint8_t*>, std::vector<uint8_t*> > res = std::make_pair(*res_a0, *res_a1);
  return res;
}

