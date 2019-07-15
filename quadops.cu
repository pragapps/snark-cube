#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <cassert>
#include "cgbn/cgbn.h"
#include "cgbn/utility/support.h"

#define TPI 32
#define BITS 768 

#define TPB 128 

const uint64_t INVERSE_64BIT_MNT4 = 0xf2044cfbe45e7fff;
const uint64_t INVERSE_64BIT_MNT6 = 0xc90776e23fffffff;
const uint32_t MNT4_INV32 = 0xe45e7fff;

typedef std::vector<uint8_t*>* vec_ptr_t;

typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> m;
  cgbn_mem_t<BITS> r;
  cgbn_mem_t<BITS> r_lo;
  cgbn_mem_t<BITS> r_hi;
} simple_t;

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, 768> env96_t;

void reduce_wide(mp_limb_t* result, mp_limb_t* num, mp_limb_t* modulus, uint64_t inv, int n) {
        mp_limb_t *res = num;
        for (size_t i = 0; i < n; ++i)
        {
            mp_limb_t k = inv * res[i];
            /* calculate res = res + k * mod * b^i */
            mp_limb_t carryout = mpn_addmul_1(res+i, modulus, n, k);
            carryout = mpn_add_1(res+n+i, res+n+i, n-i, carryout);
            assert(carryout == 0);
        }

        if (mpn_cmp(res+n, modulus, n) >= 0)
        {
            const mp_limb_t borrow = mpn_sub(res+n, res+n, n, modulus, n);
            assert(borrow == 0);
        }

        mpn_copyi(result, res+n, n);
}

__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ __forceinline__ static int32_t fast_propagate_add(const uint32_t carry, uint32_t &x) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c;
    uint64_t sum;
  
    g=__ballot_sync(sync, carry==1);
    p=__ballot_sync(sync, x==0xFFFFFFFF);
 
    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);
    
    x=x+(c!=0);
     
    return sum>>32;   // -(p==0xFFFFFFFF);
  }

__device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

  __device__ __forceinline__ static int32_t resolve_add(const int32_t carry, uint32_t &x) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c;
    uint64_t sum;
  
    c=__shfl_up_sync(sync, carry, 1);
    c=(warp_thread==0) ? 0 : c;
    x=add_cc(x, c);
    c=addc(0, 0);

    g=__ballot_sync(sync, c==1);
    p=__ballot_sync(sync, x==0xFFFFFFFF);

    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);
    x=x+(c!=0);
  
    c=carry+(sum>>32);
    return __shfl_sync(sync, c, 31);
  }

__device__
uint32_t dev_mul_by_const(uint32_t& r, uint32_t a[], uint32_t f) {
  uint32_t carry = 0;
  uint32_t prd, lane = threadIdx.x % TPI;
  prd = madlo_cc(a[lane], f, 0);
  carry=madhic(a[lane], f, 0);
  carry=resolve_add(carry, prd);
  r = prd;
  return carry;
}

// Result is stored in r = a + b. a is of size 2n, b is of size n.
__device__
uint32_t dev_add_ab(uint32_t r[], uint32_t a[], uint32_t b[]) {
  uint32_t lane = threadIdx.x % TPI;
  uint32_t sum, carry;
  sum = add_cc(a[lane], b[lane]);
  carry = addc_cc(0, 0);
  carry = fast_propagate_add(carry, sum);

  r[lane] = sum;
  
  // a[TPI] = a[TPI] + carry + b_msb_carry;
  return carry;
}

__device__
uint32_t dev_add_ab2(uint32_t& a, uint32_t b) {
  uint32_t sum, carry;
  sum = add_cc(a, b);
  carry = addc_cc(0, 0);
  carry = fast_propagate_add(carry, sum);

  a = sum;
  
  // a[TPI] = a[TPI] + carry + b_msb_carry;
  return carry;
}

__device__
uint32_t add_extra_ui32(uint32_t& a, const uint32_t extra, const uint32_t extra_carry) {
  uint32_t sum, carry, result;
  uint32_t group_thread=threadIdx.x & TPI-1;
  sum = add_cc(a, (group_thread==0) ? extra : 0);
  carry = addc_cc(0, (group_thread==0) ? extra_carry : 0);

  // Each time we call fast_propagate_add, we might have to "clear_carry()"
  // to clear extra data when Padding threads are used.
  result=fast_propagate_add(carry, sum);
  a = sum;
}

__device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

 __device__ __forceinline__ static int32_t fast_propagate_sub(const uint32_t carry, uint32_t &x) {
    uint32_t sync=0xFFFFFFFF, warp_thread=threadIdx.x & warpSize-1, lane=1<<warp_thread;
    uint32_t g, p, c;
    uint64_t sum;
  
    g=__ballot_sync(sync, carry==0xFFFFFFFF);
    p=__ballot_sync(sync, x==0);

    sum=(uint64_t)g+(uint64_t)g+(uint64_t)p;
    c=lane&(p^sum);

    x=x-(c!=0);
    return (sum>>32);     // -(p==0xFFFFFFFF);
  }

__device__
int dev_sub(uint32_t& a, uint32_t& b) {
   uint32_t carry = sub_cc(a, b);
   return -fast_propagate_sub(carry, a); 
}

__device__
void mont_mul(uint32_t a[], uint32_t x[], uint32_t y[], uint32_t m[], uint32_t inv, int n) {
  const uint32_t sync=0xFFFFFFFF;
  uint32_t lane = threadIdx.x % TPI;
  uint32_t ui, carry;
  uint32_t temp = 0, temp2 = 0;
  uint32_t temp_carry = 0, temp_carry2 = 0;
  uint32_t my_lane_a;

  for (int i = 0; i < n; i ++) {
     ui = madlo_cc(x[i], y[0], a[0]);
     ui = madlo_cc(ui, inv, 0);
     temp_carry = dev_mul_by_const(temp, y, x[i]);
     temp_carry2 = dev_mul_by_const(temp2, m, ui);

     temp_carry = temp_carry + dev_add_ab2(temp2, temp);
     temp_carry = temp_carry + dev_add_ab2(a[lane], temp2);

     // missing one BIG add.
     add_extra_ui32(a[lane], temp_carry, temp_carry2);
 
     // right shift one limb
     my_lane_a = a[lane];
     a[lane] =__shfl_down_sync(sync, my_lane_a, 1, TPI);
     a[lane] = (lane == (TPI -1)) ? 0 : a[lane];
  }

  // compare and subtract.
  uint32_t dummy_a = a[lane];
  int which = dev_sub(dummy_a, m[lane]);
  a[lane] = (which == -1) ? a[lane] : dummy_a; 
}

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
  
  cgbn_load(bn96bytes_env, a, &(problem_instances[my_instance]).a);
  cgbn_load(bn96bytes_env, b, &(problem_instances[my_instance]).b);
  cgbn_load(bn96bytes_env, m, &(problem_instances[my_instance]).m);

  cgbn_add(bn96bytes_env, res1, a, b);
  if (cgbn_compare(bn96bytes_env, res1, m) >= 0) {
       cgbn_sub(bn96bytes_env, res, res1, m);
    } else {
       cgbn_set(bn96bytes_env, res, res1); 
    }

  cgbn_store(bn96bytes_env, &(problem_instances[my_instance].r), res);
}

__global__
 void mul_wide_kernel(simple_t *problem_instances, uint32_t instance_count) {
  context_t         bn_context;                                 // create a CGBN context
  env96_t         bn96bytes_env(bn_context);                     // construct a bn environment for 1024 bit math
  env96_t::cgbn_t a, b, m;                      // three 1024-bit values (spread across a warp)
  env96_t::cgbn_wide_t mul_wide;
  // uint32_t np0;
  
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=instance_count) return;                         // return if my_instance is not valid
  
  cgbn_load(bn96bytes_env, a, &(problem_instances[my_instance]).a);
  cgbn_load(bn96bytes_env, b, &(problem_instances[my_instance]).b);
  cgbn_load(bn96bytes_env, m, &(problem_instances[my_instance]).m);


  cgbn_mul_wide(bn96bytes_env, mul_wide, a, b);

  cgbn_store(bn96bytes_env, &(problem_instances[my_instance].r_lo), mul_wide._low);
  cgbn_store(bn96bytes_env, &(problem_instances[my_instance].r_hi), mul_wide._high);
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
    std::memcpy((void*)instance_array[i].a._limbs, (const void*) a[i], num_bytes);
    std::memcpy((void*)instance_array[i].m._limbs, (const void*) input_m_base, num_bytes);
  }

  // printf("Copying instances to the GPU ...\n");
  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(simple_t)*num_elements));
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instance_array, sizeof(simple_t)*num_elements, cudaMemcpyHostToDevice));
  
  int tpb = TPB;
  int IPB = TPB/TPI;
  int tpi = TPI;

  uint32_t num_blocks = (num_elements+IPB-1)/IPB;

  mul_const_kernel<<<num_blocks, TPB>>>(gpuInstances, num_elements, 13);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(simple_t)*num_elements, cudaMemcpyDeviceToHost));

  std::vector<uint8_t*>* res_vector = new std::vector<uint8_t*>();
  for (int i = 0; i < num_elements; i ++) {
     uint8_t* result = (uint8_t*) malloc(num_bytes * sizeof(uint8_t));
     std::memcpy((void*)result, (const void*)instance_array[i].r._limbs, num_bytes);
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
    std::memcpy((void*)instance_array[i].a._limbs, (const void*) a[i], num_bytes);
    std::memcpy((void*)instance_array[i].b._limbs, (const void*) b[i], num_bytes);
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
     std::memcpy((void*)result, (const void*)instance_array[i].r._limbs, num_bytes);
     res_vector->emplace_back(result);
  }

  free(instance_array);
  cudaFree(gpuInstances);
  return res_vector;
}

std::vector<uint8_t*>* mycgbn_montmul(std::vector<uint8_t*> a, std::vector<uint8_t*> b, uint8_t* input_m_base, int num_bytes, uint64_t inv) {
  int num_elements = a.size();

  simple_t *gpuInstances;
  simple_t* instance_array = (simple_t*) malloc(sizeof(simple_t) * num_elements);
  cgbn_error_report_t *report;

  // create a cgbn_error_report for CGBN to report back errors
  NEW_CUDA_CHECK(cgbn_error_report_alloc(&report));
  for (int i = 0; i < num_elements; i ++) {
    std::memcpy((void*)instance_array[i].a._limbs, (const void*) a[i], num_bytes);
    std::memcpy((void*)instance_array[i].b._limbs, (const void*) b[i], num_bytes);
    std::memcpy((void*)instance_array[i].m._limbs, (const void*) input_m_base, num_bytes);
  }

  NEW_CUDA_CHECK(cudaSetDevice(0));
  NEW_CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(simple_t)*num_elements));
  NEW_CUDA_CHECK(cudaMemcpy(gpuInstances, instance_array, sizeof(simple_t)*num_elements, cudaMemcpyHostToDevice));
  
  int tpb = TPB;
  int IPB = TPB/TPI;
  int tpi = TPI;

  uint32_t num_blocks = (num_elements+IPB-1)/IPB;

  mul_wide_kernel<<<num_blocks, TPB>>>(gpuInstances, num_elements);
  NEW_CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  NEW_CUDA_CHECK(cudaMemcpy(instance_array, gpuInstances, sizeof(simple_t)*num_elements, cudaMemcpyDeviceToHost));

  int num_limbs = num_bytes / 8;
  mp_limb_t* num = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs * 2);
  mp_limb_t* modulus = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs);
  std::memcpy((void*) modulus, (const void*) instance_array->m._limbs, num_bytes);

  std::vector<uint8_t*>* res_vector = new std::vector<uint8_t*>();
  for (int i = 0; i < num_elements; i ++) {
    // Reduce
    std::memcpy((void*)num, (const void*)instance_array[i].r_lo._limbs, num_bytes);
    std::memcpy((void*) (num + num_limbs), (const void*)instance_array[i].r_hi._limbs, num_bytes);
    mp_limb_t* fresult = (mp_limb_t*)malloc(sizeof(mp_limb_t) * num_limbs);

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
cgbn_quad_arith(std::vector<uint8_t*> x0_a0,
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

// Logic: 
// var alpha = fq(11);

// var fq3_mul = (x, y) => {
//   var x0_y0 = fq_mul(x.a0, y.a0);
//   var x0_y1 = fq_mul(x.a0, y.a1);
//   var x0_y2 = fq_mul(x.a0, y.a2);
// 
//   var x1_y0 = fq_mul(x.a1, y.a0);
//   var x1_y1 = fq_mul(x.a1, y.a1);
//   var x1_y2 = fq_mul(x.a1, y.a2);
// 
//   var x2_y0 = fq_mul(x.a2, y.a0);
//   var x2_y1 = fq_mul(x.a2, y.a1);
//   var x2_y2 = fq_mul(x.a2, y.a2);
// 
//   return {
//     a0: fq_add(x0_y0, fq_mul(alpha, fq_add(x1_y2, x2_y1))),
//     a1: fq_add(x0_y1, fq_add(x1_y0, fq_mul(alpha, x2_y2))),
//     a2: fq_add(x0_y2, fq_add(x1_y1, x2_y0))
//   };
// };

std::tuple<vec_ptr_t, vec_ptr_t, vec_ptr_t> 
compute_cubex_cuda(std::vector<uint8_t*> x0_a0,
                    std::vector<uint8_t*> x0_a1,
                    std::vector<uint8_t*> x0_a2,
                    std::vector<uint8_t*> y0_a0,
                    std::vector<uint8_t*> y0_a1,
                    std::vector<uint8_t*> y0_a2,
                    uint8_t* input_m_base, int num_bytes, uint64_t inv) {
  int num_elements = x0_a0.size();
  std::vector<uint8_t*>* x0_y0;
  std::vector<uint8_t*>* x0_y1;
  std::vector<uint8_t*>* x0_y2;
  std::vector<uint8_t*>* x1_y0;
  std::vector<uint8_t*>* x1_y1;
  std::vector<uint8_t*>* x1_y2;
  std::vector<uint8_t*>* x2_y0;
  std::vector<uint8_t*>* x2_y1;
  std::vector<uint8_t*>* x2_y2;

  x0_y0 = compute_newcuda(x0_a0, y0_a0, input_m_base, num_bytes, inv);
  x0_y1 = compute_newcuda(x0_a0, y0_a1, input_m_base, num_bytes, inv);
  x0_y2 = compute_newcuda(x0_a0, y0_a2, input_m_base, num_bytes, inv);

  x1_y0 = compute_newcuda(x0_a1, y0_a0, input_m_base, num_bytes, inv);
  x1_y1 = compute_newcuda(x0_a1, y0_a1, input_m_base, num_bytes, inv);
  x1_y2 = compute_newcuda(x0_a1, y0_a2, input_m_base, num_bytes, inv);

  x2_y0 = compute_newcuda(x0_a2, y0_a0, input_m_base, num_bytes, inv);
  x2_y1 = compute_newcuda(x0_a2, y0_a1, input_m_base, num_bytes, inv);
  x2_y2 = compute_newcuda(x0_a2, y0_a2, input_m_base, num_bytes, inv);

  std::vector<uint8_t*>* res_a0_tmp1;
  std::vector<uint8_t*>* res_a0_tmp2;

  vec_ptr_t coeff0, coeff1, coeff2;

  res_a0_tmp1 = compute_addcuda(*x1_y2, *x2_y1, input_m_base, num_bytes);
  res_a0_tmp2 = compute_mul_by11_cuda(*res_a0_tmp1, input_m_base, num_bytes);
  coeff0 = compute_addcuda(*x0_y0, *res_a0_tmp2, input_m_base, num_bytes);

  std::vector<uint8_t*>* res_a1_tmp1;
  std::vector<uint8_t*>* res_a1_tmp2;
  res_a1_tmp1 = compute_mul_by11_cuda(*x2_y2, input_m_base, num_bytes);
  res_a1_tmp2 = compute_addcuda(*x1_y0, *res_a1_tmp1, input_m_base, num_bytes);
  coeff1 = compute_addcuda(*x0_y1, *res_a1_tmp2, input_m_base, num_bytes);

  std::vector<uint8_t*>* res_a2_tmp1;
  res_a2_tmp1 = compute_addcuda(*x1_y1, *x2_y0, input_m_base, num_bytes);
  coeff2 = compute_addcuda(*x0_y2, *res_a2_tmp1, input_m_base, num_bytes);

  freeMem(x0_y0);
  free(x0_y0);
  freeMem(x0_y1);
  free(x0_y1);
  freeMem(x0_y2);
  free(x0_y2);

  freeMem(x1_y0);
  free(x1_y0);
  freeMem(x1_y1);
  free(x1_y1);
  freeMem(x1_y2);
  free(x1_y2);

  freeMem(x2_y0);
  free(x2_y0);
  freeMem(x2_y1);
  free(x2_y1);
  freeMem(x2_y2);
  free(x2_y2);

  freeMem(res_a0_tmp1);
  free(res_a0_tmp1);
  freeMem(res_a0_tmp2);
  free(res_a0_tmp2);

  freeMem(res_a1_tmp1);
  free(res_a1_tmp1);
  freeMem(res_a1_tmp2);
  free(res_a1_tmp2);

  freeMem(res_a2_tmp1);
  free(res_a2_tmp1);

  return std::make_tuple(coeff0, coeff1, coeff2);
}
