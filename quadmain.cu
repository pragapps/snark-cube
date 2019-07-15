#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>

// Our main implementation that uses CGBN exclusively is in quadops.cu file.
#include "quadops.cu"

// Names inherited from cuda-fixnum.
const unsigned int bytes_per_elem = 128;
const unsigned int io_bytes_per_elem = 96;

using namespace std;

// Print routine inherited from cuda fixnum.
template< int fn_bytes, typename fixnum_array >
void print_fixnum_array(fixnum_array* res, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t local_results[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results[i] = 0;
    }
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);

    for (int i = 0; i < lrl; i++) {
      printf("%i ", local_results[i]);
    }
    printf("\n");
}

template< int fn_bytes, typename fixnum_array >
vector<uint8_t*> get_fixnum_array(fixnum_array* res, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t local_results[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results[i] = 0;
    }
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);
    vector<uint8_t*> res_v;
    for (int n = 0; n < nelts; n++) {
      uint8_t* a = (uint8_t*)malloc(fn_bytes*sizeof(uint8_t));
      for (int i = 0; i < fn_bytes; i++) {
        a[i] = local_results[n*fn_bytes + i];
      }
      res_v.emplace_back(a);
    }
    return res_v;
}

// Some helper routines from libff.
uint8_t* read_mnt_fq(FILE* inputs) {
  uint8_t* buf = (uint8_t*)calloc(bytes_per_elem, sizeof(uint8_t));
  // the input is montgomery representation x * 2^768 whereas cuda-fixnum expects x * 2^1024 so we shift over by (1024-768)/8 bytes
  fread((void*)( buf + (bytes_per_elem - io_bytes_per_elem)), io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  return buf;
}

uint8_t* read_mnt_fq_2(FILE* inputs) {
  uint8_t* buf = (uint8_t*)calloc(bytes_per_elem, sizeof(uint8_t));
  // the input is montgomery representation x * 2^768 whereas cuda-fixnum expects x * 2^1024 so we shift over by (1024-768)/8 bytes
  fread((void*)buf, io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  return buf;
}

void write_mnt_fq(uint8_t* fq, FILE* outputs) {
  fwrite((void *) fq, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

void fprint_uint8_array(FILE* stream, uint8_t* array, int size) {
    for (int i = 0; i < size; i ++) {
        fprintf(stream, "%02x", array[i]);
    }
    fprintf(stream, "\n");
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);

  // mnt4_q
  // inherited from cuda-fixnum.
  uint8_t mnt4_modulus[bytes_per_elem] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  // mnt6_q
  uint8_t mnt6_modulus[bytes_per_elem] = {1,0,0,64,226,118,7,217,79,58,161,15,23,153,160,78,151,87,0,63,188,129,195,214,164,58,153,52,118,249,223,185,54,38,33,41,148,202,235,62,155,169,89,200,40,92,108,178,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  auto inputs = fopen(argv[2], "r");
  auto outputs = fopen(argv[3], "w");
  auto debug_file = fopen(argv[4], "w");

  size_t n;

   while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
    if (elts_read == 0) { break; }

    printf("\n\n Fresh set N = %d\n", n);
    fprintf(debug_file, "\n\n NEW ROUND N = %d", n);
    std::vector<uint8_t*> x0_a0;
    std::vector<uint8_t*> x0_a1;
    for (size_t i = 0; i < n; ++i) {
      x0_a0.emplace_back(read_mnt_fq_2(inputs));
      x0_a1.emplace_back(read_mnt_fq_2(inputs));
       if (i < 5) {
         fprintf(debug_file, "\n Input X0_A0[%d]:", i );
         fprint_uint8_array(debug_file, x0_a0.back(), io_bytes_per_elem);
         fprintf(debug_file, "\n Input X0_A1[%d]:", i );
         fprint_uint8_array(debug_file, x0_a1.back(), io_bytes_per_elem);
       }
    }

    std::vector<uint8_t*> y0_a0;
    std::vector<uint8_t*> y0_a1;
    for (size_t i = 0; i < n; ++i) {
      y0_a0.emplace_back(read_mnt_fq_2(inputs));
      y0_a1.emplace_back(read_mnt_fq_2(inputs));
       if (i < 5) {
         fprintf(debug_file, "\n Input Y1_A0[%d]:", i );
         fprint_uint8_array(debug_file, y0_a0.back(), io_bytes_per_elem);
         fprintf(debug_file, "\n Input Y1_A1[%d]:", i );
         fprint_uint8_array(debug_file, y0_a1.back(), io_bytes_per_elem);
       }
    }
   
    //printf("\n Input 0:\n");
    //print_uint8_array(x0.front(), io_bytes_per_elem);
    //printf("\n Input 1:\n");
    //print_uint8_array(x1.front(), io_bytes_per_elem);

    //std::vector<uint8_t*> res_x = compute_product<bytes_per_elem, u64_fixnum, mul_and_convert>(x0, x1, mnt4_modulus);
    std::pair<std::vector<uint8_t*>, std::vector<uint8_t*> > res
              = cgbn_quad_arith(x0_a0, x0_a1, y0_a0, y0_a1, mnt4_modulus, bytes_per_elem, MNT4_INV32);

    //printf("\n SPECIAL SUM \n");
    //print_uint8_array(res_x.front(), bytes_per_elem);
    //uint8_t* new_res = call_mycuda(x0.front(), x1.front(), mnt4_modulus, io_bytes_per_elem);
    //printf("\n NEW CUDA SUM \n");
    //print_uint8_array(new_res, io_bytes_per_elem);

    // fprintf(debug_file, "\n res.first.size() = %d, second.size() = %d, n = %d", res.first.size(), res.second.size(), n);
    fflush(stdout);
    for (size_t i = 0; i < n; ++i) {
      write_mnt_fq(res.first[i], outputs);
      write_mnt_fq(res.second[i], outputs);
       if (i < 5) {
         fprintf(debug_file, "\n Output[%d]_A0:", i );
         fprint_uint8_array(debug_file, res.first[i], io_bytes_per_elem);
         fprintf(debug_file, "\n Output[%d]_A1:", i );
         fprint_uint8_array(debug_file, res.second[i], io_bytes_per_elem);
       }
    }

    for (size_t i = 0; i < n; ++i) {
      free(x0_a0[i]);
      free(x0_a1[i]);
      free(y0_a0[i]);
      free(y0_a1[i]);
      free(res.first[i]);
      free(res.second[i]);
    }
  }

  return 0;
}

