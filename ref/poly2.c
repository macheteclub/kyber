#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "ntt.h"
#include "reduce.h"
#include "cbd.h"
#include "symmetric.h"
#include "verify.h"
#include <immintrin.h>

// poly_add --------------------------------------------------
void poly_add_avx2(poly *r, const poly *a, const poly *b) {
    for(int i = 0; i < KYBER_N; i += 16) {
        __m256i va = _mm256_loadu_si256((__m256i*)&a->coeffs[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b->coeffs[i]);
        __m256i vc = _mm256_add_epi16(va, vb);
        _mm256_storeu_si256((__m256i*)&r->coeffs[i], vc);
    }
}

// poly_sub --------------------------------------------------
void poly_sub_avx2(poly *r, const poly *a, const poly *b){
    for(int i = 0; i < KYBER_N; i += 16) {
        __m256i va = _mm256_loadu_si256((__m256i*)&a->coeffs[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b->coeffs[i]);
        __m256i vc = _mm256_sub_epi16(va, vb);
        _mm256_storeu_si256((__m256i*)&r->coeffs[i], vc);
    }
}

// poly_reduce (reduce.c)
/*
  scalar:  (int32_t)v * a >> 26
  AVX2:    mulhi_epi16(a, v) >> 10
           = (a * v) >> 16 >> 10
           = (a * v) >> 26
*/
void poly_reduce_avx2(poly *r) {
    // v = ((1<<26) + KYBER_Q/2) / KYBER_Q = 20159
    const __m256i v = _mm256_set1_epi16(20159);
    const __m256i q = _mm256_set1_epi16(KYBER_Q);

    for(int i = 0; i < KYBER_N; i += 16) {
        __m256i a = _mm256_loadu_si256((__m256i*)&r->coeffs[i]);

        // t = (v * a) >> 26
        __m256i t = _mm256_mulhi_epi16(a, v);  // >> 16
        t = _mm256_srai_epi16(t, 10);           // >> 10 → 총 >> 26

        // t *= KYBER_Q, result = a - t
        t = _mm256_mullo_epi16(t, q);
         __m256i result = _mm256_sub_epi16(a, t);

        _mm256_storeu_si256((__m256i*)&r->coeffs[i], result);
    }
  }