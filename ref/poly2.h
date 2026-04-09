#ifndef POLY2_H
#define POLY2_H

#include "poly.h"

void poly_add_avx2(poly *r, const poly *a, const poly *b);
void poly_sub_avx2(poly *r, const poly *a, const poly *b);
void poly_reduce_avx2(poly *r);

/* NTT AVX2 — len>=16 스테이지 벡터화, len<16 스칼라 */
void poly_ntt_avx2(poly *r);
void poly_invntt_avx2(poly *r);

/*
 * poly_mul_toomcook_avx2 - Toom-Cook 2 (Karatsuba) polynomial multiplication
 * in Z_q[x]/(x^256+1), with AVX2-accelerated evaluation/interpolation steps.
 *
 * Requires: input coefficients in centered form [-(q-1)/2, (q-1)/2].
 *           Call poly_reduce() on inputs first if unsure.
 * Output:   coefficients reduced to [-(q-1)/2, (q-1)/2].
 */
void poly_mul_toomcook_avx2(poly *r, const poly *a, const poly *b);

#endif