/*
 * poly_avx2_wrap.c
 *
 * GNU ld --wrap 을 이용해 poly_add / poly_sub / poly_reduce 를
 * AVX2 버전으로 투명하게 교체하는 래퍼.
 *
 * -Wl,--wrap=poly_add -Wl,--wrap=poly_sub -Wl,--wrap=poly_reduce
 * 링크 옵션을 주면, 바이너리 전체에서 해당 함수 호출이
 * __wrap_poly_* 로 리다이렉트된다.  소스 수정 불필요.
 */

#include "poly.h"
#include "poly2.h"

void __wrap_poly_add(poly *r, const poly *a, const poly *b);
void __wrap_poly_sub(poly *r, const poly *a, const poly *b);
void __wrap_poly_reduce(poly *r);
void __wrap_poly_ntt(poly *r);
void __wrap_poly_invntt_tomont(poly *r);

void __wrap_poly_add(poly *r, const poly *a, const poly *b) {
    poly_add_avx2(r, a, b);
}

void __wrap_poly_sub(poly *r, const poly *a, const poly *b) {
    poly_sub_avx2(r, a, b);
}

void __wrap_poly_reduce(poly *r) {
    poly_reduce_avx2(r);
}

/* poly_ntt / poly_invntt_tomont → AVX2 버전으로 교체 */
void __wrap_poly_ntt(poly *r) {
    poly_ntt_avx2(r);
}

void __wrap_poly_invntt_tomont(poly *r) {
    poly_invntt_avx2(r);
}
