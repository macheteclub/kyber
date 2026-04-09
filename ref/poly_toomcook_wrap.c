/*
 * poly_toomcook_wrap.c
 *
 * GNU ld --wrap 을 이용해 KEM 내부에서 쓰이는 다항식 곱(= basemul)을
 * Toom-Cook(정확히는 poly_mul_toomcook_avx2) 기반으로 교체하기 위한 래퍼.
 *
 * 목적:
 *   - KEM keygen/enc/dec 전체를 돌릴 때, 곱셈 경로까지 Toom-Cook을 쓰는 변형을
 *     별도 바이너리로 벤치마크하기 위함.
 *
 * 주의:
 *   - ref 구현에서 곱셈은 원래 NTT 도메인에서 poly_basemul_montgomery()를 통해 이뤄집니다.
 *   - Toom-Cook은 time-domain 곱셈이므로, 단순 교체는 "동일한 수학적 연산"이 아닙니다.
 *   - 벤치 목적(속도 비교)으로만 사용하고, 기능/정확성 검증은 별도 확인이 필요합니다.
 */

#include <string.h>

#include "params.h"
#include "poly.h"

/* poly2.c에 구현되어 있음 (현재는 bench_kem.c에서 직접 링크됨) */
void poly_mul_toomcook_avx2(poly *r, const poly *a, const poly *b);

/*
 * __wrap_poly_basemul_montgomery:
 * 기존: r = a (*) b  (NTT domain에서 basemul)
 * 변경: r = a (*) b  (time domain에서 toomcook)
 */
void __wrap_poly_basemul_montgomery(poly *r, const poly *a, const poly *b)
{
    poly_mul_toomcook_avx2(r, a, b);

    /*
     * 기존 basemul의 출력은 이후 invntt_tomont에서 기대하는 형태(도메인/스케일)가 달라서
     * 결과가 깨질 수 있습니다. 벤치 시에는 "속도 측정용"으로만 사용하세요.
     */
}
