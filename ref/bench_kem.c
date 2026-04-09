/*
 * bench_kem.c  -  Kyber 다항식 곱셈 + KEM 벤치마크
 *
 * 비교 대상 (다항식 곱 레벨):
 *   1. NTT scalar   : poly_ntt        + basemul + poly_invntt_tomont  (ref)
 *   2. NTT AVX2     : poly_ntt_avx2   + basemul + poly_invntt_avx2
 *   3. Toom-Cook AVX2: poly_mul_toomcook_avx2
 *
 * 비교 대상 (KEM 레벨):
 *   bench_ref   : 표준 ref (scalar poly_add/sub/reduce)
 *   bench_avx2  : AVX2 poly_add/sub/reduce + NTT AVX2  (--wrap + poly2.c)
 *
 * 빌드:
 *   make bench_ref bench_avx2
 *
 * 실행:
 *   ./bench_ref
 *   ./bench_avx2
 *   make bench_run   ← 둘 다 실행 후 스피드업 계산
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "params.h"
#include "poly.h"
#include "poly2.h"
#include "ntt.h"
#include "reduce.h"
#include "kem.h"
#include "randombytes.h"

/*
 * 사이클 타이머 통일:
 * - Linux x86_64: rdtsc 사용 (요청사항)
 * - 그 외: 기존 cpucycles()로 폴백
 */
#if defined(__x86_64__) && defined(__linux__)
static inline uint64_t rdtsc(void)
{
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
#define BENCH_CYCLES() rdtsc()
#else
#include "test/cpucycles.h"
#define BENCH_CYCLES() cpucycles()
#endif

/* ── 설정 ─────────────────────────────────────────────────────────── */
#define NTESTS   10000
#define NWARMUP  500

/* ── 통계 ─────────────────────────────────────────────────────────── */
static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}

/* t[0..n]: 타임스탬프 배열. 인접 차이의 중앙값/평균 출력. */
static uint64_t print_stat(const char *label, uint64_t *t, int n)
{
    uint64_t *d = malloc((n - 1) * sizeof(uint64_t));
    for(int i = 0; i < n - 1; i++) d[i] = t[i+1] - t[i];
    qsort(d, n-1, sizeof(uint64_t), cmp_u64);
    uint64_t med = d[(n-1)/2];
    uint64_t sum = 0;
    for(int i = 0; i < n-1; i++) sum += d[i];
    printf("%-38s  med %8llu  avg %8llu  cycles\n",
           label, (unsigned long long)med, (unsigned long long)(sum/(n-1)));
    free(d);
    return med;
}

/* ── 랜덤 다항식 (centered) ──────────────────────────────────────── */
static void poly_rand(poly *p)
{
    uint8_t buf[KYBER_N * 2];
    randombytes(buf, sizeof(buf));
    for(int i = 0; i < KYBER_N; i++) {
        uint16_t v = ((uint16_t)buf[2*i] | ((uint16_t)buf[2*i+1] << 8)) % KYBER_Q;
        if(v > (KYBER_Q-1)/2) v -= KYBER_Q;
        p->coeffs[i] = (int16_t)v;
    }
}

/* ── NTT scalar 곱 ───────────────────────────────────────────────── */
static void poly_mul_ntt_scalar(poly *r, const poly *a, const poly *b)
{
    poly ta = *a, tb = *b;
    poly_ntt(&ta);
    poly_ntt(&tb);
    poly_basemul_montgomery(r, &ta, &tb);
    poly_invntt_tomont(r);
}

/* ── NTT AVX2 곱 ─────────────────────────────────────────────────── */
static void poly_mul_ntt_avx2(poly *r, const poly *a, const poly *b)
{
    poly ta = *a, tb = *b;
    poly_ntt_avx2(&ta);
    poly_ntt_avx2(&tb);
    poly_basemul_montgomery(r, &ta, &tb);   /* basemul은 공통 (스칼라) */
    poly_invntt_avx2(r);
}

/* ── 정확성 검증 ─────────────────────────────────────────────────── */
static int poly_eq_modq(const poly *p, const poly *q_poly)
{
    for(int i = 0; i < KYBER_N; i++) {
        int32_t d = (int32_t)p->coeffs[i] - q_poly->coeffs[i];
        if(d % KYBER_Q != 0) return 0;
    }
    return 1;
}

/* ── main ─────────────────────────────────────────────────────────── */
int main(void)
{
    poly a, b, r1, r2, r3;
    uint64_t t[NTESTS + 1];
    uint64_t med_ntt_s, med_ntt_avx, med_tc;

    poly_rand(&a);
    poly_rand(&b);

    /* ══ 1. 다항식 곱 레벨 벤치마크 ══ */
    printf("=== Poly multiplication  (KYBER_K=%d, N=%d) ===\n\n", KYBER_K, KYBER_N);

    /* 워밍업 */
    for(int i = 0; i < NWARMUP; i++) {
        poly_mul_ntt_scalar(&r1, &a, &b);
        poly_mul_ntt_avx2(&r2, &a, &b);
        poly_mul_toomcook_avx2(&r3, &a, &b);
    }

    for(int i = 0; i < NTESTS; i++) {
        t[i] = BENCH_CYCLES(); poly_mul_ntt_scalar(&r1, &a, &b);
    }
    t[NTESTS] = BENCH_CYCLES();
    med_ntt_s = print_stat("NTT scalar (ref)", t, NTESTS+1);

    for(int i = 0; i < NTESTS; i++) {
        t[i] = BENCH_CYCLES(); poly_mul_ntt_avx2(&r2, &a, &b);
    }
    t[NTESTS] = BENCH_CYCLES();
    med_ntt_avx = print_stat("NTT AVX2 (len>=16 vectorized)", t, NTESTS+1);

    for(int i = 0; i < NTESTS; i++) {
        t[i] = BENCH_CYCLES(); poly_mul_toomcook_avx2(&r3, &a, &b);
    }
    t[NTESTS] = BENCH_CYCLES();
    med_tc = print_stat("Toom-Cook 2 AVX2 (Karatsuba)", t, NTESTS+1);

    /* 정확성 */
    printf("\n--- Correctness ---\n");
    printf("  NTT AVX2  vs NTT scalar : %s\n", poly_eq_modq(&r1,&r2) ? "PASS":"FAIL");
    printf("  ToomCook  vs NTT scalar : %s\n", poly_eq_modq(&r1,&r3) ? "PASS":"FAIL");

    /* 배수 */
    printf("\n--- Speedup vs NTT scalar ---\n");
    printf("  NTT AVX2  : %.2fx\n", (double)med_ntt_s / med_ntt_avx);
    printf("  Toom-Cook : %.2fx\n", (double)med_ntt_s / med_tc);

    /* ══ 2. KEM 레벨 벤치마크 ══ */
    printf("\n=== KEM operations (%s) ===\n\n",
#ifdef USE_AVX2
           "AVX2: poly_add/sub/reduce + NTT AVX2"
#else
           "ref: scalar"
#endif
    );

    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t ct[CRYPTO_CIPHERTEXTBYTES];
    uint8_t key_a[CRYPTO_BYTES], key_b[CRYPTO_BYTES];

    for(int i = 0; i < NWARMUP; i++) crypto_kem_keypair(pk, sk);

    for(int i = 0; i < NTESTS; i++) {
        t[i] = BENCH_CYCLES(); crypto_kem_keypair(pk, sk);
    }
    t[NTESTS] = BENCH_CYCLES();
    print_stat("KeyGen", t, NTESTS+1);

    crypto_kem_keypair(pk, sk);
    for(int i = 0; i < NTESTS; i++) {
        t[i] = BENCH_CYCLES(); crypto_kem_enc(ct, key_a, pk);
    }
    t[NTESTS] = BENCH_CYCLES();
    print_stat("Encaps", t, NTESTS+1);

    crypto_kem_enc(ct, key_a, pk);
    for(int i = 0; i < NTESTS; i++) {
        t[i] = BENCH_CYCLES(); crypto_kem_dec(key_b, ct, sk);
    }
    t[NTESTS] = BENCH_CYCLES();
    print_stat("Decaps", t, NTESTS+1);

    crypto_kem_keypair(pk, sk);
    crypto_kem_enc(ct, key_a, pk);
    crypto_kem_dec(key_b, ct, sk);
    printf("\nShared secret match: %s\n",
           memcmp(key_a, key_b, CRYPTO_BYTES) == 0 ? "PASS" : "FAIL");

    return 0;
}
