#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "ntt.h"
#include "reduce.h"
#include "cbd.h"
#include "symmetric.h"
#include "verify.h"
#include <immintrin.h>

#include "poly2.h"

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

// poly_reduce --------------------------------------------------
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

// NTT AVX2 ─────────────────────────────────────────────────────
/*
 * AVX2 Montgomery 곱셈 (16개 int16_t 동시 처리)
 *
 * 스칼라: montgomery_reduce(a*b) = (a*b - ((a*b mod 2^16)*QINV mod 2^16)*q) >> 16
 *
 * AVX2 유도:
 *   lo    = mullo_epi16(a, b)         // (a*b) mod 2^16
 *   hi    = mulhi_epi16(a, b)         // (a*b) >> 16
 *   t     = mullo_epi16(lo, QINV)     // (lo * QINV) mod 2^16
 *   tq_hi = mulhi_epi16(t, q)         // (t*q) >> 16
 *
 *   result = hi - tq_hi
 *
 * 증명: t*q ≡ lo (mod 2^16) [∵ QINV = q^{-1} mod 2^16]
 *   → (a*b - t*q) 의 하위 16비트는 항상 0
 *   → >> 16 은 단순히 hi - tq_hi
 */
static inline __m256i fqmul_vec(__m256i a, __m256i b)
{
    const __m256i q    = _mm256_set1_epi16(KYBER_Q);
    const __m256i qinv = _mm256_set1_epi16(QINV);
    __m256i lo    = _mm256_mullo_epi16(a, b);
    __m256i hi    = _mm256_mulhi_epi16(a, b);
    __m256i t     = _mm256_mullo_epi16(lo, qinv);
    __m256i tq_hi = _mm256_mulhi_epi16(t, q);
    return _mm256_sub_epi16(hi, tq_hi);
}

/* AVX2 Barrett 환원 (16개 동시, poly_reduce_avx2와 동일 로직) */
static inline __m256i barrett_vec(__m256i a)
{
    const __m256i v = _mm256_set1_epi16(20159);   // round(2^26 / q)
    const __m256i q = _mm256_set1_epi16(KYBER_Q);
    __m256i t = _mm256_srai_epi16(_mm256_mulhi_epi16(a, v), 10);
    return _mm256_sub_epi16(a, _mm256_mullo_epi16(t, q));
}

/* 스칼라 fqmul (ntt.c 의 static 함수를 로컬 복제) */
static inline int16_t fqmul_s(int16_t a, int16_t b)
{
    return montgomery_reduce((int32_t)a * b);
}

/*
 * ntt_avx2 - AVX2 가속 NTT
 *
 * 스테이지별 처리:
 *   len = 128, 64, 32, 16  → 각 그룹이 16개 이상의 나비 쌍을 가짐
 *                             → AVX2로 16쌍씩 동시 처리
 *   len = 8, 4, 2           → 그룹당 나비 < 16, 스칼라 처리
 *
 * 나비 연산 (Cooley-Tukey):
 *   t         = fqmul(zeta, r[j + len])
 *   r[j+len]  = r[j] - t
 *   r[j]      = r[j] + t
 */
static void ntt_avx2(int16_t r[256])
{
    unsigned int len, start, j, k;
    int16_t zeta;
    k = 1;

    /* ── len >= 16: AVX2 ── */
    for(len = 128; len >= 16; len >>= 1) {
        for(start = 0; start < 256; start = j + len) {
            zeta = zetas[k++];
            __m256i vzeta = _mm256_set1_epi16(zeta);
            for(j = start; j < start + len; j += 16) {
                __m256i rj    = _mm256_loadu_si256((const __m256i*)&r[j]);
                __m256i rjlen = _mm256_loadu_si256((const __m256i*)&r[j + len]);
                __m256i t     = fqmul_vec(vzeta, rjlen);
                _mm256_storeu_si256((__m256i*)&r[j],       _mm256_add_epi16(rj, t));
                _mm256_storeu_si256((__m256i*)&r[j + len], _mm256_sub_epi16(rj, t));
            }
        }
    }

    /* ── len < 16: scalar ── */
    for(len = 8; len >= 2; len >>= 1) {
        for(start = 0; start < 256; start = j + len) {
            zeta = zetas[k++];
            for(j = start; j < start + len; j++) {
                int16_t t   = fqmul_s(zeta, r[j + len]);
                r[j + len]  = r[j] - t;
                r[j]        = r[j] + t;
            }
        }
    }
}

/*
 * invntt_avx2 - AVX2 가속 역 NTT
 *
 * 나비 연산 (Gentleman-Sande):
 *   t         = r[j]
 *   r[j]      = barrett_reduce(t + r[j + len])
 *   r[j+len]  = fqmul(zeta, r[j+len] - t)
 */
static void invntt_avx2(int16_t r[256])
{
    unsigned int start, len, j, k;
    int16_t zeta;
    const int16_t f = 1441;    // mont^2 / 128 (스칼라 invntt와 동일)
    k = 127;

    /* ── len < 16: scalar (역순 시작) ── */
    for(len = 2; len <= 8; len <<= 1) {
        for(start = 0; start < 256; start = j + len) {
            zeta = zetas[k--];
            for(j = start; j < start + len; j++) {
                int16_t t   = r[j];
                r[j]        = barrett_reduce(t + r[j + len]);
                r[j + len] -= t;
                r[j + len]  = fqmul_s(zeta, r[j + len]);
            }
        }
    }

    /* ── len >= 16: AVX2 ── */
    for(len = 16; len <= 128; len <<= 1) {
        for(start = 0; start < 256; start = j + len) {
            zeta = zetas[k--];
            __m256i vzeta = _mm256_set1_epi16(zeta);
            for(j = start; j < start + len; j += 16) {
                __m256i rj    = _mm256_loadu_si256((const __m256i*)&r[j]);
                __m256i rjlen = _mm256_loadu_si256((const __m256i*)&r[j + len]);
                __m256i sum   = _mm256_add_epi16(rj, rjlen);
                __m256i diff  = _mm256_sub_epi16(rjlen, rj);
                _mm256_storeu_si256((__m256i*)&r[j],       barrett_vec(sum));
                _mm256_storeu_si256((__m256i*)&r[j + len], fqmul_vec(vzeta, diff));
            }
        }
    }

    /* 마지막 스케일링 (×f = mont^2/128) */
    __m256i vf = _mm256_set1_epi16(f);
    for(j = 0; j < 256; j += 16) {
        __m256i a = _mm256_loadu_si256((const __m256i*)&r[j]);
        _mm256_storeu_si256((__m256i*)&r[j], fqmul_vec(vf, a));
    }
}

/* 공개 래퍼 */
void poly_ntt_avx2(poly *r)
{
    ntt_avx2(r->coeffs);
    poly_reduce_avx2(r);
}

void poly_invntt_avx2(poly *r)
{
    invntt_avx2(r->coeffs);
}

// poly_mul_toomcook_avx2 ----------------------------------------
/*
 * Toom-Cook 2 (Karatsuba)를 이용한 Z_q[x]/(x^256+1) 위의 다항식 곱셈.
 *
 * 분리 (splitting):
 *   a = a0 + x^128 * a1
 *   b = b0 + x^128 * b1
 *
 * 평가점 {0, 1, ∞} 에서의 값:
 *   point 0:   (a0,      b0)
 *   point 1:   (a0+a1,   b0+b1)   <- AVX2 벡터화
 *   point inf: (a1,      b1)
 *
 * 세 번의 schoolbook 곱:
 *   c0   = a0 * b0
 *   c2   = a1 * b1
 *   c1   = (a0+a1) * (b0+b1)
 *
 * 보간 (interpolation):
 *   c_mid = c1 - c0 - c2           <- AVX2 벡터화 (int32)
 *
 * 재조합 (reconstruction) mod (x^256+1):
 *   r[i]   = c0[i] - c2[i] - c_mid[i+128]   for i = 0..126
 *   r[127] = c0[127] - c2[127]               (c_mid[255] = 0)
 *   r[i]   = c0[i] + c_mid[i-128] - c2[i]   for i = 128..254
 *   r[255] = c_mid[127]                      (c0[255] = c2[255] = 0)
 *
 * 오버플로 분석 (입력 계수가 [-(q-1)/2, (q-1)/2] = [-1664, 1664] 일 때):
 *   c0, c2  : 각 계수 최대 128 * 1664^2 ≈ 354M   (int32_t 범위 내)
 *   c1      : 각 계수 최대 128 * 3328^2 ≈ 1418M  (int32_t 범위 내)
 *   c_mid   : 각 계수 최대 1418M + 2*354M ≈ 2126M < INT32_MAX ✓
 *   재조합 전 cmod 중간 환원 필수 (미환원 시 최대 2836M → 오버플로)
 */

/* 128개 계수 다항식의 schoolbook 곱.
 * 입력: int16_t, 범위 [-q/2, q/2] (또는 평가 후 최대 2q).
 * 출력: 255개 int32_t 계수 (환원 전). */
static void schoolbook128(int32_t r[255],
                          const int16_t a[128], const int16_t b[128])
{
    for(int k = 0; k < 255; k++) r[k] = 0;
    for(int i = 0; i < 128; i++)
        for(int j = 0; j < 128; j++)
            r[i + j] += (int32_t)a[i] * b[j];
}

/* int32_t → 중심 int16_t mod q, 결과 ∈ [-(q-1)/2, (q-1)/2]. */
static inline int16_t cmod(int32_t a)
{
    a %= (int32_t)KYBER_Q;
    if(a >  (int32_t)(KYBER_Q - 1) / 2) a -= KYBER_Q;
    if(a < -((int32_t)(KYBER_Q - 1) / 2)) a += KYBER_Q;
    return (int16_t)a;
}

void poly_mul_toomcook_avx2(poly *r, const poly *a, const poly *b)
{
    const int16_t *a0 = &a->coeffs[0];
    const int16_t *a1 = &a->coeffs[128];
    const int16_t *b0 = &b->coeffs[0];
    const int16_t *b1 = &b->coeffs[128];

    /* ── 1. 평가 (Evaluation) ── AVX2, 128 = 8 × 16개 int16_t ── */
    int16_t w1[128], u1[128];   /* point-1 평가 다항식 */
    for(int i = 0; i < 128; i += 16) {
        _mm256_storeu_si256((__m256i*)&w1[i],
            _mm256_add_epi16(
                _mm256_loadu_si256((const __m256i*)&a0[i]),
                _mm256_loadu_si256((const __m256i*)&a1[i])));
        _mm256_storeu_si256((__m256i*)&u1[i],
            _mm256_add_epi16(
                _mm256_loadu_si256((const __m256i*)&b0[i]),
                _mm256_loadu_si256((const __m256i*)&b1[i])));
    }

    /* ── 2. 세 번의 schoolbook 곱 (scalar) ── */
    int32_t c0[255], c1[255], c2[255];
    schoolbook128(c0, a0, b0);      /* c0 = a0 * b0  (point 0) */
    schoolbook128(c2, a1, b1);      /* c2 = a1 * b1  (point inf) */
    schoolbook128(c1, w1, u1);      /* c1 = w1 * u1  (point 1) */

    /* ── 3. 보간 (Interpolation): c1 ← c1 - c0 - c2 ── AVX2, int32 ──
     * 255 = 31×8 + 7 → 31 AVX2 청크(248개) + 나머지 7개 scalar       */
    for(int i = 0; i < 248; i += 8) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)&c0[i]);
        __m256i v1 = _mm256_loadu_si256((const __m256i*)&c1[i]);
        __m256i v2 = _mm256_loadu_si256((const __m256i*)&c2[i]);
        _mm256_storeu_si256((__m256i*)&c1[i],
            _mm256_sub_epi32(_mm256_sub_epi32(v1, v0), v2));
    }
    for(int i = 248; i < 255; i++)
        c1[i] -= c0[i] + c2[i];

    /* ── 4. int32 → centered int16 mod q (재조합 오버플로 방지) ── */
    int16_t r0[255], rm[255], r2[255];
    for(int i = 0; i < 255; i++) {
        r0[i] = cmod(c0[i]);
        rm[i] = cmod(c1[i]);   /* c_mid */
        r2[i] = cmod(c2[i]);
    }

    /* ── 5. 재조합 (Reconstruction) ── AVX2, int16 ──
     *
     * Part A  r[0..126]  = r0[i] - r2[i] - rm[i+128]
     *   7 AVX2 청크(i=0..111) + scalar(i=112..126) + 특수(i=127)
     *
     * Part B  r[128..254] = r0[i] + rm[i-128] - r2[i]
     *   7 AVX2 청크(i=128..239) + scalar(i=240..254) + 특수(i=255)
     *
     * 각 항의 범위: [-1664, 1664] 세 개 합산 → 최대 ±4992 ∈ int16_t ✓ */

    /* Part A — AVX2 (i = 0..111, 7 × 16개) */
    for(int i = 0; i < 112; i += 16) {
        _mm256_storeu_si256((__m256i*)&r->coeffs[i],
            _mm256_sub_epi16(
                _mm256_sub_epi16(
                    _mm256_loadu_si256((const __m256i*)&r0[i]),
                    _mm256_loadu_si256((const __m256i*)&r2[i])),
                _mm256_loadu_si256((const __m256i*)&rm[i + 128])));
    }
    /* Part A — scalar tail (i = 112..126) */
    for(int i = 112; i < 127; i++)
        r->coeffs[i] = r0[i] - r2[i] - rm[i + 128];
    /* Part A — 특수 케이스 i=127: rm[255] = 0 (c_mid의 차수 ≤ 254) */
    r->coeffs[127] = r0[127] - r2[127];

    /* Part B — AVX2 (i = 128..239, 7 × 16개) */
    for(int i = 128; i < 240; i += 16) {
        _mm256_storeu_si256((__m256i*)&r->coeffs[i],
            _mm256_sub_epi16(
                _mm256_add_epi16(
                    _mm256_loadu_si256((const __m256i*)&r0[i]),
                    _mm256_loadu_si256((const __m256i*)&rm[i - 128])),
                _mm256_loadu_si256((const __m256i*)&r2[i])));
    }
    /* Part B — scalar tail (i = 240..254) */
    for(int i = 240; i < 255; i++)
        r->coeffs[i] = r0[i] + rm[i - 128] - r2[i];
    /* Part B — 특수 케이스 i=255: c0[255] = c2[255] = 0 (차수 ≤ 254) */
    r->coeffs[255] = rm[127];

    /* ── 6. 최종 Barrett 환원 (AVX2) ── */
    poly_reduce_avx2(r);
}