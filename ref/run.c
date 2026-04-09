#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "poly.h"
#include "poly2.h"

static inline uint64_t rdtsc(void) {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

#define REPEAT 10000

int main(void) {
    poly a, b, r;
    uint64_t start, end, total;

    // 초기화
    for (int i = 0; i < KYBER_N; i++) {
        a.coeffs[i] = i % 100;
        b.coeffs[i] = i % 50;
    }

    // poly_add scalar
    total = 0;
    for (int i = 0; i < REPEAT; i++) {
        start = rdtsc();
        poly_add(&r, &a, &b);
        end = rdtsc();
        total += end - start;
    }
    printf("poly_add    scalar: %lu\n", total / REPEAT);

    // poly_add AVX2
    total = 0;
    for (int i = 0; i < REPEAT; i++) {
        start = rdtsc();
        poly_add_avx2(&r, &a, &b);
        end = rdtsc();
        total += end - start;
    }
    printf("poly_add    AVX2  : %lu\n", total / REPEAT);

    // poly_sub scalar
    total = 0;
    for (int i = 0; i < REPEAT; i++) {
        start = rdtsc();
        poly_sub(&r, &a, &b);
        end = rdtsc();
        total += end - start;
    }
    printf("poly_sub    scalar: %lu\n", total / REPEAT);

    // poly_sub AVX2
    total = 0;
    for (int i = 0; i < REPEAT; i++) {
        start = rdtsc();
        poly_sub_avx2(&r, &a, &b);
        end = rdtsc();
        total += end - start;
    }
    printf("poly_sub    AVX2  : %lu\n", total / REPEAT);

    // poly_reduce scalar
    total = 0;
    for (int i = 0; i < REPEAT; i++) {
        memcpy(&r, &a, sizeof(poly));
        start = rdtsc();
        poly_reduce(&r);
        end = rdtsc();
        total += end - start;
    }
    printf("poly_reduce scalar: %lu\n", total / REPEAT);

    // poly_reduce AVX2
    total = 0;
    for (int i = 0; i < REPEAT; i++) {
        memcpy(&r, &a, sizeof(poly));
        start = rdtsc();
        poly_reduce_avx2(&r);
        end = rdtsc();
        total += end - start;
    }
    printf("poly_reduce AVX2  : %lu\n", total / REPEAT);

    return 0;
}