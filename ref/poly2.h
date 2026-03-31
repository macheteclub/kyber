#ifndef POLY2_H
#define POLY2_H

#include "poly.h"

void poly_add_avx2(poly *r, const poly *a, const poly *b);
void poly_sub_avx2(poly *r, const poly *a, const poly *b);
void poly_reduce_avx2(poly *r);

#endif