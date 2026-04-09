CC ?= /usr/bin/cc
CFLAGS += -Wall -Wextra -Wpedantic -Wmissing-prototypes -Wredundant-decls \
  -Wshadow -Wpointer-arith -O3 -fomit-frame-pointer -z noexecstack
NISTFLAGS += -Wno-unused-result -O3 -fomit-frame-pointer
RM = /bin/rm

SOURCES = kem.c indcpa.c polyvec.c poly.c ntt.c cbd.c reduce.c verify.c
SOURCESKECCAK = $(SOURCES) fips202.c symmetric-shake.c
HEADERS = params.h kem.h indcpa.h polyvec.h poly.h ntt.h cbd.h reduce.c verify.h symmetric.h
HEADERSKECCAK = $(HEADERS) fips202.h

.PHONY: all speed shared clean bench bench_run

all: test speed shared nistkat

test: \
  test/test_kyber512 \
  test/test_kyber768 \
  test/test_kyber1024 \
  test/test_vectors512 \
  test/test_vectors768 \
  test/test_vectors1024 \

speed: \
  test/test_speed512 \
  test/test_speed768 \
  test/test_speed1024 \

shared: \
  lib/libpqcrystals_kyber512_ref.so \
  lib/libpqcrystals_kyber768_ref.so \
  lib/libpqcrystals_kyber1024_ref.so \
  lib/libpqcrystals_fips202_ref.so \

nistkat: \
	nistkat/PQCgenKAT_kem512 \
	nistkat/PQCgenKAT_kem768 \
	nistkat/PQCgenKAT_kem1024 \


lib/libpqcrystals_fips202_ref.so: fips202.c fips202.h
	mkdir -p lib
	$(CC) -shared -fPIC $(CFLAGS) fips202.c -o $@

lib/libpqcrystals_kyber512_ref.so: $(SOURCES) $(HEADERS) symmetric-shake.c
	mkdir -p lib
	$(CC) -shared -fPIC $(CFLAGS) -DKYBER_K=2 $(SOURCES) symmetric-shake.c -o $@

lib/libpqcrystals_kyber768_ref.so: $(SOURCES) $(HEADERS) symmetric-shake.c
	mkdir -p lib
	$(CC) -shared -fPIC $(CFLAGS) -DKYBER_K=3 $(SOURCES) symmetric-shake.c -o $@

lib/libpqcrystals_kyber1024_ref.so: $(SOURCES) $(HEADERS) symmetric-shake.c
	mkdir -p lib
	$(CC) -shared -fPIC $(CFLAGS) -DKYBER_K=4 $(SOURCES) symmetric-shake.c -o $@

test/test_kyber512: $(SOURCESKECCAK) $(HEADERSKECCAK) test/test_kyber.c randombytes.c
	$(CC) $(CFLAGS) -DKYBER_K=2 $(SOURCESKECCAK) randombytes.c test/test_kyber.c -o $@

test/test_kyber768: $(SOURCESKECCAK) $(HEADERSKECCAK) test/test_kyber.c randombytes.c
	$(CC) $(CFLAGS) -DKYBER_K=3 $(SOURCESKECCAK) randombytes.c test/test_kyber.c -o $@

test/test_kyber1024: $(SOURCESKECCAK) $(HEADERSKECCAK) test/test_kyber.c randombytes.c
	$(CC) $(CFLAGS) -DKYBER_K=4 $(SOURCESKECCAK) randombytes.c test/test_kyber.c -o $@

test/test_vectors512: $(SOURCESKECCAK) $(HEADERSKECCAK) test/test_vectors.c
	$(CC) $(CFLAGS) -DKYBER_K=2 $(SOURCESKECCAK) test/test_vectors.c -o $@

test/test_vectors768: $(SOURCESKECCAK) $(HEADERSKECCAK) test/test_vectors.c
	$(CC) $(CFLAGS) -DKYBER_K=3 $(SOURCESKECCAK) test/test_vectors.c -o $@

test/test_vectors1024: $(SOURCESKECCAK) $(HEADERSKECCAK) test/test_vectors.c
	$(CC) $(CFLAGS) -DKYBER_K=4 $(SOURCESKECCAK) test/test_vectors.c -o $@

test/test_speed512: $(SOURCESKECCAK) $(HEADERSKECCAK) test/cpucycles.h test/cpucycles.c test/speed_print.h test/speed_print.c test/test_speed.c randombytes.c
	$(CC) $(CFLAGS) -DKYBER_K=2 $(SOURCESKECCAK) randombytes.c test/cpucycles.c test/speed_print.c test/test_speed.c -o $@

test/test_speed768: $(SOURCESKECCAK) $(HEADERSKECCAK) test/cpucycles.h test/cpucycles.c test/speed_print.h test/speed_print.c test/test_speed.c randombytes.c
	$(CC) $(CFLAGS) -DKYBER_K=3 $(SOURCESKECCAK) randombytes.c test/cpucycles.c test/speed_print.c test/test_speed.c -o $@

test/test_speed1024: $(SOURCESKECCAK) $(HEADERSKECCAK) test/cpucycles.h test/cpucycles.c test/speed_print.h test/speed_print.c test/test_speed.c randombytes.c
	$(CC) $(CFLAGS) -DKYBER_K=4 $(SOURCESKECCAK) randombytes.c test/cpucycles.c test/speed_print.c test/test_speed.c -o $@

nistkat/PQCgenKAT_kem512: $(SOURCESKECCAK) $(HEADERSKECCAK) nistkat/PQCgenKAT_kem.c nistkat/rng.c nistkat/rng.h
	$(CC) $(NISTFLAGS) -DKYBER_K=2 -o $@ $(SOURCESKECCAK) nistkat/rng.c nistkat/PQCgenKAT_kem.c $(LDFLAGS) -lcrypto

nistkat/PQCgenKAT_kem768: $(SOURCESKECCAK) $(HEADERSKECCAK) nistkat/PQCgenKAT_kem.c nistkat/rng.c nistkat/rng.h
	$(CC) $(NISTFLAGS) -DKYBER_K=3 -o $@ $(SOURCESKECCAK) nistkat/rng.c nistkat/PQCgenKAT_kem.c $(LDFLAGS) -lcrypto

nistkat/PQCgenKAT_kem1024: $(SOURCESKECCAK) $(HEADERSKECCAK) nistkat/PQCgenKAT_kem.c nistkat/rng.c nistkat/rng.h
	$(CC) $(NISTFLAGS) -DKYBER_K=4 -o $@ $(SOURCESKECCAK) nistkat/rng.c nistkat/PQCgenKAT_kem.c $(LDFLAGS) -lcrypto

# ── KEM 벤치마크: ref vs AVX2 ────────────────────────────────────
# 사용법:
#   make bench          → bench_ref, bench_avx2 빌드
#   make bench_run      → 두 바이너리 실행 후 결과 출력
#
# 원리 (--wrap):
#   bench_ref  : poly_add/sub/reduce  →  poly.c  스칼라 구현
#   bench_avx2 : poly_add/sub/reduce  →  poly_avx2_wrap.c (AVX2)
#                GNU ld --wrap 옵션으로 링크 시점 교체, 소스 무수정

BENCH_CFLAGS  = $(CFLAGS) -DKYBER_K=3
BENCH_SOURCES = $(SOURCESKECCAK) randombytes.c bench_kem.c

bench: bench_ref bench_avx2 bench_toomcook

bench_ref: bench_kem.c $(SOURCESKECCAK) $(HEADERSKECCAK) randombytes.c
	$(CC) $(BENCH_CFLAGS) $(BENCH_SOURCES) -o $@

bench_avx2: bench_kem.c poly2.c poly_avx2_wrap.c \
            $(SOURCESKECCAK) $(HEADERSKECCAK) randombytes.c poly2.h
	$(CC) $(BENCH_CFLAGS) -mavx2 -DUSE_AVX2 \
	  $(BENCH_SOURCES) poly2.c poly_avx2_wrap.c \
	  -Wl,--wrap=poly_add \
	  -Wl,--wrap=poly_sub \
	  -Wl,--wrap=poly_reduce \
	  -Wl,--wrap=poly_ntt \
	  -Wl,--wrap=poly_invntt_tomont \
	  -o $@

# KEM-toomcook 변형 (벤치용): basemul까지 Toom-Cook으로 교체
# NOTE: 수학적으로 동일 연산이 아닐 수 있으므로 "속도 비교" 용도.
bench_toomcook: bench_kem.c poly2.c poly_avx2_wrap.c poly_toomcook_wrap.c \
            $(SOURCESKECCAK) $(HEADERSKECCAK) randombytes.c poly2.h
	$(CC) $(BENCH_CFLAGS) -mavx2 -DUSE_AVX2 \
	  $(BENCH_SOURCES) poly2.c poly_avx2_wrap.c poly_toomcook_wrap.c \
	  -Wl,--wrap=poly_add \
	  -Wl,--wrap=poly_sub \
	  -Wl,--wrap=poly_reduce \
	  -Wl,--wrap=poly_ntt \
	  -Wl,--wrap=poly_invntt_tomont \
	  -Wl,--wrap=poly_basemul_montgomery \
	  -o $@

bench_run: bench_ref bench_avx2
	@echo ""; ./bench_ref
	@echo ""; ./bench_avx2
	@echo ""
	@echo "=== Speedup (ref median / avx2 median) ==="
	@ref_kg=$$(./bench_ref   | awk '/KeyGen/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 ref_en=$$(./bench_ref   | awk '/Encaps/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 ref_de=$$(./bench_ref   | awk '/Decaps/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 avx_kg=$$(./bench_avx2  | awk '/KeyGen/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 avx_en=$$(./bench_avx2  | awk '/Encaps/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 avx_de=$$(./bench_avx2  | awk '/Decaps/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 awk "BEGIN { printf \"  KeyGen: %.3fx\\n\", $$ref_kg/$$avx_kg }"; \
	 awk "BEGIN { printf \"  Encaps: %.3fx\\n\", $$ref_en/$$avx_en }"; \
	 awk "BEGIN { printf \"  Decaps: %.3fx\\n\", $$ref_de/$$avx_de }"

# ref vs AVX2(poly add/sub/reduce + ntt/invntt 교체)의 KeyGen/Enc/Dec를 표 형태로 출력
bench_table: bench_ref bench_avx2
	@echo ""; \
	 echo "[bench_table] running ./bench_ref ..." 1>&2; \
	 ref_out=$$(./bench_ref); \
	 echo "[bench_table] running ./bench_avx2 ..." 1>&2; \
	 avx_out=$$(./bench_avx2); \
	 ref_pa=$$(printf "%s\n" "$$ref_out" | awk '/poly_add/    {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 ref_ps=$$(printf "%s\n" "$$ref_out" | awk '/poly_sub/    {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 ref_pr=$$(printf "%s\n" "$$ref_out" | awk '/poly_reduce/ {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 avx_pa=$$(printf "%s\n" "$$avx_out" | awk '/poly_add/    {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 avx_ps=$$(printf "%s\n" "$$avx_out" | awk '/poly_sub/    {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 avx_pr=$$(printf "%s\n" "$$avx_out" | awk '/poly_reduce/ {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 ref_kg=$$(printf "%s\n" "$$ref_out" | awk '/KeyGen/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 ref_en=$$(printf "%s\n" "$$ref_out" | awk '/Encaps/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 ref_de=$$(printf "%s\n" "$$ref_out" | awk '/Decaps/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 avx_kg=$$(printf "%s\n" "$$avx_out" | awk '/KeyGen/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 avx_en=$$(printf "%s\n" "$$avx_out" | awk '/Encaps/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 avx_de=$$(printf "%s\n" "$$avx_out" | awk '/Decaps/  {for(i=1;i<=NF;i++) if($$i=="med") print $$(i+1)}'); \
	 if [ -z "$$ref_pa" ] || [ -z "$$avx_pa" ] || [ -z "$$ref_kg" ] || [ -z "$$avx_kg" ]; then \
	   echo "[bench_table] ERROR: couldn't parse med cycles from bench output" 1>&2; \
	   echo "--- bench_ref ---" 1>&2; printf "%s\n" "$$ref_out" 1>&2; \
	   echo "--- bench_avx2 ---" 1>&2; printf "%s\n" "$$avx_out" 1>&2; \
	   exit 2; \
	 fi; \
	 awk "BEGIN { \
	   printf \"%-8s | %-12s | %-12s | %-25s\\n\", \"Op\", \"Ref(med)\", \"AVX2(med)\", \"SpeedUp / Reduction\"; \
	   printf \"%s\\n\", \"--------------------------------------------------------------------\"; \
	   printf \"%-8s | %-12s | %-12s | %6.2fx, %6.2f%%\\n\", \"add\", $$ref_pa, $$avx_pa, ($$ref_pa/$$avx_pa), (1.0-($$avx_pa/$$ref_pa))*100.0; \
	   printf \"%-8s | %-12s | %-12s | %6.2fx, %6.2f%%\\n\", \"sub\", $$ref_ps, $$avx_ps, ($$ref_ps/$$avx_ps), (1.0-($$avx_ps/$$ref_ps))*100.0; \
	   printf \"%-8s | %-12s | %-12s | %6.2fx, %6.2f%%\\n\", \"reduce\", $$ref_pr, $$avx_pr, ($$ref_pr/$$avx_pr), (1.0-($$avx_pr/$$ref_pr))*100.0; \
	   printf \"%s\\n\", \"--------------------------------------------------------------------\"; \
	   printf \"%-8s | %-12s | %-12s | %6.2fx, %6.2f%%\\n\", \"KeyGen\", $$ref_kg, $$avx_kg, ($$ref_kg/$$avx_kg), (1.0-($$avx_kg/$$ref_kg))*100.0; \
	   printf \"%-8s | %-12s | %-12s | %6.2fx, %6.2f%%\\n\", \"Enc\",    $$ref_en, $$avx_en, ($$ref_en/$$avx_en), (1.0-($$avx_en/$$ref_en))*100.0; \
	   printf \"%-8s | %-12s | %-12s | %6.2fx, %6.2f%%\\n\", \"Dec\",    $$ref_de, $$avx_de, ($$ref_de/$$avx_de), (1.0-($$avx_de/$$ref_de))*100.0; \
	 }"

bench_run_all: bench_ref bench_avx2 bench_toomcook
	@echo ""; ./bench_ref
	@echo ""; ./bench_avx2
	@echo ""; ./bench_toomcook

clean:
	-$(RM) -f *.gcno *.gcda *.lcov *.o *.so
	-$(RM) -f test/test_kyber512
	-$(RM) -f test/test_kyber768
	-$(RM) -f test/test_kyber1024
	-$(RM) -f test/test_vectors512
	-$(RM) -f test/test_vectors768
	-$(RM) -f test/test_vectors1024
	-$(RM) -f test/test_speed512
	-$(RM) -f test/test_speed768
	-$(RM) -f test/test_speed1024
	-$(RM) -f nistkat/PQCgenKAT_kem512
	-$(RM) -f nistkat/PQCgenKAT_kem768
	-$(RM) -f nistkat/PQCgenKAT_kem1024
	-$(RM) -f nistkat/*.req
	-$(RM) -f nistkat/*.rsp
	-$(RM) -rf lib/
	-$(RM) -f bench_ref bench_avx2 bench_toomcook

