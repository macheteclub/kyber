#include <stdio.h>          
#include <stdlib.h>         
#include <string.h>         
#include <stdint.h>         
#include <unistd.h>        
// ----------------------------------------------
#include <arpa/inet.h>      // sockaddr_in, htons, inet_pton
#include <sys/socket.h>     // socket, bind, sendto, recvfrom
#include <oqs/oqs.h>        // liboqs ML-KEM

// AEAD + KDF (OpenSSL)
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <errno.h>
#include <sys/time.h>

#define PORT 5000 // 포트
#define MAX_MSG 256 // 최대 길이

// 네트워크/암호 파라미터
#define REQ_STR "REQ"
#define MAGIC 0x4b594252u /* 'KYBR' */
#define AEAD_KEY_LEN 32   /* ChaCha20-Poly1305 key */
#define AEAD_NONCE_LEN 12 /* ChaCha20-Poly1305 nonce */
#define AEAD_TAG_LEN 16

#define HANDSHAKE_RETRIES 5
#define HANDSHAKE_TIMEOUT_MS 700

/* Replay protection (demo-grade):
 * - Keep a small sliding window of seen seq values.
 * - Reject duplicates and very old packets.
 * NOTE: This assumes seq is roughly increasing; for a real system you'd bind
 * it to a proper per-session record counter and handle wrap/epochs.
 */
#define REPLAY_WINDOW 64

typedef struct {
    uint32_t base;   /* lowest seq tracked */
    uint64_t bitmap; /* bit i => (base+i) has been seen */
    int initialized;
} replay_window_t;

static void replay_window_init(replay_window_t *w) {
    w->base = 0;
    w->bitmap = 0;
    w->initialized = 0;
}

/* returns 1 if accepted (new), 0 if replay/too old */
static int replay_window_check_and_mark(replay_window_t *w, uint32_t seq) {
    if (!w->initialized) {
        w->base = seq;
        w->bitmap = 1ULL;
        w->initialized = 1;
        return 1;
    }

    if (seq < w->base) {
        return 0; /* too old */
    }

    uint32_t delta = seq - w->base;
    if (delta >= REPLAY_WINDOW) {
        /* slide window forward so that seq becomes the last element */
        uint32_t shift = delta - (REPLAY_WINDOW - 1);
        if (shift >= REPLAY_WINDOW) {
            w->bitmap = 0;
        } else {
            w->bitmap >>= shift;
        }
        w->base += shift;
        delta = seq - w->base;
    }

    uint64_t mask = 1ULL << delta;
    if (w->bitmap & mask) {
        return 0; /* replay */
    }
    w->bitmap |= mask;
    return 1;
}

/* 패킷 포맷 (DATA)
 * [magic(4)][seq(4)][ct_len(4)][ciphertext(ct_len)]
 * [nonce(12)][pt_len(4)][aead_ct(pt_len)][tag(16)]
 *
 * - aead_ct는 "msg"(평문)만 암호화.
 * - AAD로 (magic|seq|ciphertext|nonce|pt_len) 를 넣어 무결성 보호.
 */

static uint64_t now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000ULL + (uint64_t)tv.tv_usec / 1000ULL;
}

static int set_rcv_timeout_ms(int sockfd, int timeout_ms) {
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    return setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
}

/* KDF: ss -> aead_key (SHA-256 기반) */
static int kdf_sha256(uint8_t out_key[AEAD_KEY_LEN],
                      const uint8_t *ss, size_t ss_len,
                      const uint8_t *ctx, size_t ctx_len)
{
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    if(!mdctx) return 0;
    int ok = 0;
    unsigned int mdlen = 0;
    uint8_t md[EVP_MAX_MD_SIZE];
    if(EVP_DigestInit_ex(mdctx, EVP_sha256(), NULL) != 1) goto done;
    if(EVP_DigestUpdate(mdctx, ss, ss_len) != 1) goto done;
    if(ctx && ctx_len > 0) {
        if(EVP_DigestUpdate(mdctx, ctx, ctx_len) != 1) goto done;
    }
    if(EVP_DigestFinal_ex(mdctx, md, &mdlen) != 1) goto done;
    if(mdlen < AEAD_KEY_LEN) goto done;
    memcpy(out_key, md, AEAD_KEY_LEN);
    ok = 1;
done:
    EVP_MD_CTX_free(mdctx);
    return ok;
}

static int aead_encrypt(uint8_t *out_ct, size_t *out_ct_len,
                        uint8_t out_tag[AEAD_TAG_LEN],
                        const uint8_t key[AEAD_KEY_LEN],
                        const uint8_t nonce[AEAD_NONCE_LEN],
                        const uint8_t *aad, size_t aad_len,
                        const uint8_t *pt, size_t pt_len)
{
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if(!ctx) return 0;
    int ok = 0;
    int len = 0;
    int c_len = 0;

    if(EVP_EncryptInit_ex(ctx, EVP_chacha20_poly1305(), NULL, NULL, NULL) != 1) goto done;
    if(EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_IVLEN, AEAD_NONCE_LEN, NULL) != 1) goto done;
    if(EVP_EncryptInit_ex(ctx, NULL, NULL, key, nonce) != 1) goto done;
    if(aad && aad_len > 0) {
        if(EVP_EncryptUpdate(ctx, NULL, &len, aad, (int)aad_len) != 1) goto done;
    }
    if(EVP_EncryptUpdate(ctx, out_ct, &len, pt, (int)pt_len) != 1) goto done;
    c_len = len;
    if(EVP_EncryptFinal_ex(ctx, out_ct + c_len, &len) != 1) goto done;
    c_len += len;
    if(EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, AEAD_TAG_LEN, out_tag) != 1) goto done;

    *out_ct_len = (size_t)c_len;
    ok = 1;
done:
    EVP_CIPHER_CTX_free(ctx);
    return ok;
}

static int aead_decrypt(uint8_t *out_pt, size_t *out_pt_len,
                        const uint8_t key[AEAD_KEY_LEN],
                        const uint8_t nonce[AEAD_NONCE_LEN],
                        const uint8_t *aad, size_t aad_len,
                        const uint8_t *ct, size_t ct_len,
                        const uint8_t tag[AEAD_TAG_LEN])
{
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if(!ctx) return 0;
    int ok = 0;
    int len = 0;
    int p_len = 0;

    if(EVP_DecryptInit_ex(ctx, EVP_chacha20_poly1305(), NULL, NULL, NULL) != 1) goto done;
    if(EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_IVLEN, AEAD_NONCE_LEN, NULL) != 1) goto done;
    if(EVP_DecryptInit_ex(ctx, NULL, NULL, key, nonce) != 1) goto done;
    if(aad && aad_len > 0) {
        if(EVP_DecryptUpdate(ctx, NULL, &len, aad, (int)aad_len) != 1) goto done;
    }
    if(EVP_DecryptUpdate(ctx, out_pt, &len, ct, (int)ct_len) != 1) goto done;
    p_len = len;
    if(EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_TAG, AEAD_TAG_LEN, (void*)tag) != 1) goto done;
    if(EVP_DecryptFinal_ex(ctx, out_pt + p_len, &len) != 1) goto done;
    p_len += len;

    *out_pt_len = (size_t)p_len;
    ok = 1;
done:
    EVP_CIPHER_CTX_free(ctx);
    return ok;
}

// XOR는 데모용으로는 가능하지만 제출물/실사용엔 비권장(무결성/KDF 없음)

//  receiver ------------------------------------------
void receiver() {
    int sockfd;
    struct sockaddr_in my_addr, peer_addr;
    socklen_t peer_len = sizeof(peer_addr);

    // ML-KEM-768 키 객체 생성
    OQS_KEM *kem = OQS_KEM_new(OQS_KEM_alg_ml_kem_768);
    if (kem == NULL) {
        printf("KEM init failed\n");
        return;
    }

    // 길이는 kem 객체 안에 들어 있음
    uint8_t public_key[kem->length_public_key];
    uint8_t secret_key[kem->length_secret_key];
    uint8_t ciphertext[kem->length_ciphertext];
    uint8_t shared_secret[kem->length_shared_secret];
    uint8_t aead_key[AEAD_KEY_LEN];

    uint8_t buffer[4096];
    uint8_t plain[MAX_MSG + 1];

    replay_window_t rw;
    replay_window_init(&rw);

    // receiver가 공개키/비밀키 생성
    if (OQS_KEM_keypair(kem, public_key, secret_key) != OQS_SUCCESS) {
        printf("KeyGen failed\n");
        OQS_KEM_free(kem);
        return;
    }

    // UDP 소켓 생성
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        OQS_KEM_free(kem);
        return;
    }

    memset(&my_addr, 0, sizeof(my_addr));
    my_addr.sin_family = AF_INET;
    my_addr.sin_addr.s_addr = INADDR_ANY;
    my_addr.sin_port = htons(PORT);

    if (bind(sockfd, (struct sockaddr *)&my_addr, sizeof(my_addr)) < 0) {
        perror("bind");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    printf("[receiver] waiting on port %d...\n", PORT);

    // 핸드셰이크/데이터 수신 시 무한 대기 방지
    if (set_rcv_timeout_ms(sockfd, HANDSHAKE_TIMEOUT_MS) < 0) {
        perror("setsockopt(SO_RCVTIMEO)");
    }

    // sender의 공개키 요청 받기
    int n = recvfrom(sockfd, buffer, sizeof(buffer), 0,
                     (struct sockaddr *)&peer_addr, &peer_len);
    if (n < 0) {
        perror("recvfrom(REQ)");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }
    if (n < (int)strlen(REQ_STR) || memcmp(buffer, REQ_STR, strlen(REQ_STR)) != 0) {
        printf("[receiver] invalid request\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    // 공개키 보내기
    sendto(sockfd, public_key, kem->length_public_key, 0,
           (struct sockaddr *)&peer_addr, peer_len);

    printf("[receiver] public key sent\n");

    // DATA 패킷 받기
    n = recvfrom(sockfd, buffer, sizeof(buffer), 0,
                (struct sockaddr *)&peer_addr, &peer_len);

    if (n < 0) {
        perror("recvfrom(DATA)");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    // 최소 헤더 검사
    if (n < 4 + 4 + 4) {
        printf("packet too short\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    size_t off = 0;
    uint32_t magic_net = 0;
    memcpy(&magic_net, buffer + off, 4); off += 4;
    uint32_t magic = ntohl(magic_net);
    if (magic != MAGIC) {
        printf("[receiver] bad magic\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    uint32_t seq_net = 0;
    memcpy(&seq_net, buffer + off, 4); off += 4;
    uint32_t seq = ntohl(seq_net);

    /* Replay protection: reject duplicates/old seq before spending CPU on crypto */
    if (!replay_window_check_and_mark(&rw, seq)) {
        printf("[receiver] replay/old packet rejected seq=%u\n", seq);
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    uint32_t ct_len_net = 0;
    memcpy(&ct_len_net, buffer + off, 4); off += 4;
    uint32_t ct_len = ntohl(ct_len_net);

    if (ct_len != kem->length_ciphertext || n < (int)(off + ct_len + AEAD_NONCE_LEN + 4 + AEAD_TAG_LEN)) {
        printf("[receiver] invalid ciphertext length\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    memcpy(ciphertext, buffer + off, ct_len);
    off += ct_len;

    uint8_t nonce[AEAD_NONCE_LEN];
    memcpy(nonce, buffer + off, AEAD_NONCE_LEN);
    off += AEAD_NONCE_LEN;

    uint32_t pt_len_net = 0;
    memcpy(&pt_len_net, buffer + off, 4); off += 4;
    uint32_t pt_len = ntohl(pt_len_net);
    if (pt_len > MAX_MSG) {
        printf("[receiver] invalid plaintext length\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }
    if (n < (int)(off + pt_len + AEAD_TAG_LEN)) {
        printf("[receiver] packet truncated\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    const uint8_t *aead_ct = buffer + off;
    off += pt_len;
    const uint8_t *tag = buffer + off;

    // decaps: receiver가 shared secret 복원
    if (OQS_KEM_decaps(kem, shared_secret, ciphertext, secret_key) != OQS_SUCCESS) {
        printf("Decaps failed\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    // shared secret -> AEAD key (KDF)
    const char ctx[] = "kyber-udp-demo-v1";
    if (!kdf_sha256(aead_key, shared_secret, kem->length_shared_secret, (const uint8_t*)ctx, sizeof(ctx)-1)) {
        printf("[receiver] KDF failed\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    // AAD = 헤더+KEM ciphertext+nonce+pt_len
    size_t aad_len = 4 + 4 + 4 + (size_t)ct_len + AEAD_NONCE_LEN + 4;
    if (aad_len > sizeof(buffer)) {
        printf("[receiver] internal aad too large\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    size_t out_pt_len = 0;
    if (!aead_decrypt(plain, &out_pt_len,
                      aead_key, nonce,
                      buffer, aad_len,
                      aead_ct, pt_len,
                      tag)) {
        printf("[receiver] AEAD decrypt failed (tampered/invalid) seq=%u\n", seq);
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    if (out_pt_len > MAX_MSG) out_pt_len = MAX_MSG;
    plain[out_pt_len] = '\0';

    printf("[receiver] ok seq=%u\n", seq);
    printf("[receiver] decrypted message: %s\n", plain);

    close(sockfd);
    OQS_KEM_free(kem);
}


//  sender ------------------------------------------
void sender(const char *ip, const char *msg) {
    int sockfd;
    struct sockaddr_in peer_addr;
    socklen_t peer_len = sizeof(peer_addr);

    // ML-KEM-768 객체 생성
    OQS_KEM *kem = OQS_KEM_new(OQS_KEM_alg_ml_kem_768);
    if (kem == NULL) {
        printf("KEM init failed\n");
        return;
    }

    uint8_t public_key[kem->length_public_key];
    uint8_t ciphertext[kem->length_ciphertext];
    uint8_t shared_secret[kem->length_shared_secret];
    uint8_t aead_key[AEAD_KEY_LEN];

    uint8_t packet[4096];

    const char request[] = REQ_STR;
    uint32_t msg_len = (uint32_t)strlen(msg);

    if (msg_len > MAX_MSG) {
        printf("message too long\n");
        OQS_KEM_free(kem);
        return;
    }

    // UDP 소켓 생성 
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        OQS_KEM_free(kem);
        return;
    }

    memset(&peer_addr, 0, sizeof(peer_addr));
    peer_addr.sin_family = AF_INET;
    peer_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, ip, &peer_addr.sin_addr) <= 0) {
        perror("inet_pton");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    if (set_rcv_timeout_ms(sockfd, HANDSHAKE_TIMEOUT_MS) < 0) {
        perror("setsockopt(SO_RCVTIMEO)");
    }

    // receiver에게 공개키 요청 및 받기
    int got_pk = 0;
    for(int attempt = 0; attempt < HANDSHAKE_RETRIES && !got_pk; attempt++) {
        sendto(sockfd, request, strlen(request), 0,
               (struct sockaddr *)&peer_addr, peer_len);

        int n = recvfrom(sockfd, public_key, kem->length_public_key, 0,
                         (struct sockaddr *)&peer_addr, &peer_len);
        if (n == (int)kem->length_public_key) {
            got_pk = 1;
            break;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            printf("[sender] pk timeout, retry %d/%d\n", attempt+1, HANDSHAKE_RETRIES);
            continue;
        }
        if (n < 0) {
            perror("recvfrom(PK)");
        } else {
            printf("[sender] invalid pk length (%d)\n", n);
        }
    }
    if (!got_pk) {
        printf("[sender] failed to get public key\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    printf("[sender] public key received\n");

    // encaps: sender가 ciphertext + shared secret 생성 
    if (OQS_KEM_encaps(kem, ciphertext, shared_secret, public_key) != OQS_SUCCESS) {
        printf("Encaps failed\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    // shared secret -> AEAD key (KDF)
    const char ctx[] = "kyber-udp-demo-v1";
    if (!kdf_sha256(aead_key, shared_secret, kem->length_shared_secret, (const uint8_t*)ctx, sizeof(ctx)-1)) {
        printf("[sender] KDF failed\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    // nonce 생성
    uint8_t nonce[AEAD_NONCE_LEN];
    if (RAND_bytes(nonce, AEAD_NONCE_LEN) != 1) {
        printf("[sender] RAND_bytes failed\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }

    uint32_t seq = (uint32_t)(now_ms() & 0xffffffffu);

    // 헤더 작성
    size_t off = 0;
    uint32_t magic_net = htonl(MAGIC);
    memcpy(packet + off, &magic_net, 4); off += 4;
    uint32_t seq_net = htonl(seq);
    memcpy(packet + off, &seq_net, 4); off += 4;
    uint32_t ct_len_net = htonl((uint32_t)kem->length_ciphertext);
    memcpy(packet + off, &ct_len_net, 4); off += 4;

    // ciphertext
    memcpy(packet + off, ciphertext, kem->length_ciphertext);
    off += kem->length_ciphertext;

    // nonce
    memcpy(packet + off, nonce, AEAD_NONCE_LEN);
    off += AEAD_NONCE_LEN;

    // pt_len
    uint32_t pt_len_net = htonl(msg_len);
    memcpy(packet + off, &pt_len_net, 4);
    off += 4;

    // AAD는 지금까지의 바이트 전부
    const uint8_t *aad = packet;
    size_t aad_len = off;

    // AEAD encrypt
    uint8_t aead_ct[MAX_MSG];
    size_t aead_ct_len = 0;
    uint8_t tag[AEAD_TAG_LEN];
    if (!aead_encrypt(aead_ct, &aead_ct_len, tag,
                      aead_key, nonce,
                      aad, aad_len,
                      (const uint8_t*)msg, msg_len)) {
        printf("[sender] AEAD encrypt failed\n");
        close(sockfd);
        OQS_KEM_free(kem);
        return;
    }
    if (aead_ct_len != msg_len) {
        printf("[sender] unexpected aead_ct_len\n");
    }

    // 암호문 + tag
    memcpy(packet + off, aead_ct, msg_len);
    off += msg_len;
    memcpy(packet + off, tag, AEAD_TAG_LEN);
    off += AEAD_TAG_LEN;

    // 보내기
    sendto(sockfd, packet, off, 0, (struct sockaddr *)&peer_addr, peer_len);

    printf("[sender] ok seq=%u\n", seq);
    printf("[sender] encrypted message sent (%u bytes)\n", msg_len);

    close(sockfd);
    OQS_KEM_free(kem);
}

//  main ------------------------------------------
int main(int argc, char *argv[]) {
    // - recv: receiver() 실행
    //   1) ML-KEM 키쌍(public/secret) 생성
    //   2) UDP로 sender의 "REQ"(공개키 요청) 수신
    //   3) public_key 전송
    //   4) sender가 보낸 패킷 수신: [ciphertext][msg_len(4)][enc_msg]
    //   5) decaps(ciphertext, secret_key)로 shared_secret 복원
    //   6) shared_secret을 키로 XOR 복호화하여 평문 출력

    // - send <ip> <message>: sender() 실행
    //   1) UDP로 receiver에 "REQ" 송신 → public_key 수신
    //   2) encaps(public_key)로 (ciphertext, shared_secret) 생성
    //   3) shared_secret을 키로 XOR 암호화(enc_msg)
    //   4) 패킷 구성 후 전송: [ciphertext][msg_len(4)][enc_msg]
    if (argc < 2) {
        printf("Usage:\n");
        printf("  %s recv\n", argv[0]);
        printf("  %s send <ip> <message>\n", argv[0]);
        return 1;
    }

    // argv[1]로 recv/send 모드를 선택
    if (strcmp(argv[1], "recv") == 0) {
        // 수신 측: 키 생성 → 공개키 제공 → 패킷 수신/decaps → 복호화
        receiver();
    } else if (strcmp(argv[1], "send") == 0) {
        if (argc < 4) {
            printf("Usage: %s send <ip> <message>\n", argv[0]);
            return 1;
        }
        // 송신 측: 공개키 요청/수신 → encaps → 암호화 → 패킷 전송
        sender(argv[2], argv[3]);
    } else {
        printf("wrong mode\n");
        return 1;
    }

    return 0;
}