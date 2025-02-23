# 🖥 컴퓨터 구조 과제 2

## 📌  과제 설명
이 과제에서는 **행렬 곱셈(Matrix Multiplication)** 을  **AVX-512 SIMD 명령어와 Blocking 기법** 을 활용하여 최적화하는 것을 목표로 합니다.  
기존의 일반적인 행렬 곱셈 알고리즘을 개선하여, **CPU 캐시 효율성을 높이고 연산 속도를 향상**시켰습니다.

---

## 📂 파일 구성

| 파일명       | 설명 |
|-------------|-----------------------------------------------------|
| `sem.c`    |  행렬 연산을 위한 기본적인 함수들을 포함|
| `sem.h` | `sem.c`의 헤더 파일 |
| `matrix_multiplication.c`  | 최적화된 행렬 곱셈 연산을 수행하는 핵심 코드 |

-------


## 코드 변경 내용 및 설명

### 기존 코드 (`dgemm` 함수)
```cpp
#include <x86intrin.h>
void dgemm (int n, double* A, double* B, double* C) {
    for (int i = 0; i < n; i+=8)
        for (int j = 0; j < n; ++j) {
            __m512d c0 = _mm512_load_pd(C + i + j * n);
            for (int k = 0; k < n; k++) {
                __m512d bb = _mm512_broadcastsd_pd(_mm_load_sd(B + j * n + k));
                c0 = _mm512_fmadd_pd(_mm512_load_pd(A + n * k + i), bb, c0);
            }
            _mm512_store_pd(C + i + j * n, c0);
        }
}
```
#### 기존 코드의 문제점
- **캐시 활용이 비효율적** → `A`와 `B`의 접근 방식이 최적화되지 않아 **캐시 미스(Cache Miss)가 증가**.
- **SIMD 활용 부족** → SIMD를 사용했지만, 데이터 활용이 최적화되지 않음.
- **Blocking 기법 미사용** → 행렬을 작은 블록 단위로 나누어 연산하지 않기 때문에 메모리 접근 효율성이 낮음.

---

### 변경된 코드 (`OptimizedMatrixMultiplication` 함수)
```cpp
void OptimizedMatrixMultiplication(int **A, int **B, int **C, int size) {
    for (uint32_t j = 0; j < size; j += 16) {
        for (uint32_t i = 0; i < size; i += 16) {
            for (uint32_t k = 0; k < size; k += 16) {
                for (uint32_t x = i; x < i + 16; x += 16) {
                    for (uint32_t y = j; y < j + 16; y++) {
                        __m512i c0 = _mm512_loadu_si512(&C[y][x]);
                        for (uint32_t z = k; z < k + 16; z++) {
                            __m512i b = _mm512_set1_epi32(B[y][z]);
                            __m512i a = _mm512_loadu_si512(&A[z][x]);
                            __m512i c1 = _mm512_mullo_epi32(a, b);
                            c0 = _mm512_add_epi32(c0, c1);
                        }
                        _mm512_storeu_si512(&C[y][x], c0);
                    }
                }
            }
        }
    }
}
```

#### ✅ 변경된 코드의 최적화 포인트
1. **AVX-512 SIMD 활용 증가**
   - 기존보다 더 많은 데이터를 **한 번에 읽어와 연산**하여 성능을 최적화.
   - `__m512i`를 사용하여 512비트(16개의 32비트 정수)를 한 번에 처리.

2. **Blocking 기법 적용 (`BLOCKSIZE = 16`)**
   - 행렬을 **16×16 블록 단위로 나누어 연산**하여 **CPU 캐시 효율성을 극대화**.
   - **한 번 캐시에 올린 데이터를 재사용**하여 메모리 접근 횟수를 줄임.

3. **메모리 접근 최적화**
   - `A[z][x]`와 `B[y][z]`를 **캐시 친화적인 방식으로 로드**하여 **캐시 미스를 줄임**.
   - `_mm512_set1_epi32(B[y][z])`를 사용하여 **B[y][z] 값을 512비트로 확장**하고, `A[z][x]`와 연산.

---

## AVX-512 `__m512i` 관련 함수 설명

✅ **`_mm512_loadu_si512(&C[y][x])`** → `C[y][x] ~ C[y][x+15]`를 한 번에 로드  
✅ **`_mm512_set1_epi32(B[y][z])`** → `B[y][z]` 값을 16개로 복사(브로드캐스트)  
✅ **`_mm512_loadu_si512(&A[z][x])`** → `A[z][x] ~ A[z][x+15]`를 한 번에 로드  
✅ **`_mm512_mullo_epi32(a, b)`** → `a`와 `b`의 각 요소를 정수 곱셈 수행  
✅ **`_mm512_add_epi32(c0, c1)`** → 곱셈 결과를 `c0`에 누적  
✅ **`_mm512_storeu_si512(&C[y][x], c0)`** → `C[y][x] ~ C[y][x+15]`까지 한 번에 저장  

이러한 AVX-512 명령어를 활용하여 **기존보다 최대 16배 빠른 행렬 연산을 수행**할 수 있습니다. 

---
성능 비교


|방식|	SIMD 활용|	캐시 최적화	|연산 속도|
|-------------|-------|-------|-----|
|기존 코드 (dgemm) |	❌ 적음|❌ 없음|	느림|
|Blocking 적용 코드 (OptimizedMatrixMultiplication)|	✅ 많음|	✅ 최적화 최대 16배 향상|

- Blocking을 활용하면 연산 속도가 획기적으로 향상됨
- CPU 캐시를 효율적으로 활용하여 메모리 접근 최적화
- SIMD를 적극 활용하여 병렬 연산을 극대화 
---
##  결론
✔ **기존 행렬 곱셈에서 SIMD와 Blocking 기법을 적용하여 속도를 향상시킴**  
✔ **`j → i → k` 순서로 연산을 수행하여 캐시 미스를 줄이고, 메모리 접근을 최적화**  
✔ **최적화된 코드(`OptimizedMatrixMultiplication`)는 기존 코드보다 **최대 16배 빠름!**   



