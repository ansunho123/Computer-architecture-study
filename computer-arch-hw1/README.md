# 🖥️ 컴퓨터 구조 과제 1 (Computer Architecture 2023)

## 과제 내용
이 프로젝트는 **브랜치 예측기(Branch Predictor)**를 설계하고 구현하는 과제입니다. 기본적으로 1-bit predictor가 제공되었으며, 이를 개선하여 **더 높은 정확도를 갖는 예측기를 개발**하는 것이 목표입니다.

본 프로젝트에서는 **2-bit Saturating Counter**를 활용한 예측기를 구현하였으며, 이를 통해 브랜치 예측 성능을 높였습니다.

---

## **Branch Predictor란?**
**Branch Predictor(브랜치 예측기)**는 CPU가 분기(Branch) 명령어를 실행할 때 **해당 분기가 Taken(실행됨)인지 Not Taken(실행되지 않음)인지 예측**하는 역할을 합니다.

브랜치 예측이 중요한 이유는 **파이프라인 성능 향상**을 위해 미리 예측을 수행하여 불필요한 실행 지연을 줄이기 위함입니다.

### **2-bit Saturating Counter 방식**
1. **각 브랜치 명령어에 대해 2-bit 상태 값을 저장** (0, 1, 2, 3)
2. **값이 2 이상이면 Taken(1)으로 예측, 2 미만이면 Not Taken(0)으로 예측**
3. 예측이 틀릴 경우 상태 값을 증가(또는 감소)시켜 **학습**

---
## 📂 파일 구성

| 파일명       | 설명 |
|-------------|-----------------------------------------------------|
| `predictor_main.c`    | `predictor` 함수의 를 테스트하는 코드 |
| `student_predictor.hpp`  | 브랜치 예측기 클래스 your_own의 정의 **헤더 파일** |
| `student_predictor.cpp` | 2 bit 브렌치 예측기 기능을 직접 구현한 소스 코드 |

-------
## 실행 방법 (Build & Run Instructions)
###  Linux & Mac
```bash
(in target directory)
mkdir build && cd build
cmake ..
make
```
빌드 후 실행 파일이 **release 디렉토리**에 생성됩니다.

실행 결과를 확인하려면 다음 명령어를 사용하세요:
```bash
./predictor_main
```
빌드 결과 삭제:
```bash
make clean
```

---

## 입력 데이터

### **실험을 위해 두가지 유형의 입력 데이터가 제공된다**
1. Synthesized branch flow (합성된 브랜치 흐름)
- `if`문과 `중첩 for` 루프를 포함하는 규칙적인 반복 패턴을 가진 데이터.
- 단순한 제어 흐름을 갖기 때문에 특정 예측기(예: 2-bit predictor)에 유리할 수 있음.

2. SPEC2006
- *SPEC CPU 2006* 벤치마크 데이터를 사용.
- 실제 CPU 워크로드에서 예측기의 성능을 측정하는 데 활용됨.
- 더 복잡한 브랜치 패턴을 포함하여, 보다 일반적인 성능 평가가 가능.

  
----
##  구현한 예측기 설명 (2-bit Predictor)**

### ** 예측 함수 (`get_pred`)**
```cpp
int your_own::get_pred(int pc)
{
  int idx = pc % num_predictor_entry;
  int prediction = pred_arr[idx];
  if (prediction >= 2)
  {
    prediction = 1;
  }
  else
  {
    prediction = 0;
  }
  return prediction;
}
```
- **PC 값을 해시 처리하여 테이블 인덱스 계산** (`pc % num_predictor_entry`)
- 해당 인덱스의 2-bit 값을 확인하여 예측 수행
  - 값이 `2 이상`이면 **Taken(1)**
  - 값이 `1 이하`이면 **Not Taken(0)**

### ** 업데이트 함수 (`update`)**
```cpp
void your_own::update(int pc, int res)
{
  int idx = pc % num_predictor_entry;
  int *arr = pred_arr;
  int prediction = pred_arr[idx];
  if (res == 1)
  {
    if (prediction < 3)
    {
      arr[idx]++;
    }
    else
    {
      arr[idx] = 3;
    }
  }
  else
  {
    if (prediction != 0)
    {
      arr[idx]--;
    }
    else
    {
      arr[idx] = 0;
    }
  }
}
```
- **실제 결과(res)에 따라 카운터 업데이트**
  - **res 1일 경우 → 값을 증가** (최대 `3`까지)
  - **res 0일 경우 → 값을 감소** (최소 `0`까지)
  - 즉, 포화 카운터(Saturating Counter)로 동작하여 **너무 쉽게 변화하지 않도록 설계**

---

## **실행 결과 예시**
```bash
total : 10000 branch, correct : 8500 , ratio : 85.0%
```
- `total` : 전체 브랜치 개수
- `correct` : 맞힌 예측 개수
- `ratio` : 예측 정확도 (%)

---

## **결론 및 개선점**
### **장점**
- **1-bit predictor보다 향상된 정확도 제공**
- 간단한 구현 방식으로도 **비교적 높은 예측 성능을 보임**
- 실제 CPU에서도 사용되는 **2-bit predictor 방식**을 적용

### **개선 가능점**
- **Global History 사용** → 더 정교한 예측 가능
- **Tournament Predictor** → 다양한 방식 조합하여 더 높은 성능
- **Perceptron Branch Predictor** → 신경망 기반 예측 가능

---

## **참고 자료**
- "Computer Architecture: A Quantitative Approach" - John L. Hennessy, David A. Patterson
- SPEC CPU2006 벤치마크 공식 문서
- 과제 제공 문서

---

이 프로젝트는 기본적인 2-bit branch predictor를 구현하는 실습이었으며, 이를 확장하여 다양한 예측 기법을 연구할 수 있습니다.

