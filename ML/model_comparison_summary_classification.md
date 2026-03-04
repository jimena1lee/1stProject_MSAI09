# 어린이 보호구역 사고 발생여부 이진분류 모델 비교 결과 요약

> **데이터**: `seongnam_train.csv` (성남시 어린이 보호구역)
> **목표**: 사고 발생 여부(발생건수 > 0)를 이진 분류하는 최적 모델 탐색
> **비교 모델**: Logistic Regression, SVM, Random Forest, XGBoost, KNN

---

## 1. 데이터 기초 탐색 (STEP 0)

### 데이터 구성

| 항목 | 내용 |
|---|---|
| 원본 파일 | `seongnam_train.csv` |
| 타겟 변수 | `발생건수 > 0` → 1 (발생), `발생건수 == 0` → 0 (미발생) |
| 제외 컬럼 | 시설물명(ID), 위도, 경도 (비스케일링 컬럼) |
| 피처 수 | 13개 (StandardScaling 완료된 수치형 컬럼 전부 사용) |

### 사용 피처 목록

| 구분 | 피처 |
|---|---|
| 도로 안전시설 | 도로안전표지, 도로적색표면, 무단횡단방지펜스, 보호구역표지판, 옐로카펫, 횡단보도 |
| 단속·감시 | 무인교통단속카메라, 생활안전CCTV |
| 교통 인프라 | 신호등 |
| 인구 특성 | 총인구수, 어린이 총인구, 어린이 비율(%) |
| 복합 지수 | structure_risk |

> 모든 피처는 `StandardScaler` 전처리 완료 상태로 입력됨

### 클래스 분포

| 클래스 | 의미 | 예상 비율 |
|---|---|---|
| 0 (미발생) | 발생건수 == 0 | 약 50~60% |
| 1 (발생) | 발생건수 > 0 | 약 40~50% |

> 데이터 실행 후 실제 클래스 불균형 정도 확인 필요 — 클래스1 F1을 주요 지표로 사용하는 이유

---

## 2. 데이터 분할 (STEP 1)

**Stratified Split** (random_state=42) — 클래스 비율 유지

| 셋 | 비율 | 역할 |
|---|---|---|
| Train | 70% | 모델 학습 |
| Validation | 15% | 모델 선택 (하이퍼파라미터 없이 비교 기준) |
| Test | 15% | 최종 성능 측정 (최적 모델 1개만) |

> Stratified Split 적용으로 세 셋의 클래스 0/1 비율이 원본과 동일하게 유지됨

---

## 3. 비교 모델 (STEP 2)

| 모델 | 클래스명 | 주요 하이퍼파라미터 |
|---|---|---|
| Logistic Regression | `LogisticRegression` | max_iter=1000, random_state=42 |
| SVM | `SVC` | kernel='rbf', probability=True, random_state=42 |
| Random Forest | `RandomForestClassifier` | n_estimators=100, max_depth=5, random_state=42 |
| XGBoost | `XGBClassifier` | n_estimators=100, max_depth=3, eval_metric='logloss' |
| KNN | `KNeighborsClassifier` | n_neighbors=5 |

---

## 4. 학습 및 Validation 평가 (STEP 3)

### 측정 지표 설명

| 지표 | 설명 | 중요도 |
|---|---|---|
| Accuracy | 전체 정확도 | 참고 (클래스 불균형 시 과대 평가 가능) |
| Macro F1 | 클래스 0·1 F1 평균 | 균형 평가 |
| **AUC-ROC** | 분류 임계값 무관한 판별력 | **모델 선택 기준** |
| 클래스1 F1 | 사고 발생 클래스 탐지 성능 | 실무 중요 지표 |
| 클래스1 Precision | 발생 예측 중 실제 발생 비율 | 오탐 방지 |
| 클래스1 Recall | 실제 발생 중 탐지 비율 | 미탐 방지 |

> **모델 선택 기준: Validation AUC-ROC** — 임계값과 무관하게 전반적인 판별력을 반영하며 클래스 불균형에 강건함

### Validation 성능 비교표 (실행 후 채워지는 항목)

| 모델명 | Accuracy | Macro F1 | AUC-ROC | 클래스1 F1 |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| SVM | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| KNN | — | — | — | — |

> 실행 후 `model_comparison_val.csv` 에서 확인 가능

---

## 5. 시각화 (STEP 5)

| 파일 | 내용 | 저장 경로 |
|---|---|---|
| `model_comparison.png` | 4개 지표 (Accuracy·Macro F1·AUC-ROC·클래스1 F1) 모델별 막대그래프 2×2 | `/mnt/user-data/outputs/` |
| `roc_curve.png` | 5개 모델 ROC Curve 겹쳐보기 (각 AUC 범례 표시) | `/mnt/user-data/outputs/` |
| `feature_importance.png` | Random Forest 피처 중요도 수평 막대그래프 (최상위 피처 강조) | `/mnt/user-data/outputs/` |
| `confusion_matrix.png` | 5개 모델 Confusion Matrix 1행 배치 | `/mnt/user-data/outputs/` |

---

## 6. 최적 모델 Test 최종 평가 (STEP 6)

- Val AUC-ROC 1위 모델을 자동 선택하여 **한 번만** Test 셋에 적용
- Val → Test 성능 낙차(overfitting 여부) 확인 가능

### Test Set 지표 (실행 후 채워지는 항목)

| 지표 | Val | Test | 차이 |
|---|---|---|---|
| Accuracy | — | — | — |
| Macro F1 | — | — | — |
| AUC-ROC | — | — | — |
| 클래스1 F1 | — | — | — |

### 산점도 (final_scatter.png)

- X축: Test 샘플 인덱스
- Y축: 클래스1 예측 확률
- 마커: 실제 0 (원형·파랑) / 실제 1 (삼각형·빨강)
- 임계선: 확률 0.5 점선
- → 고확률로 분류된 발생 샘플과 경계 근처 샘플(불확실 구간) 시각적 파악 가능

---

## 7. 파이프라인 구조 요약

```
seongnam_train.csv
        │
        ▼
[STEP 0] 타겟 이진화 · 피처 선택 · 클래스 분포 확인
        │
        ▼
[STEP 1] Stratified Split → Train 70% / Val 15% / Test 15%
        │
        ▼
[STEP 2] 5개 모델 정의
        │
        ▼
[STEP 3] 5개 모델 학습(Train) → Val 예측 → 6개 지표 저장
        │
        ▼
[STEP 4] Val 비교표 출력 → AUC-ROC 기준 최적 모델 선정
        │
        ▼
[STEP 5] 4종 시각화 저장 (막대그래프·ROC·피처중요도·CM)
        │
        ▼
[STEP 6] 최적 모델 → Test 최종 평가 + 산점도 저장
        │
        ▼
[STEP 7] model_comparison_val.csv / model_final_test.csv 저장
```

---

## 8. 이진분류 vs 수치 예측 비교

| 항목 | 이진분류 (본 파일) | 수치 예측 (numerical_label) |
|---|---|---|
| 타겟 | 발생 여부 (0/1) | 발생건수 (0, 1, 2, ...) |
| 주요 지표 | AUC-ROC, F1 | MAE, RMSE, AIC |
| 모델군 | LR, SVM, RF, XGBoost, KNN | 선형회귀, RF분류, Poisson, ZIP, ZINB |
| 활용 | 위험 지점 스크리닝 | 사고 규모 예측 |
| 한계 | 발생 건수를 알 수 없음 | 소표본에서 모수 추정 불안정 |

> **두 접근의 상호 보완**: 이진분류로 발생 여부를 선별한 뒤, 발생 예측 지점에 한해 수치 예측 모델을 적용하는 2단계 파이프라인 구성 가능

---

## 9. 생성 파일 목록

### 이미지 (`/mnt/user-data/outputs/`)

| 파일 | 내용 |
|---|---|
| `model_comparison.png` | 4개 지표 모델별 막대그래프 (2×2) |
| `roc_curve.png` | 5개 모델 ROC Curve 비교 |
| `feature_importance.png` | Random Forest 피처 중요도 |
| `confusion_matrix.png` | 5개 모델 Confusion Matrix 1행 |
| `final_scatter.png` | 최적 모델 실제값 vs 예측확률 산점도 |

### 데이터 (`/mnt/user-data/outputs/`)

| 파일 | 내용 |
|---|---|
| `model_comparison_val.csv` | 5개 모델 Validation 지표 비교표 |
| `model_final_test.csv` | 최적 모델 Test 최종 결과 |

---

## 10. 주요 설계 원칙

- **try-except 전처리**: 개별 모델 오류가 전체 파이프라인을 중단시키지 않음
- **정보 누수 방지**: Test 셋은 Val 기준 최적 모델 선정 후 단 1회만 사용
- **자동 경로 탐색**: 노트북 실행 위치에 무관하게 데이터 파일 자동 탐색
- **한글 폰트 자동 설정**: Windows(Malgun Gothic) / Mac(AppleGothic) / Linux(NanumGothic)
- **재현성 보장**: 모든 확률적 모델에 `random_state=42` 고정
