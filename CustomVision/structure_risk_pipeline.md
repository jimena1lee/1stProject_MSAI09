# 구조 위험(Structure Risk) 계산 파이프라인

## 전체 흐름 요약

```
로드뷰 이미지 (4004장)
    │
    ├─ [STEP 1] Object Detection (Azure Custom Vision)
    │       → road_width_relative, sidewalk_ratio, parked_density
    │
    ├─ [STEP 2] Classification × 2 (Azure Custom Vision)
    │       → p_wide, p_barrier_yes
    │
    ▼
[STEP 3] 구조 변수 테이블 생성 (4004행 × 5열)
    │
    ▼
[STEP 4] Logistic Regression 학습  ← 사고여부 Y 라벨 사용
    │
    ▼
[STEP 5] Structure Risk (0~1) 출력 → 교차 검증(AUC) → 성남 재검증 → 시설 변수 추가
```

---

## STEP 1. Object Detection — Azure Custom Vision으로 객체 검출

### 목적
이미지에서 **차도, 보도, 차량**의 위치(바운딩박스)를 검출하여 공간 비율 및 밀도 변수를 계산합니다.

### 1-1. Custom Vision 학습 (사전 작업)

- Azure Custom Vision 포털에서 **Object Detection** 프로젝트 생성
- 각 이미지에 아래 3가지 태그로 바운딩박스를 직접 그려 라벨링

| 태그명 | 설명 |
|---|---|
| `차도` | 차량이 주행하는 도로 영역 |
| `보도` | 보행자가 이용하는 인도 영역 |
| `차량` | 이미지 내 주차/정차된 차량 |

- 라벨링 완료 후 학습 → **Prediction API 엔드포인트** 발급

### 1-2. 추론 (Python)

```python
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

client = CustomVisionPredictionClient(endpoint, credentials)
result = client.detect_image(project_id, model_name, image_data)

# 바운딩박스 좌표 추출 (left, top, width, height — 이미지 크기 대비 상대 비율)
for pred in result.predictions:
    print(pred.tag_name, pred.probability, pred.bounding_box)
```

> Custom Vision의 바운딩박스 좌표는 **이미지 크기 대비 0~1 상대 비율**로 반환됩니다.  
> 실제 픽셀 면적으로 변환하려면 `width × height × 이미지_가로 × 이미지_세로`로 계산합니다.

### 1-3. 구조 변수 계산

| 변수명 | 계산식 | 해석 |
|---|---|---|
| `road_width_relative` | 차도 bbox 면적 ÷ 전체 이미지 면적 | 화면에서 도로가 차지하는 상대적 비율 |
| `sidewalk_ratio` | 보도 bbox 면적 ÷ 전체 이미지 면적 | 보행자 보호 공간 확보 정도 |
| `parked_density` | 검출 차량 수 ÷ 차도 영역 면적 | 주정차 밀집도 (시야 방해 가능성) |

```python
img_area = 1262 * 572  # 전체 이미지 면적 (픽셀)

road_bbox_area  = road_w * road_h * 1262 * 572   # 상대 비율 → 픽셀 면적 변환
sidewalk_area   = sw_w * sw_h * 1262 * 572

road_width_relative = road_bbox_area / img_area
sidewalk_ratio      = sidewalk_area  / img_area
parked_density      = num_vehicles   / road_bbox_area
```

---

## STEP 2. Classification — 도로 폭 / 차단시설 분류

### 목적
이미지 전체를 보고 **"넓다/좁다"**, **"차단시설 있음/없음"** 두 가지 기준으로 분류하고 **softmax 확률값**을 추출합니다.

> Object Detection으로 측정 가능한 수치 데이터(도로 폭 cm 등)가 없기 때문에,  
> 추상적 개념을 확률로 표현하는 Classification을 사용합니다.

### 2-1. Custom Vision 학습 (사전 작업)

Classification 프로젝트를 **2개** 별도로 생성합니다 (Multi-class).

| 프로젝트 | 태그 |
|---|---|
| 프로젝트 ① — 도로 폭 | `wide` (2차선 이상), `narrow` (1차선) |
| 프로젝트 ② — 차단시설 | `barrier_yes` (시설 있음), `barrier_no` (시설 없음) |

### 2-2. 추론 (Python)

```python
result = client.classify_image(project_id, model_name, image_data)

for pred in result.predictions:
    print(pred.tag_name, pred.probability)
    # 예시 출력: wide 0.82 / narrow 0.18
    #           barrier_yes 0.65 / barrier_no 0.35
```

### 2-3. 구조 변수 추출

| 변수명 | 추출 방법 | 모델 포함 여부 |
|---|---|---|
| `p_wide` | 프로젝트① `wide` 확률 | ✅ 포함 |
| `p_narrow` | 프로젝트① `narrow` 확률 | ❌ 제외 |
| `p_barrier_yes` | 프로젝트② `barrier_yes` 확률 | ✅ 포함 |
| `p_barrier_no` | 프로젝트② `barrier_no` 확률 | ❌ 제외 |

> **p_wide + p_narrow = 1**, **p_barrier_yes + p_barrier_no = 1**  
> 두 변수를 동시에 투입하면 **완전 선형 종속**이 발생하여 회귀 계수가 불안정해집니다.  
> 따라서 각 쌍에서 하나씩만 모델에 포함합니다.

---

## STEP 3. 구조 변수 테이블 생성

STEP 1, 2를 이미지 4004장 전체에 적용하면 아래와 같은 테이블이 만들어집니다.

| 위도 | 경도 | road_width_relative | sidewalk_ratio | parked_density | p_wide | p_barrier_yes | 사고여부(Y) |
|---|---|---|---|---|---|---|---|
| 37.45 | 127.13 | 0.42 | 0.08 | 1.23 | 0.71 | 0.35 | 1 |
| 37.47 | 127.15 | 0.19 | 0.22 | 0.41 | 0.23 | 0.82 | 0 |

- **Y = 1** : 전국 스쿨존 사고 다발지 (1,742개)
- **Y = 0** : 사고 기록 없는 스쿨존 (경기도 2,931개 중 비사고 지역)

---

## STEP 4. Logistic Regression 학습 — Structure Risk 계산

### 목적
5개 구조 변수를 입력받아 **사고 발생 확률 P(Y=1)** 을 추정합니다.

### 4-1. 수식

**① 위험 점수(z) 계산 — 선형 조합**

```
z = w1·road_width_relative
  + w2·sidewalk_ratio
  + w3·parked_density
  + w4·p_wide
  + w5·p_barrier_yes
  + b
```

각 구조 변수에 학습된 가중치(w)를 곱해 더하면 위험 점수 z가 됩니다.

**② 로그오즈 (Log-Odds)**

```
log(P / (1-P)) = z
```

z는 사고 확률을 로그오즈로 선형 표현한 값입니다.

**③ 시그모이드 변환 — 확률로 변환**

```
Structure Risk = sigmoid(z) = 1 / (1 + exp(-z))
```

z에 시그모이드 함수를 적용하면 값의 범위가 **0~1** 사이 확률로 변환됩니다.

- 0에 가까울수록 → 구조적으로 **안전**
- 1에 가까울수록 → 구조적으로 **위험**

### 4-2. sklearn 구현 방식

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 변수 정규화 (각 변수의 스케일이 달라 반드시 필요)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # X: 5개 구조 변수 (4004행)

# 모델 학습
model = LogisticRegression()
model.fit(X_scaled, y)  # y: 사고 여부 0/1

# Structure Risk 예측 — P(Y=1) 추출
risk_proba = model.predict_proba(X_scaled)[:, 1]
```

> StandardScaler를 사용하는 이유:  
> `road_width_relative`는 0~1, `parked_density`는 0~수십 단위로 스케일이 달라  
> 정규화하지 않으면 가중치 크기가 변수 단위에 의존하게 됩니다.

---

## STEP 5. 모델 검증 — AUC와 교차 검증

### 5-1. AUC란

> **AUC (Area Under the ROC Curve)**  
> 무작위로 사고 지역 1개, 비사고 지역 1개를 뽑았을 때,  
> 모델이 사고 지역의 Structure Risk를 더 높게 줄 확률.
>
> - AUC = 0.5 → 랜덤 수준 (의미 없음)
> - AUC = 1.0 → 완벽한 분류
> - AUC = 0.71 → 71%의 확률로 사고 지역을 올바르게 더 위험하게 평가

### 5-2. 두 가지 교차 검증 방식

| 방식 | 설명 | 사용 이유 |
|---|---|---|
| **Stratified K-Fold** | 각 fold에서 사고/비사고 비율을 동일하게 유지 | 클래스 불균형 보정 |
| **Group K-Fold** | 같은 지역(그룹)의 데이터가 train/test에 동시에 포함되지 않도록 분리 | 지역 내 이미지 유사성으로 인한 **데이터 누수(leakage) 방지** |

```python
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5)
auc_skf = cross_val_score(model, X_scaled, y, cv=skf, scoring='roc_auc')

# Group K-Fold (지역 단위 분리)
gkf = GroupKFold(n_splits=5)
auc_gkf = cross_val_score(model, X_scaled, y,
                           cv=gkf, groups=region_labels, scoring='roc_auc')

print(f"Stratified AUC 평균: {auc_skf.mean():.4f}")
print(f"Group AUC 평균:      {auc_gkf.mean():.4f}")
```

### 5-3. 검증 결과 해석

**전국 범위 (4,673개 지역)**

| 검증 방식 | AUC 평균 |
|---|---|
| Stratified K-Fold | 0.7123 |
| Group K-Fold | 0.7023 |

→ AUC 0.71 — **구조적으로 위험한 지역일수록 사고 확률이 높다**는 가설에 부합.

**정책 활용 성능**

| 지표 | 값 | 의미 |
|---|---|---|
| Top-20% Capture | 0.32 | 위험 상위 20% 지역이 **전체 사고의 32% 설명** |
| Precision@20% | 0.75 | 상위 위험 지역 중 **75%가 실제 사고 지역** |


→ 모델이 선택한 **위험 상위 20% 지역 중 약 75%가 실제 사고 지역**

즉, 모델이 **위험 지역을 비교적 정확하게 선별한다.**

**성남시 (142개 지역) 재검증**

| 검증 방식 | AUC 평균 |
|---|---|
| Stratified K-Fold | 0.63 |

→ AUC 0.63 — **가설 유지.**  
학습에 사용하지 않은 지역에서도 모델이 예측 능력을 유지한다. 다만, 성능이 감소한다. 

### 6. 종합 평가
이 모델은

- 사고 위험 지역을 **중간 수준 정확도로 구분(AUC ≈ 0.71)**
- 지역이 바뀌어도 **비슷한 성능 유지**
- 위험 상위 지역을 **정책적으로 선별 가능**

하지만

- 사고의 약 **1/3 정도만 상위 위험지역에 집중**

따라서

> 이 모델은 **기초 위험 선별 모델**로 사용 가능하며  
> 이후 **시설 변수 및 이미지 위험 변수 추가 모델이 필요하다.**