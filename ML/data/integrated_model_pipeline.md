모델링 계획 수립
데이터 현황 요약
seongnam_scaled.csv 기준으로 최종 데이터는 아래와 같습니다.

Features (11개): 도로안전표지, 도로적색표면, 무단횡단방지펜스, 무인교통단속카메라, 보호구역표지판, 생활안전CCTV, 신호등, 옐로카펫, 횡단보도, 어린이 비율(%), 구조 위험도 점수 (로드뷰 기반 추가 예정)
Label (1개): 발생건수 (0이 대부분인 불균형 분포)


전체 파이프라인 개요
데이터 로드 → EDA → 데이터 분할 → 모델 3종 학습 → 평가 → 비교 → 최종 선택

Step 1. 데이터 로드 및 EDA
목적: 레이블 분포 확인 및 모델 선택 근거 마련

seongnam_scaled.csv 로드
발생건수 분포 시각화 (countplot / histogram)
클래스 비율 계산: 0건 vs 1건 이상의 비율 확인
결측치, 이상치 확인
feature 간 상관관계 히트맵

핵심 확인 사항: 0의 비율이 전체의 몇 %인지 → 모델 전략 결정의 근거가 됨

Step 2. 데이터 전처리 및 분할
목적: 3가지 모델 각각에 맞는 형태로 레이블 변환
모델레이블 형태변환 방법로지스틱 이진 분류0 / 1발생건수 > 0 이면 1, 아니면 0Zero-Inflated원본 정수변환 없음일반 Regression원본 정수 or 연속형변환 없음

train_test_split 으로 8:2 분할 (random_state 고정)
이진 분류용 레이블 별도 생성
분할 후 각 셋의 클래스 비율도 재확인


Step 3-A. 로지스틱 이진 분류
목적: "사고가 발생할 것인가 / 아닌가"를 예측
학습

sklearn.linear_model.LogisticRegression
클래스 불균형 처리를 위해 class_weight='balanced' 옵션 적용

평가지표

Accuracy, Precision, Recall, F1-score
ROC-AUC 곡선
Confusion Matrix

주의점: Accuracy만 보면 0으로만 예측해도 높게 나올 수 있으므로, Recall과 F1을 주 지표로 삼아야 함

Step 3-B. Zero-Inflated Model
목적: 0이 과도하게 많은 카운트 데이터에 특화된 모델
학습

statsmodels의 ZeroInflatedPoisson 또는 ZeroInflatedNegativeBinomialP 사용
두 단계로 구성: ① 0인지 아닌지 판별하는 이진 파트 + ② 0 초과 값을 예측하는 카운트 파트

평가지표

MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
실제 0 비율 vs 예측 0 비율 비교 (Zero-Inflation 재현 여부 확인)

주의점: statsmodels 기반이라 sklearn과 인터페이스가 다름, 수렴 실패 가능성 있으므로 maxiter 조정 필요

Step 3-C. 일반 Regression 모델
목적: 발생건수를 연속형 수치로 직접 예측, 베이스라인 역할
학습 후보 모델 3가지

LinearRegression (기본 베이스라인)
RandomForestRegressor (비선형 관계 포착)
GradientBoostingRegressor (불균형 데이터에 상대적으로 강건)

평가지표

MAE, RMSE, R² Score
예측값 분포 시각화 (실제 vs 예측 scatter plot)

주의점: 회귀 모델은 음수 예측값이 나올 수 있으므로 후처리로 max(0, y_pred) 클리핑 필요

Step 4. 모델 비교 및 검증
교차검증

데이터가 적으므로 StratifiedKFold (이진 분류) 또는 KFold (회귀) 5-fold 교차검증 적용
단순 train/test 분할의 분산을 줄이기 위해 필수

모델별 최종 비교표 (예시)
모델주요 지표장점단점로지스틱 이진 분류F1, AUC해석 쉬움건수 예측 불가Zero-InflatedMAE, RMSE0 과다 분포에 이론적으로 적합구현 복잡, 수렴 불안정RegressionMAE, RMSE, R²건수 직접 예측0 집중 분포에 취약

Step 5. 최종 모델 선택 기준
프로젝트 목적이 "사고 위험 스쿨존 식별" 이라면:

False Negative(사고 있는데 없다고 예측)를 최소화하는 것이 중요
따라서 Recall을 우선 지표로 삼고, 로지스틱 이진 분류가 유리할 가능성이 높음

프로젝트 목적이 "사고 건수 예측" 이라면:

Zero-Inflated 또는 Regression 모델의 MAE/RMSE 기준으로 선택