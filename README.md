# 스쿨존 안전시설물 기반 어린이 사고 위험도 예측 모델
# School Zone Safety Risk Prediction Model

> MS AI School 9기 1차 프로젝트 | 1st Project — MS AI School Cohort 9

---

## 프로젝트 개요 | Overview

전국 스쿨존(어린이 보호구역)의 안전시설물 현황과 도로 구조를 분석하여, 스쿨존별 **어린이 교통사고 위험도**를 예측하는 머신러닝 모델을 개발합니다.
성남시·광명시 두 지역을 대상으로 시설물 데이터와 로드뷰 이미지를 결합하여 위험 점수(Risk Score)를 산출합니다.

We develop a machine learning model that predicts **child traffic accident risk** in school zones across South Korea by analyzing safety facility data and road structure.
Using data from Seongnam and Gwangmyung cities, we combine facility counts with road-view imagery to produce a risk score per school zone.

- **분석 대상** : 성남시·광명시 스쿨존 142개소 + 외부 검증 51개소
- **데이터** : 시설 현황 10종 CSV + 로드뷰 이미지 약 4,000장 (2018~2023)
- **타겟** : 사고 발생 여부 (0: 무사고 / 1: 사고 1건 이상)

---

## 팀원 | Team

| 이름   | 메인 역할              | 서브 역할 (협업 참여)                     | 참여 파트                         |
|--------|------------------------|-------------------------------------------|----------------------------------|
| 이지원 | PM / CV / 통합         | 로드뷰 처리, 라벨링, 모델 → 대시보드 연결 | PM / CV / ML / Integration       |
| 조윤지 | 기획 / 데이터          | 발표 자료, 데이터 수집 관리               | Planning / Data / Presentation   |
| 전광민 | ML 모델링              | 모델 검증, 데이터 협업                    | ML / Validation / Data           |
| 김시연 | 데이터 전처리          | EDA, 모델 입력 데이터 최적화              | Data / EDA / ML                  |
| 이승아 | 대시보드 개발          | 시각화 설계, 결과 해석                    | Dashboard / Front / EDA          |
| 지경민 | 데이터 처리 / ML 연결  | 공간결합, 안전점수, 모델 검증             | Data / ML / Integration          |

---

## 프로젝트 흐름 | Pipeline

```
[01_data_ingestion]       원시 데이터 수집 및 저장
                         - 스쿨존 목록, 사고 기록, 안전시설물 (CSV)
                         - 로드뷰 이미지 수집 (KakaoMap)
                         - Azure Blob Storage 저장
     ↓
[02_preprocessing]        데이터 정제 및 통합 (성남시, 광명시)
                         - 결측치 처리 / 스케일링
                         - 공간 결합 (시설물 + 사고 + 위치)
     ↓
[03_computer_vision]      도로 구조 분석 (1단계 모델)
                         - Azure Custom Vision
                         - Object Detection / Classification
                         → Structure Risk (구조 위험도) 생성
     ↓
[04_feature_integration]  데이터 통합
                         - 정형 데이터 + Structure Risk 결합
                         → 모델 입력 Feature 구성
     ↓
[05_modeling]             위험도 예측 모델 (2단계 모델)
     ‖                   ├── prototype: 회귀/분류 실험, 모델 비교
     ‖                   └── final: Logistic Regression (L1/Lasso)
     ‖                       → 위험 확률 및 안전 점수 산출
     ↓
[06_model_interpretation] 모델 해석
                         - SHAP 기반 변수 중요도 분석
     ↓
[07_service]              서비스 및 시각화
                         - Streamlit Dashboard
                         - 지도 시각화 (Folium)
                         - 정책 시뮬레이터 (시설 추가 효과 분석)
```

---

## 디렉토리 구조 | Repository Structure

```
.
├── 01_data/
│   ├── raw/                          # 원시 데이터
│   │   ├── 00_school_zone_list/
│   │   ├── 01_accident_data/
│   │   ├── 02_safety_facilities/
│   │   ├── 03_road_characteristics/
│   │   └── 04_external_risk_factors/
│   └── processed/                    # 전처리 완료 데이터
│       ├── seongnam_scaled.csv
│       ├── gwangmyung_scaled.csv
│       └── facility_feature_summary_*.csv
│
├── 02_preprocessing/
│   ├── seongnam/
│   ├── gwangmyung/
│   └── *.ipynb
│
├── 03_modeling/
│   ├── prototype/                    # 실험·탐색 단계
│   └── final/                        # 최종 모델
│       ├── 0_ModelingCode_20260304.ipynb
│       ├── structure_risk_model.pkl
│       ├── 2nd_model_pipeline.pkl
│       └── 2nd_model_calibrated.pkl
│
├── 04_computer_vision/
│   ├── notebooks/
│   ├── data/
│   └── roadview_images/              # 로드뷰 이미지 (~4,000장)
│
├── docs/
│   └── 시스템_아키텍처.jpg
├── requirements.txt
└── setup.sh
```

---

## 주요 결과 | Key Results

### 1단계 — Computer Vision (구조 위험도)

- Azure Custom Vision으로 로드뷰 이미지 **약 4,000장** 학습
- **7개 클래스** 분류 (중앙선 유무, 방호울타리, 도로폭 등)
- AUC **0.71** (전국), Precision@20% **0.75** (위험 상위 20% 중 75%가 실제 사고 구역)
- 추출한 CV 특징값 5개를 ML 모델 피처로 활용

### 2단계 — ML 모델 (최종 위험도 예측)

| 모델 | 설명 | AUC |
|---|---|---|
| V6 룰 기반 점수 | 시설 지표 가중합 공식 | — |
| **LR (L1 + SMOTE + OOF)** | **최종 채택 모델** | **0.807** |
| GridSearchCV 최적화 | 하이퍼파라미터 탐색 최고치 | 0.865 |

- Recall **0.750** — 사고 구역 미탐지 최소화
- L1 정규화로 17개 변수 → **5개 핵심 변수** 선택
- OOF 5-Fold 교차검증으로 과적합 방지

**핵심 위험 요인 (LR 계수 기준)**

| 방향 | 변수 | 계수 |
|---|---|---|
| ↑ 위험 증가 | 어린이 인구 (log 변환) | +0.36 |
| ↑ 위험 증가 | CCTV × 어린이 인구 교호항 | +0.33 |
| ↑ 위험 증가 | 사고 이력 (log 변환) | +0.21 |
| ↓ 위험 감소 | 과속방지턱 수 | −0.048 |
| ↓ 위험 감소 | 횡단보도 수 | −0.044 |

### 주요 인사이트

- **D등급** 평균 사고 12.2건 vs **A등급** 1.6건 — **7.6배 차이**
- CCTV 1대 추가 시 위험 확률 **−2.3%p** (LR 시뮬레이션)
- 광명시 51개 스쿨존 외부 검증 완료

**안전 등급 분포 (142개 스쿨존)**

| 등급 | 구역 수 | 평균 사고건수 |
|---|---|---|
| A (안전) | 36개 (25.4%) | 1.6건 |
| B | 35개 (24.6%) | — |
| C | 35개 (24.6%) | — |
| D (위험) | 36개 (25.4%) | 12.2건 |

### 대시보드

Streamlit 기반 인터랙티브 대시보드 배포

👉 **[라이브 데모](https://schoolzone-dashboard-ybojphvjsgnxxx6cfj6uix.streamlit.app)**

- Folium 지도: 스쿨존 위치 + 안전 등급 시각화
- D등급 구역 시설 개선 시뮬레이션
- CSV 다운로드 기능

---

## 사용법 | Getting Started

```bash
pip install -r requirements.txt
# 또는
bash setup.sh
```

권장 실행 순서:

1. `02_preprocessing/facility_data_preprocessing_SN.ipynb` — 성남시 전처리
2. `02_preprocessing/facility_data_preprocessing_GM.ipynb` — 광명시 전처리
3. `04_computer_vision/notebooks/street_view_preprocessing.ipynb` — 로드뷰 전처리
4. `03_modeling/prototype/structure_risk_logistic.ipynb` — Structure Risk 학습
5. `03_modeling/final/0_ModelingCode_20260304.ipynb` — 최종 위험도 스코어링

---

## 기술 스택 | Tech Stack

| 역할 | 기술 |
|---|---|
| ML 모델 | Logistic Regression (L1, SMOTE, OOF), Random Forest, XGBoost |
| Computer Vision | Azure Custom Vision (Object Detection + Classification) |
| 대시보드 | Streamlit, Folium |
| 데이터 처리 | Python, pandas, scikit-learn, geopandas |

---

## 프로젝트 일정 | Schedule

| 기간 | 단계 |
|------|------|
| 2/23–2/24 | 주제 설정 및 역할 분담 |
| 2/25–2/26 | 데이터 수집 및 검증 |
| 2/27–3/3  | 데이터 전처리 |
| 3/4–3/5   | 모델링 및 평가 |
| 3/6       | 상품화 (앱/웹 구현) |
| 3/9       | 발표자료 작성 |
| 3/10      | 발표 및 평가 |
