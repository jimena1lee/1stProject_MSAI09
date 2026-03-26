# 스쿨존 안전시설물 기반 어린이 사고 위험도 예측 모델
# School Zone Safety Risk Prediction Model

> MS AI School 9기 1차 프로젝트 | 1st Project — MS AI School Cohort 9

---

## 프로젝트 개요 | Overview

전국 스쿨존(어린이 보호구역)의 안전시설물 현황과 도로 구조를 분석하여, 스쿨존별 **어린이 교통사고 위험도**를 예측하는 머신러닝 모델을 개발합니다.
성남시·광명시 두 지역을 대상으로 시설물 데이터와 로드뷰 이미지를 결합하여 위험 점수(Risk Score)를 산출합니다.

We develop a machine learning model that predicts **child traffic accident risk** in school zones across South Korea by analyzing safety facility data and road structure.
Using data from Seongnam and Gwangmyung cities, we combine facility counts with road-view imagery to produce a risk score per school zone.

---

## 팀원 | Team

| 이름   | 메인 역할              | 서브 역할 (협업 참여)                    | 참여 파트                         |
|--------|------------------------|------------------------------------------|----------------------------------|
| 이지원 | PM / CV / 통합         | 로드뷰 처리, 라벨링, 모델 → 대시보드 연결 | PM / CV / ML / Integration       |
| 조윤지 | 기획 / 데이터          | 발표 자료, 데이터 수집 관리              | Planning / Data / Presentation   |
| 전광민 | ML 모델링              | 모델 검증, 데이터 협업                   | ML / Validation / Data           |
| 김시연 | 데이터 전처리          | EDA, 모델 입력 데이터 최적화             | Data / EDA / ML                  |
| 이승아 | 대시보드 개발          | 시각화 설계, 결과 해석                   | Dashboard / Front / EDA          |
| 지경민 | 데이터 처리 / ML 연결  | 공간결합, 안전점수, 모델 검증            | Data / ML / Integration          |

---

## 프로젝트 흐름 | Pipeline

```
flowchart LR

    A[01 Data Ingestion<br/>- 스쿨존, 사고, 시설물<br/>- 로드뷰 이미지 수집<br/>- Azure Blob 저장]

    B[02 Preprocessing<br/>- 지역 필터링 (성남/광명)<br/>- 결측치 처리 / 스케일링<br/>- 공간 결합]

    C[03 Computer Vision<br/>Azure Custom Vision<br/>- Object Detection<br/>- Classification<br/>→ Structure Risk 생성]

    D[04 Feature Integration<br/>- 정형 데이터 + CV 결과 결합<br/>→ 모델 입력 데이터]

    E[05 Modeling<br/>Logistic Regression (L1/Lasso)<br/>- 교차검증<br/>→ 위험 확률 / 안전 점수]

    F[06 Model Interpretation<br/>- SHAP 기반 변수 중요도]

    G[07 Service<br/>Streamlit Dashboard<br/>- Folium 지도<br/>- 정책 시뮬레이터<br/>- 의사결정 지원]

    A --> B --> C --> D --> E --> F --> G

- 본 시스템은 **2-stage pipeline 구조**로 설계됨
  1. Computer Vision을 통해 도로 구조 위험도(Structure Risk) 생성
  2. 정형 데이터와 결합하여 사고 위험 확률을 예측

- 주요 흐름:
  Data → Preprocessing → CV Feature Extraction → Feature Integration → ML Modeling → Interpretation → Dashboard

- 최종 산출물:
  - 스쿨존별 위험 확률
  - 안전 점수
  - 정책 시뮬레이션 기반 의사결정 지원
```



---

## 디렉토리 구조 | Repository Structure

```
.
├── 01_data/
│   ├── raw/                        # 원시 데이터 | Raw data
│   │   ├── 00_school_zone_list/    # 어린이 보호구역 목록
│   │   ├── 01_accident_data/       # 스쿨존 교통사고 데이터
│   │   ├── 02_safety_facilities/   # 안전시설물 (CCTV, 신호등, 횡단보도 등)
│   │   ├── 03_road_characteristics/# 도로 특성 (차로수, 버스정류장 등)
│   │   └── 04_external_risk_factors/ # 외부 위험요소 (인구, 조명, 지구대 등)
│   └── processed/                  # 전처리 완료 데이터 | Processed data
│       ├── seongnam_scaled.csv
│       ├── seongnam_scaled_with_info.csv
│       ├── gwangmyung_scaled.csv
│       ├── gwangmyung_scaled_with_info.csv
│       ├── facility_feature_summary_sn.csv
│       └── facility_feature_summary_gm.csv
│
├── 02_preprocessing/               # 전처리 코드 | Preprocessing notebooks
│   ├── gwangmyung/                 # 광명시 전처리
│   ├── seongnam/                   # 성남시 전처리
│   ├── facility_data_preprocessing_GM.ipynb
│   ├── facility_data_preprocessing_SN.ipynb
│   ├── gm_scaling_analysis_report.md
│   └── sn_scaling_analysis_report.md
│
├── 03_modeling/                    # 모델링 | Modeling
│   ├── prototype/                  # 실험·탐색 단계 | Experimental stage
│   │   ├── model_prototype.ipynb
│   │   ├── regression_integrated.ipynb
│   │   ├── structure_risk_logistic.ipynb
│   │   ├── model_comparison_classification.ipynb
│   │   └── regression_integrated_pipeline.md
│   └── final/                      # 최종 모델 | Final model
│       ├── 0_ModelingCode_20260304.ipynb
│       ├── 2nd_Model_final.ipynb
│       ├── structure_risk_model.pkl
│       ├── 2nd_model_pipeline.pkl
│       └── 2nd_model_calibrated.pkl
│
├── 04_computer_vision/             # 로드뷰 이미지 분석 | Road-view image analysis
│   ├── notebooks/                  # Azure Custom Vision 분석 코드
│   ├── data/                       # 추출된 이미지 피처 CSV
│   ├── roadview_images/            # 스쿨존 로드뷰 이미지 (~4,000장)
│   └── structure_risk_pipeline.md  # Structure Risk 계산 파이프라인 문서
│
├── docs/
│   └── 시스템_아키텍처.jpg          # 전체 시스템 아키텍처 다이어그램
│
├── requirements.txt
└── setup.sh
```

---

## 주요 결과 | Key Results

### Structure Risk (도로 구조 위험도)
- **방법**: 로드뷰 이미지 4,004장 → Azure Custom Vision Object Detection + Classification → Logistic Regression
- **AUC**: 0.71 (전국 범위), 0.63 (성남시 재검증)
- **Precision@20%**: 0.75 — 위험 상위 20% 지역 중 75%가 실제 사고 지역

### 시설물 기반 위험도 모델 (Facility Risk)
- **방법**: 안전시설물 10개 변수 + 어린이 인구 비율 → Random Forest / XGBoost
- 성남시·광명시 두 지역 교차 검증 완료
- 최종 모델: `03_modeling/final/structure_risk_model.pkl`

---

## 사용법 | Getting Started

```bash
# 환경 설치 | Install dependencies
pip install -r requirements.txt

# 또는 setup 스크립트 실행 | Or run setup script
bash setup.sh
```

주요 실행 순서 | Recommended execution order:

1. `02_preprocessing/facility_data_preprocessing_SN.ipynb` — 성남시 전처리
2. `02_preprocessing/facility_data_preprocessing_GM.ipynb` — 광명시 전처리
3. `04_computer_vision/notebooks/street_view_preprocessing.ipynb` — 로드뷰 이미지 전처리
4. `03_modeling/prototype/structure_risk_logistic.ipynb` — Structure Risk 모델 학습
5. `03_modeling/final/0_ModelingCode_20260304.ipynb` — 최종 위험도 스코어링

---

## 기술 스택 | Tech Stack

- **Data**: Python (pandas, scikit-learn, geopandas)
- **Modeling**: Logistic Regression, Random Forest, XGBoost
- **Computer Vision**: Azure Custom Vision (Object Detection + Classification)
- **Visualization**: matplotlib, seaborn

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
