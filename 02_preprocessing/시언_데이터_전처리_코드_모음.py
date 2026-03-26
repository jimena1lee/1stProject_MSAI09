데이터 전처리 코드 모음.
1. TOP 10 학교별 세부 위험 점수 분석 코드

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df_acc_raw = pd.read_csv("성남시전체_스쿨존_사고_전처리 - 성남시_스쿨존_반경300m_최종_전처리 (1).csv")
df_pop_raw = pd.read_csv("어린이비율_행정동별.csv")

acc_summary = df_acc_raw.groupby('대상시설명').agg({
    '발생건수': 'sum',
    '사망자수': 'sum',
    '중상자수': 'sum',
    '경상자수': 'sum',
    '부상자수': 'sum',
    '소재지지번주소': 'first'
}).reset_index()

# 사고 심각도 계산 (가중치: 사망 10, 중상 5, 경상 2, 부상 1)
acc_summary['사고심각도'] = (acc_summary['사망자수']*10 + acc_summary['중상자수']*5 + 
                          acc_summary['경상자수']*2 + acc_summary['부상자수']*1)

top10_schools = acc_summary.sort_values(by='발생건수', ascending=False).head(10).copy()

def extract_dong(addr):
    if pd.isna(addr): return "알수없음"
    parts = addr.split()
    for p in parts:
        if p.endswith('동'): return re.sub(r'\d+', '', p) # 숫자 제거 (수진1동 -> 수진동)
    return "알수없음"

top10_schools['법정동'] = top10_schools['소재지지번주소'].apply(extract_dong)

df_pop_raw['법정동'] = df_pop_raw['동명'].apply(lambda x: re.sub(r'\d+', '', x))
legal_pop = df_pop_raw.groupby('법정동').agg({
    '총인구수': 'sum',
    '어린이인구_0_14': 'sum'
}).reset_index()

legal_pop['어린이비율_최신'] = (legal_pop['어린이인구_0_14'] / legal_pop['총인구수']) * 100


df_final = top10_schools.merge(legal_pop, on='법정동', how='left')

# 가중치 점수 산출 (기초표 기준 반영: 30 / 13 / 7)
max_acc_risk = (df_final['발생건수'] + df_final['사고심각도']).max()
df_final['사고감점'] = ((df_final['발생건수'] + df_final['사고심각도']) / max_acc_risk) * 30

max_child_pop = legal_pop['어린이인구_0_14'].max()
df_final['인구감점'] = (df_final['어린이인구_0_14'] / max_child_pop) * 13

max_child_ratio = legal_pop['어린이비율_최신'].max()
df_final['비율감점'] = (df_final['어린이비율_최신'] / max_child_ratio) * 7

df_final['위험합계'] = df_final['사고감점'] + df_final['인구감점'] + df_final['비율감점']

fig, ax = plt.subplots(figsize=(14, 9))
y_pos = np.arange(len(df_final))

# 쌓인 바(Stacked Bar) 그리기
ax.barh(y_pos, df_final['사고감점'], label='사고위험 감점 (30%)', color='#FF6B6B', alpha=0.9)
ax.barh(y_pos, df_final['인구감점'], left=df_final['사고감점'], label='어린이인구 감점 (13%)', color='#FFD93D', alpha=0.9)
ax.barh(y_pos, df_final['비율감점'], left=df_final['사고감점']+df_final['인구감점'], label='어린이비율 감점 (7%)', color='#6BCB77', alpha=0.9)

# --- 각 영역별 점수 텍스트 추가 ---
for i in range(len(df_final)):
    # 1. 사고감점 (빨간색 영역 중앙)
    v1 = df_final['사고감점'].iloc[i]
    ax.text(v1/2, i, f'{v1:.1f}', va='center', ha='center', color='white', fontweight='bold')
    
    # 2. 인구감점 (노란색 영역 중앙)
    v2 = df_final['인구감점'].iloc[i]
    ax.text(df_final['사고감점'].iloc[i] + v2/2, i, f'{v2:.1f}', va='center', ha='center', color='black', fontweight='bold')
    
    # 3. 비율감점 (초록색 영역 중앙)
    v3 = df_final['비율감점'].iloc[i]
    ax.text(df_final['사고감점'].iloc[i] + v2 + v3/2, i, f'{v3:.1f}', va='center', ha='center', color='black', fontweight='bold')
    
    # 4. 전체 합계 (바 끝부분)
    total = df_final['위험합계'].iloc[i]
    ax.text(total + 0.5, i, f'총 {total:.1f}점', va='center', fontweight='black', color='#333333', fontsize=11)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_final['대상시설명'], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('위험 감점 합계 (총 50점 만점)', fontsize=12)
ax.set_title('사고 TOP 10 학교별 세부 위험 점수 분석 (최신 데이터 반영)', fontsize=16, pad=25)
ax.legend(title='감점 항목 (가중치)', loc='lower right', fontsize=10)

plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()



2. 성남시 스쿨존 사고 발생 TOP 10 코드
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_acc = pd.read_csv("성남시전체_스쿨존_사고_전처리 - 성남시_스쿨존_반경300m_최종_전처리 (1).csv")

def extract_dong(address):
    try:
        parts = address.split()
        return f"{parts[2]} {parts[3]}"
    except:
        return "정보없음"

df_acc['행정동'] = df_acc['소재지지번주소'].apply(extract_dong)

top10_names = df_acc.groupby('대상시설명')['발생건수'].sum().sort_values(ascending=False).head(10).index
df_top10 = df_acc[df_acc['대상시설명'].isin(top10_names)].copy()

school_dong_map = df_top10[['대상시설명', '행정동']].drop_duplicates().set_index('대상시설명')['행정동'].to_dict()
df_top10['라벨'] = df_top10['대상시설명'].apply(lambda x: f"{x} ({school_dong_map.get(x, '')})")

yearly_acc = df_top10.groupby(['라벨', '사고년도'])['발생건수'].sum().reset_index()
pivot_df = yearly_acc.pivot(index='라벨', columns='사고년도', values='발생건수').fillna(0)
pivot_df['Total'] = pivot_df.sum(axis=1)
pivot_df = pivot_df.sort_values(by='Total', ascending=True) # 가로 막대 그래프는 아래서 위로 정렬되므로 ascending=True
total_scores = pivot_df['Total']
pivot_df = pivot_df.drop(columns='Total')

plt.figure(figsize=(12, 10))
pivot_df.plot(kind='barh', stacked=True, ax=plt.gca(), colormap='viridis', alpha=0.9)

plt.title('성남시 스쿨존 사고 발생 TOP 10 (학교명 및 행정동)', fontsize=16, pad=20)
plt.xlabel('누적 사고 발생 건수', fontsize=12)
plt.ylabel('학교명 (관할 행정동)', fontsize=12)
plt.legend(title='사고년도', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.7)

for i, (label, total) in enumerate(total_scores.items()):
    plt.text(total + 1, i, f'{int(total)}건', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('top10_accidents_horizontal.png')

3. 성남시 시설물 평균 설치량 분포도 (클러스터) 코드

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371000 # 지구 반지름 (m)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df_schools = pd.read_csv("어린이보호구역_위치정보.csv")[['대상시설명', '위도', '경도']]

infra_files = {
    'CCTV': '생활안전CCTV_전처리1.csv',
    '무인카메라': '무인교통단속카메라_전처리1.csv',
    '신호등': '신호등_전처리1.csv',
    '횡단보도': '횡단보도_전처리1.csv',
    '도로안전표지': '도로안전표지_전처리1.csv',
    '표지판': '보호구역표지판_전처리1.csv',
    '옐로카펫': '옐로카펫_전처리1.csv',
    '울타리_펜스': '무단횡단방지펜스_전처리1.csv',
    '도로적색표면': '도로적색표면_전처리1.csv',
    '지킴이집': '성남시_아동안전지킴이집현황_전처리.csv'
}

print("시설물 위치를 계산하고 있습니다. 잠시만 기다려주세요...")
for label, file in infra_files.items():
    df_fac = pd.read_csv(file)
    counts = []
    for _, school in df_schools.iterrows():
        dist = haversine_dist(school['위도'], school['경도'], df_fac['위도'], df_fac['경도'])
        counts.append(np.sum(dist <= 300))
    df_schools[label] = counts

infra_features = list(infra_files.keys())
df_schools['시설_다양성'] = (df_schools[infra_features] > 0).sum(axis=1)

infra_weights = {
    'CCTV': 0.08, '무인카메라': 0.08, '신호등': 0.06, '횡단보도': 0.06,
    '도로안전표지': 0.04, '표지판': 0.04, '옐로카펫': 0.08, '울타리_펜스': 0.06,
    '도로적색표면': 0.06, '지킴이집': 0.06, '시설_다양성': 0.06
}

scaler = MinMaxScaler()
cols_to_scale = list(infra_weights.keys())
df_scaled = pd.DataFrame(scaler.fit_transform(df_schools[cols_to_scale]), columns=cols_to_scale)

df_schools['시설_점수'] = sum(df_scaled[f] * infra_weights[f] for f in infra_weights.keys())
df_schools['시설_점수_100'] = (df_schools['시설_점수'] / 0.68) * 100

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_schools['시설_클러스터'] = kmeans.fit_predict(df_scaled)
color_map = {0: '#FF4B4B', 1: '#FFA500', 2: '#2E8B57'} # 0:Red, 1:Orange, 2:Green
label_names = {0: '0단계(취약)', 1: '1단계(보통)', 2: '2단계(우수)'}

plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
sns.scatterplot(x='pca_x', y='pca_y', hue='시설_클러스터', 
                palette=color_map, 
                data=df_schools, s=120, alpha=0.8, edgecolor='w')

plt.title('성남시 시설물 클러스터 분포 (빨강:취약, 주황:보통, 초록:우수)', fontsize=14)
plt.xlabel('인프라 풍부도/다양성')
plt.ylabel('인프라 구성 특성')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
profile = df_schools.groupby('시설_클러스터')[cols_to_scale].mean().T

profile.plot(kind='bar', ax=plt.gca(), width=0.8, color=[color_map[0], color_map[1], color_map[2]])

plt.title('클러스터별 시설물 평균 설치량 비교', fontsize=14)
plt.ylabel('평균 설치 개수')
plt.xticks(rotation=45)
plt.legend(labels=['0단계(취약)', '1단계(보통)', '2단계(우수)'], title='시설 등급')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

print("=== [시설물 등급별 기획 가이드] ===")
for i in range(3):
    count = len(df_schools[df_schools['시설_클러스터'] == i])
    score = df_schools[df_schools['시설_클러스터'] == i]['시설_점수_100'].mean()
    print(f"📍 {label_names[i]} (색상: {color_map[i]})")
    print(f"   - 학교 수: {count}개 / 평균 점수: {score:.2f}점")
    print(f"   - 전략: {'인프라 신설 필요' if i==0 else '인프라 밀도 보완' if i==1 else '표준 모델 유지'}\n")

4. 성남시 스쿨존 인프라 등급 분포도 코드

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False

# --- [설정] K값 입력 ---
K_VALUE = 3 

df = pd.read_csv("성남시_스쿨존_통합_데이터.csv")
features = [
    '개수_횡단보도', '개수_옐로카펫', '개수_신호등', '개수_CCTV', '개수_표지판', 
    '개수_단속카메라', '개수_펜스', '개수_적색표면', '개수_안전표지', '개수_지킴이집', '조명개수'
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

model = KMeans(n_clusters=K_VALUE, random_state=42, n_init=10)
df['등급'] = model.fit_predict(X_scaled)

cluster_summary = df.groupby('등급')[features].mean().sum(axis=1).sort_values(ascending=False)
rank_map = {cluster_id: f"{i+1}등급 (상위 {int((i+1)/K_VALUE*100)}%)" for i, cluster_id in enumerate(cluster_summary.index)}
df['등급_명칭'] = df['등급'].map(rank_map)

print(f"\n📊 [성남시 스쿨존 안전 인프라 등급 리포트 (K={K_VALUE})]")
print("=" * 70)
for cluster_id in cluster_summary.index:
    print(f"\n📍 {rank_map[cluster_id]}:")
    avg_vals = df[df['등급'] == cluster_id][features].mean().round(2)
    top_feature = avg_vals.idxmax()
    print(f"  - 주요 특징: {top_feature} 등이 상대적으로 많이 설치됨")
    print(f"  - 평균 시설물 보유량:\n{avg_vals.to_dict()}")
print("=" * 70)

pca = PCA(n_components=2)
pca_res = pca.fit_transform(X_scaled)
df['x'], df['y'] = pca_res[:, 0], pca_res[:, 1]

plt.figure(figsize=(12, 7))
scatter = sns.scatterplot(x='x', y='y', hue='등급_명칭', data=df, palette='coolwarm', s=100)
plt.title(f'성남시 스쿨존 인프라 등급 분포도 (K={K_VALUE})', fontsize=15)
plt.xlabel('인프라 풍부도 (오른쪽일수록 시설 많음)', fontsize=12)
plt.ylabel('인프라 구성 특성', fontsize=12)
plt.legend(title='안전 등급', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
