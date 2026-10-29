# 유방암 진단 데이터셋 (Breast Cancer Wisconsin Diagnostic)

수행평가용 데이터입니다. 유방 종양 세포 이미지에서 측정한 수치로 **악성(M)** 인지 **양성(B)** 인지를 분류하는 데이터셋입니다.

- 출처: UCI Machine Learning Repository — [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (CC BY 4.0)
- 원작자: Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (University of Wisconsin, 1995)

## 파일 구성

| 파일 | 설명 |
|---|---|
| `breast_cancer_wisconsin.csv` | 수행평가용 데이터 (CSV, 헤더 포함) |
| `README.md` | 데이터 및 속성 안내 (이 문서) |

> 수행평가는 **Google Colab**에서 진행합니다. 아래 **데이터 불러오기** 코드로 데이터를 불러온 뒤, 이후 과정은 스스로 해결해 보세요.

## 데이터 개요

| 항목 | 내용 |
|---|---|
| 샘플(행) 수 | 569개 |
| 열 수 | 32개 (ID 1 + 진단 결과 1 + 특성 30) |
| 결측치 | 없음 |
| 클래스 분포 | 양성(B) 357개 (62.7%), 악성(M) 212개 (37.3%) |

## 데이터는 어떻게 만들어졌나?

유방에 생긴 혹(종양)에 가는 바늘을 찔러 세포를 채취(**세침흡인검사, FNA**)한 뒤, 그 세포를 현미경으로 촬영한 디지털 이미지에서 **세포핵의 모양**을 측정했습니다.
한 이미지 안에는 세포핵이 여러 개 있으므로, 측정한 10가지 특징마다 아래 3가지 통계값을 계산해 총 **10 × 3 = 30개 특성**을 만들었습니다.

| 접미사 | 의미 |
|---|---|
| `_mean` | 이미지 속 세포핵들의 **평균값** |
| `_se` | **표준오차** (Standard Error) — 값이 얼마나 흩어져 있는지 |
| `_worst` | **가장 나쁜(큰) 값** — 값이 가장 큰 3개 세포핵의 평균 |

## 속성(열) 설명

### 식별 및 정답(타깃)

| 열 이름 | 설명 | 값 |
|---|---|---|
| `id` | 환자 식별 번호 | 정수 (분석에는 사용하지 않음) |
| `diagnosis` | **진단 결과 (예측해야 할 정답)** | `M` = 악성(Malignant), `B` = 양성(Benign) |

### 세포핵의 10가지 특징

| 특징 | 한글 이름 | 설명 |
|---|---|---|
| `radius` | 반지름 | 세포핵 중심에서 둘레 위 점들까지 거리의 평균 |
| `texture` | 질감 | 회색조(gray-scale) 밝기 값의 표준편차 — 클수록 얼룩덜룩함 |
| `perimeter` | 둘레 | 세포핵 테두리의 길이 |
| `area` | 넓이 | 세포핵의 면적 |
| `smoothness` | 매끄러움 | 반지름 길이의 국소적 변화량 — 테두리가 얼마나 울퉁불퉁한지 |
| `compactness` | 조밀도 | 둘레² / 넓이 − 1.0 — 원에 가까울수록 작음 |
| `concavity` | 오목함 정도 | 테두리에서 안쪽으로 들어간(오목한) 부분이 얼마나 심한지 |
| `concave_points` | 오목점 개수 | 테두리에서 오목한 부분의 개수 |
| `symmetry` | 대칭성 | 세포핵 모양의 대칭 정도 |
| `fractal_dimension` | 프랙탈 차원 | 테두리의 복잡도 ("해안선 근사" − 1) |

### 전체 30개 특성 열 목록

| 번호 | 평균 (`_mean`) | 표준오차 (`_se`) | 최악값 (`_worst`) |
|---|---|---|---|
| 1 | `radius_mean` | `radius_se` | `radius_worst` |
| 2 | `texture_mean` | `texture_se` | `texture_worst` |
| 3 | `perimeter_mean` | `perimeter_se` | `perimeter_worst` |
| 4 | `area_mean` | `area_se` | `area_worst` |
| 5 | `smoothness_mean` | `smoothness_se` | `smoothness_worst` |
| 6 | `compactness_mean` | `compactness_se` | `compactness_worst` |
| 7 | `concavity_mean` | `concavity_se` | `concavity_worst` |
| 8 | `concave_points_mean` | `concave_points_se` | `concave_points_worst` |
| 9 | `symmetry_mean` | `symmetry_se` | `symmetry_worst` |
| 10 | `fractal_dimension_mean` | `fractal_dimension_se` | `fractal_dimension_worst` |

모든 특성 값은 실수이며, 유효숫자 4자리로 기록되어 있습니다.

## 데이터 불러오기 (Google Colab)

Colab 새 노트북의 코드 셀에 아래 코드를 붙여넣고 실행하세요.

```python
import pandas as pd

url = "https://raw.githubusercontent.com/thiskorea81/Breast_cancer/main/breast_cancer_wisconsin.csv"
df = pd.read_csv(url)

print(df.shape)       # (569, 32)
df.head()
```

이후 과정은 스스로 해결해 보세요.
