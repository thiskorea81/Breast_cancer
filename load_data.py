"""
유방암 진단 데이터셋 불러오기 예제
(Breast Cancer Wisconsin Diagnostic, UCI)
"""
import pandas as pd

# 1. 데이터 불러오기 -------------------------------------------------
# GitHub에서 바로 불러오려면 아래 두 줄의 주석(#)을 지우고 사용하세요.
# url = "https://raw.githubusercontent.com/thiskorea81/Breast_cancer/main/breast_cancer_wisconsin.csv"
# df = pd.read_csv(url)

df = pd.read_csv("breast_cancer_wisconsin.csv")

# 2. 데이터 기본 정보 확인 -------------------------------------------
print("데이터 크기 (행, 열):", df.shape)
print()
print(df.head())
print()
df.info()
print()

# 3. 결측치 확인 ------------------------------------------------------
print("결측치 개수 합계:", df.isnull().sum().sum())
print()

# 4. 정답(diagnosis) 분포 확인 ----------------------------------------
print("진단 결과 분포 (M=악성, B=양성)")
print(df["diagnosis"].value_counts())
print()

# 5. 기술 통계 --------------------------------------------------------
print(df.describe().T)
print()

# 6. 특성(X)과 정답(y) 나누기 -----------------------------------------
X = df.drop(columns=["id", "diagnosis"])        # 30개 특성
y = df["diagnosis"].map({"M": 1, "B": 0})       # 악성=1, 양성=0

print("X 크기:", X.shape)
print("y 크기:", y.shape)

# 7. 평균(_mean) 특성만 골라보기 --------------------------------------
mean_cols = [c for c in df.columns if c.endswith("_mean")]
print()
print("평균 특성 10개:", mean_cols)
print(df.groupby("diagnosis")[mean_cols].mean().T)
