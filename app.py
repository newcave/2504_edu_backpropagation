import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# 데이터 파일 경로
file_paths = {
    "Johnstone River - Coquette Point": "./Johnstone_river_coquette_point_joined.csv",
    "Johnstone River - Innisfail": "./Johnstone_river_innisfail_joined.csv",
    "Mulgrave River - Deeral": "./Mulgrave_river_deeral_joined.csv",
    "Pioneer - Dumbleton": "./Pioneer_Dumbleton_joined.csv",
    "Plane Creek - Sucrogen": "./Plane_ck_sucrogen_joined.csv",
    "Proserpine River - Glen Isla": "./Proserpine_river_glen_isla_joined.csv",
    "Russell River - East Russell": "./russell_river_east_russell_joined.csv",
    "Sandy Creek - Homebush": "./sandy_ck_homebush_joined.csv",
    "Sandy Creek - Sorbellos Road": "./sandy_ck_sorbellos_road_joined.csv",
    "Tully River - Euramo": "./Tully_river_euramo_joined.csv"
}

st.set_page_config(page_title="Water Quality Dashboard", layout="wide")
st.title("🌊 Water Quality Dashboard")

# 지점 선택
dataset_name = st.selectbox("Select a monitoring site:", list(file_paths.keys()))

# 데이터 불러오기
df = pd.read_csv(file_paths[dataset_name])

# 날짜 컬럼이 있는 경우 datetime 형식으로 변환
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# 수질 변수 선택 (숫자형만 필터링)
numeric_cols = df.select_dtypes(include='number').columns.tolist()
selected_columns = st.multiselect("Select water quality variables to visualize:", numeric_cols, default=numeric_cols[:2])

# 데이터 테이블 표시
st.subheader(f"📋 Data Preview for {dataset_name}")
st.dataframe(df.head(10))

# 시계열 시각화
if selected_columns and 'Date' in df.columns:
    st.subheader("📈 Time Series of Selected Variables")
    for col in selected_columns:
        fig, ax = plt.subplots()
        ax.plot(df['Date'], df[col], label=col)
        ax.set_xlabel("Date")
        ax.set_ylabel(col)
        ax.set_title(f"{col} over Time")
        ax.legend()
        st.pyplot(fig)
else:
    st.info("Please ensure 'Date' column exists and at least one variable is selected.")
