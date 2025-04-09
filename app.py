import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

# Hide sidebar and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-18e3th9 {padding-top: 1rem; padding-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🔁 역전파 알고리즘 수동 계산 학습 도구")
st.markdown("""
이 도구는 간단한 인공신경망 구조에서 순전파, 오차 계산, 역전파, 가중치 업데이트까지의 과정을 **직접 수동 입력**하거나 반복 학습을 실행하며 학습할 수 있도록 구성되어 있습니다.
""")

# --- 입력값 및 초기 가중치 수동 입력 ---
st.header("1단계: 입력값과 초기 가중치 입력")
col1, col2, col3 = st.columns(3)
with col1:
    x1 = st.number_input("x1", value=0.1)
    x2 = st.number_input("x2", value=0.2)
with col2:
    target_o1 = st.number_input("실제값 y1 (target_o1)", value=0.4)
    target_o2 = st.number_input("실제값 y2 (target_o2)", value=0.6)
with col3:
    lr = st.number_input("학습률 (learning rate)", value=0.5)
    epochs = st.slider("학습 반복 횟수 (2 ~ 1000)", min_value=2, max_value=1000, value=100, step=1)

st.subheader("초기 가중치 입력")
col1, col2, col3, col4 = st.columns(4)
with col1:
    w1 = st.number_input("w1", value=0.3)
    w2 = st.number_input("w2", value=0.25)
with col2:
    w3 = st.number_input("w3", value=0.4)
    w4 = st.number_input("w4", value=0.35)
with col3:
    w5 = st.number_input("w5", value=0.45)
    w6 = st.number_input("w6", value=0.4)
with col4:
    w7 = st.number_input("w7", value=0.7)
    w8 = st.number_input("w8", value=0.6)

# --- 학습 반복 ---
st.header("2단계: 반복 학습 시뮬레이션")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

error_list = []
o1_list = []
o2_list = []
report_data = []

for epoch in range(1, epochs + 1):
    # 순전파
    z1 = x1 * w1 + x2 * w3
    h1 = sigmoid(z1)
    z2 = x1 * w2 + x2 * w4
    h2 = sigmoid(z2)

    z3 = h1 * w5 + h2 * w6
    o1 = sigmoid(z3)
    z4 = h1 * w7 + h2 * w8
    o2 = sigmoid(z4)

    # 오차
    E1 = 0.5 * (target_o1 - o1)**2
    E2 = 0.5 * (target_o2 - o2)**2
    E_total = E1 + E2

    # 역전파 (출력층만)
    d_o1 = -(target_o1 - o1) * sigmoid_deriv(o1)
    d_o2 = -(target_o2 - o2) * sigmoid_deriv(o2)

    w5 -= lr * d_o1 * h1
    w6 -= lr * d_o1 * h2
    w7 -= lr * d_o2 * h1
    w8 -= lr * d_o2 * h2

    error_list.append(E_total)
    o1_list.append(o1)
    o2_list.append(o2)

    # 50회 단위로 결과 저장
    if epoch % 50 == 0 or epoch == epochs:
        report_data.append({
            'Epoch': epoch,
            '출력값 o1': round(o1, 4),
            '출력값 o2': round(o2, 4),
            'y1 오차율(%)': round(abs((target_o1 - o1) / target_o1) * 100, 2),
            'y2 오차율(%)': round(abs((target_o2 - o2) / target_o2) * 100, 2),
            '총 오차': round(E_total, 6)
        })

# --- 출력 요약 ---
st.success(f"🎯 최종 출력: o1 = {round(o1_list[-1], 4)}, o2 = {round(o2_list[-1], 4)}")
st.info(f"총 오차: {round(error_list[-1], 6)} (감소율: {round((error_list[0] - error_list[-1]) / error_list[0] * 100, 2)}%)")

# --- 요약 테이블 출력 ---
st.header("3단계: 50회 단위 학습 요약 테이블")
report_df = pd.DataFrame(report_data)
st.dataframe(report_df, use_container_width=True)

# --- 시각자료 첨부 위치 ---
st.header("4단계: 관련 시각자료 보기")

image1 = st.file_uploader("📌 최초 신경망 구조 그림 업로드", type=["png", "jpg"])
if image1:
    st.image(image1, caption="초기 구조도", use_column_width=True)

image2 = st.file_uploader("📌 1회 순전파 결과 그림 업로드", type=["png", "jpg"])
if image2:
    st.image(image2, caption="1회 순전파 결과", use_column_width=True)

image3 = st.file_uploader("📌 2회 역전파 결과 그림 업로드", type=["png", "jpg"])
if image3:
    st.image(image3, caption="2회 역전파 결과", use_column_width=True)

image4 = st.file_uploader("📌 엑셀 기반 반복 학습 시각화", type=["png", "jpg"])
if image4:
    st.image(image4, caption="엑셀 반복학습 구조", use_column_width=True)

st.markdown("---")
st.success("✅ 모든 단계를 수동 또는 반복 학습으로 실습할 수 있습니다. 목표 출력에 가까워지는 과정을 직접 확인해보세요!")
