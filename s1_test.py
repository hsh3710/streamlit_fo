import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.write("""
          ```python 
            st.title("st.write() 활용 예시")
         ``` 
         """)

st.title("st.write() 활용 예시")

# 1. 텍스트 출력 (Markdown 지원)
st.write("## 마크다운 제목")
st.write("*강조된 텍스트*")
st.write("***")
st.write("- 다음")
st.write("          - 다음")
st.write("""
        - `st.write()`에서 연결해서 쓰기
            - 전 후로 "를 3개 씩 붙여준뒤 사용
                - 그러면 이렇게 연결된 하위 표시 만들 수 있음.
                - **마크다운**을 지원하는 곳에서만 사용 가능하다.
         """)

# 2. 데이터프레임 출력
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
st.write("### 데이터프레임")
st.write(df)

# 3. 그래프 출력 (Matplotlib)
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])  # 좌표 (1,1) (2,4) (3,2) 이 3개 좌표가 연결된다.
st.write("### Matplotlib 그래프")
st.pyplot(fig)

# 4. 이미지 출력
st.write("### 이미지")
st.image("https://cdn.pixabay.com/photo/2024/02/26/19/39/monochrome-image-8598798_1280.jpg", width=300)

st.title("레이아웃 예시")

# 5. 컬럼 나누기
col1, col2, col3 = st.columns(3)

with col1:
    st.header("첫 번째 컬럼")
    st.write("컬럼 1 내용")

with col2:
    st.header("두 번째 컬럼")
    st.image("https://cdn.pixabay.com/photo/2024/02/26/19/39/monochrome-image-8598798_1280.jpg", width=150)

with col3:
    st.header("세 번째 컬럼")
    st.image("https://cdn.pixabay.com/photo/2024/02/26/19/39/monochrome-image-8598798_1280.jpg", width=150)


# 6. 사이드바
with st.sidebar:
    st.header("사이드바 메뉴")
    menu = st.radio("메뉴 선택", ["메뉴 1", "메뉴 2", "메뉴 3"])
    st.write("선택된 메뉴:", menu)

st.write("메인 화면 내용")