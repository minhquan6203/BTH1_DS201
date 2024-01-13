import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from io import BytesIO

st.balloons()
st.title("Bai 10 xy dựng ứng dụng nhận dạng cảm xúc từ câu đánh giá môn học")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Simple Naive Bayes classifier
    content = uploaded_file.getvalue()
    df = pd.read_csv(BytesIO(content))
    X=df['sentence']
    y=df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    st.write(f"Đã train xong, giờ thử nhé!!!")
    st.title("Ứng Dụng Nhận Dạng Cảm Xúc Từ Câu Đánh Giá Môn Học")

    review = st.text_area("Nhập câu đánh giá môn học:")

    if st.button("Dự đoán cảm xúc"):
        if review:
            sentiment = model.predict([review])[0]
            st.subheader("Kết quả dự đoán:")
            st.write(f"Cảm xúc từ câu đánh giá: {sentiment}")
        else:
            st.warning("Vui lòng nhập câu đánh giá để dự đoán.")
