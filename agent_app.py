import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
import re
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import plotly.express as px
import os

# ==========================================
# 1. KHỞI TẠO DATABASE (CHỈ CHẠY 1 LẦN)
# ==========================================
DB_PATH = os.path.abspath("ecommerce.db").replace('\\', '/')
DB_URI = f"sqlite:///{DB_PATH}"

@st.cache_resource(show_spinner="Đang nạp Data... Vui lòng đợi!")
def init_db():
    conn = sqlite3.connect("ecommerce.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='df_orders'")
    if cursor.fetchone()[0] == 0:
        opts = {'sep': None, 'engine': 'python', 'on_bad_lines': 'skip'}
        # Tự động nạp 5 file CSV
        for table in ["df_Customers", "df_Orders", "df_Payments", "df_Products", "df_OrderItems"]:
            try: pd.read_csv(f"{table}.csv", **opts).to_sql(table.lower(), conn, index=False)
            except: pass
    conn.close()

init_db()

# ==========================================
# 2. GIAO DIỆN TỐI GIẢN
# ==========================================
st.set_page_config(page_title="AI Analyst Lite", page_icon="⚡", layout="wide")
st.title("⚡ AI Analyst (Chế độ Tiết kiệm API)")

with st.sidebar:
    st.markdown("### ⚙️ Cấu hình")
    api_key = st.text_input("Nhập Gemini API Key:", type="password")
    st.markdown("---")
    if st.button("🗑️ Xóa Lịch Sử Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị tin nhắn cũ
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

# ==========================================
# 3. PROMPT & XỬ LÝ (ÉP XUNG TIẾT KIỆM)
# ==========================================
instructions = """
Bạn là Data Analyst. Dùng công cụ sql_db_query truy vấn CSDL. Trả lời NGẮN GỌN THEO ĐÚNG FORMAT NÀY:

[CHART:bar|cot_x|cot_y|Tieu de] (loại: bar, pie, line, scatter)
[PHAN TICH] Viết 1-2 câu insight.
[SQL]
```sql
SELECT ...
"""

if prompt := st.chat_input("Hỏi tôi về dữ liệu..."):
# Thêm câu hỏi vào UI
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"): st.markdown(prompt)

if not api_key:
    st.error("⚠️ Vui lòng nhập API Key ở thanh bên trái!")
    st.stop()

with st.spinner("Đang truy vấn siêu tốc..."):
    try:
        # 🔴 BÍ QUYẾT TIẾT KIỆM: sample_rows_in_table_info=0 giúp prompt nhẹ đi 80%
        db = SQLDatabase.from_uri(DB_URI, sample_rows_in_table_info=0)
        
        # Khởi tạo Gemini 3.6 Flash
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0)
        
        # 🔴 BÍ QUYẾT TIẾT KIỆM 2: max_iterations=2 cấm AI lặp lại quá nhiều lần
        agent = create_sql_agent(llm=llm, db=db, agent_type="zero-shot-react-description", prefix=instructions, max_iterations=2)
        
        # Gọi AI (Không gửi kèm lịch sử để đỡ tốn token)
        raw_answer = agent.invoke({"input": prompt})["output"]
        
        # Lưu lại text gốc
        st.session_state.messages.append({"role": "assistant", "content": raw_answer})
        
        with st.chat_message("assistant"):
            # Dọn rác văn bản để in ra đẹp hơn
            clean_ans = re.sub(r'```.*?```', '', raw_answer, flags=re.DOTALL)
            clean_ans = re.sub(r'\[CHART:.*?\]', '', clean_ans).replace("Final Answer:", "").strip()
            st.markdown(clean_ans)
            
            # Bóc tách SQL và vẽ biểu đồ
            sql_match = re.search(r'```sql\s*(.*?)\s*```', raw_answer, re.IGNORECASE | re.DOTALL)
            chart_match = re.search(r'\[CHART:(.*?)\|(.*?)\|(.*?)\|(.*?)\]', raw_answer)
            
            if sql_match:
                sql_query = sql_match.group(1).strip()
                with st.expander("⚙️ Xem lệnh SQL & Data"):
                    st.code(sql_query, language="sql")
                    df = pd.read_sql(sql_query, create_engine(DB_URI))
                    st.dataframe(df.head(10)) # In 10 dòng đầu
                    
                # Vẽ Plotly
                if chart_match and not df.empty:
                    c_type, x_col, y_col, title = chart_match.groups()
                    try:
                        if c_type == "bar": st.plotly_chart(px.bar(df, x=x_col.strip(), y=y_col.strip(), title=title.strip()), use_container_width=True)
                        elif c_type == "pie": st.plotly_chart(px.pie(df, names=x_col.strip(), values=y_col.strip(), title=title.strip()), use_container_width=True)
                        elif c_type == "line": st.plotly_chart(px.line(df, x=x_col.strip(), y=y_col.strip(), title=title.strip()), use_container_width=True)
                        elif c_type == "scatter": st.plotly_chart(px.scatter(df, x=x_col.strip(), y=y_col.strip(), title=title.strip()), use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Cột biểu đồ AI chọn chưa khớp hoàn toàn với SQL ({e})")
                        
    except Exception as e:
        st.error(f"Lỗi truy xuất (Vui lòng thử lại bằng câu hỏi khác): {e}")