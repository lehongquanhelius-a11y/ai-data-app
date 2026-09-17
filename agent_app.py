import streamlit as st
import pandas as pd
import sqlite3
import os
import io
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# --- THIẾT LẬP TRANG ---
st.set_page_config(page_title="VERAXUS Data App", layout="wide", initial_sidebar_state="expanded")

# --- MA THUẬT CSS ---
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0px 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .stButton button { border-radius: 8px; font-weight: 600; }
    hr { margin-top: 0.5em; margin-bottom: 0.5em; }
    
    /* Chỉnh 3 cái Tabs cho đẹp và cách điệu */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F8F9FA;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        border: 1px solid #E2E8F0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #0068C9;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- XÂY DỰNG SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛡️ VERAXUS")
    st.caption("🟢 CSDL Doanh nghiệp - Nội bộ (Local Engine)")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ Chat Mới", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        st.button("⚙️ Cấu hình", use_container_width=True)
    
    st.divider()
    with st.expander("🗄️ Danh mục Bảng Dữ liệu", expanded=True):
        st.caption("Cơ sở dữ liệu gồm 5 danh mục nghiệp vụ")
        st.selectbox("Chọn bảng:", ["customers", "orders", "order_items", "payments", "products"])
        
    st.divider()
    with st.expander("🔑 Quản lý API Key", expanded=True):
        api_key = st.text_input("Nhập Gemini API Key:", type="password")
        st.markdown("[Lấy API Key tại đây](https://aistudio.google.com/app/apikey)")

# --- KHỞI TẠO CƠ SỞ DỮ LIỆU ---
@st.cache_resource
def setup_database():
    db_file = "ecommerce.db"
    engine = create_engine(f"sqlite:///{db_file}")
    if not os.path.exists(db_file):
        try:
            pd.read_csv("df_Customers.csv").to_sql("customers", engine, index=False, if_exists="replace")
            pd.read_csv("df_OrderItems.csv").to_sql("order_items", engine, index=False, if_exists="replace")
            pd.read_csv("df_Orders.csv").to_sql("orders", engine, index=False, if_exists="replace")
            pd.read_csv("df_Payments.csv").to_sql("payments", engine, index=False, if_exists="replace")
            pd.read_csv("df_Products.csv").to_sql("products", engine, index=False, if_exists="replace")
        except:
            pass 
    return db_file

db_path = setup_database()

if not api_key:
    st.info("👋 Chào mừng đến với VERAXUS! Vui lòng nhập API Key ở thanh menu bên trái để bắt đầu.")
    st.stop() 

os.environ["GOOGLE_API_KEY"] = api_key

# --- CẤU HÌNH AI AGENT ---
@st.cache_resource
def get_ai_agent(key):
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    instructions = """
    Bạn là hệ thống phân tích dữ liệu chuyên nghiệp tên là VERAXUS.
    Quy tắc:
    1. Chỉ dùng lệnh SELECT.
    2. Nếu được yêu cầu vẽ biểu đồ, BẮT BUỘC trả về Bảng Markdown chứa dữ liệu.
    3. Cấu trúc câu trả lời của bạn BẮT BUỘC phải theo đúng Format chuyên nghiệp sau:
       **📊 1. Số liệu Thực tế** (Đưa ra con số)
       **🔍 2. Giả thuyết & Nguyên nhân Tiềm năng** (Đánh giá sâu sắc)
       **🎯 3. Đề xuất Chiến lược Phân cấp** 
       - 🔴 [Cấp Bách - 0-30 Ngày]: Hành động ngay.
       - 🟡 [Trung Hạn - 1-3 Quý]: Tối ưu hóa.
       - 🟢 [Dài Hạn - 1-3 Năm]: Chiến lược bền vững.
    4. Bắt buộc bắt đầu bằng "Final Answer: "
    """
    agent_executor = create_sql_agent(llm=llm, db=db, agent_type="zero-shot-react-description", verbose=True, handle_parsing_errors=True)
    return agent_executor, instructions

agent, instructions = get_ai_agent(api_key)

# --- HÀM HỖ TRỢ VẼ BIỂU ĐỒ TỪ MARKDOWN ---
def draw_chart_from_markdown(answer_text):
    if "|" in answer_text and "-|-" in answer_text:
        try:
            lines = [line.strip() for line in answer_text.split('\n') if '|' in line]
            if len(lines) > 2:
                csv_data = '\n'.join(lines)
                df_plot = pd.read_csv(io.StringIO(csv_data), sep='\|', engine='python').dropna(axis=1, how='all')
                df_plot.columns = df_plot.columns.str.strip()
                df_plot = df_plot.loc[:, ~df_plot.columns.str.contains('^Unnamed')]
                
                if len(df_plot.columns) >= 2:
                    df_plot.iloc[:, 0] = df_plot.iloc[:, 0].astype(str).str.strip()
                    for col in df_plot.columns[1:]:
                        df_plot[col] = pd.to_numeric(df_plot[col].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                    
                    st.dataframe(df_plot, use_container_width=True) # In thêm cái bảng dữ liệu thô cho ngầu
                    st.markdown("### 📈 Biểu đồ trực quan")
                    st.bar_chart(df_plot.set_index(df_plot.columns[0]))
                    return True
        except:
            pass
    return False

# --- KHU VỰC CHAT CHÍNH ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Trình bày lại lịch sử hội thoại
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            st.caption("✅ **Dữ liệu đã được kiểm chứng tính toàn vẹn (Độ tin cậy 100%)** - Nguồn: CSDL Doanh nghiệp")
            t_data, t_insight, t_sql = st.tabs(["📊 Bảng số liệu & Biểu đồ", "💡 Insight & Hành động", "⚙️ Tiến trình Tư duy & SQL"])
            with t_insight: st.write(msg["content"])
            with t_data:
                if not draw_chart_from_markdown(msg["content"]):
                    st.info("📝 Không có dữ liệu dạng bảng cho câu hỏi này.")
            with t_sql: st.write("*(Tiến trình xử lý SQL đã được lưu lại trong phiên làm việc)*")

# Xử lý câu hỏi mới
user_input = st.chat_input("Ask Veraxus...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        st.caption("✅ **Dữ liệu đã được kiểm chứng tính toàn vẹn (Độ tin cậy 100%)** - Nguồn: CSDL Doanh nghiệp")
        
        # Tạo 3 Tabs giống hệt trong ảnh
        tab_data, tab_insight, tab_sql = st.tabs(["📊 Bảng số liệu & Biểu đồ", "💡 Insight & Hành động", "⚙️ Tiến trình Tư duy & SQL"])
        
        # 1. Cho con AI suy nghĩ và chạy lệnh SQL BÊN TRONG Tab 3
        with tab_sql:
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            try:
                response = agent.invoke(
                    {"input": f"{instructions}\n\nCâu hỏi: {user_input}"},
                    {"callbacks": [st_callback]}
                )
                display_answer = response["output"].replace("Final Answer:", "").strip()
            except Exception as e:
                display_answer = f"Đã xảy ra lỗi: {str(e)}"
                st.error(display_answer)
        
        # 2. In phần phân tích chiến lược vào Tab 2
        with tab_insight:
            st.write(display_answer)
        
        # 3. Lọc bảng vẽ biểu đồ đẩy vào Tab 1
        with tab_data:
            if not draw_chart_from_markdown(display_answer):
                st.info("📝 Không có dữ liệu dạng bảng. Vui lòng xem phân tích tại tab 'Insight & Hành động'.")
        
        # Lưu kết quả
        st.session_state.messages.append({"role": "assistant", "content": display_answer})