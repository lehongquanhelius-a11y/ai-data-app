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

st.set_page_config(page_title="AI Data Agent", layout="centered")
st.title("🤖 Chat với CSDL E-commerce")

# --- 1. TẠO Ô NHẬP API KEY TRÊN GIAO DIỆN WEB ---
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    api_key = st.text_input("Nhập Gemini API Key của bạn vào đây:", type="password")
    st.markdown("👉 [Bấm vào đây để lấy API Key mới](https://aistudio.google.com/app/apikey)")

# --- 2. CHUYỂN DỮ LIỆU CSV THÀNH SQLITE DATABASE ---
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
            print("✅ Đã kết nối thành công 5 bảng dữ liệu!") 
        except Exception as e:
            print(f"Lỗi đọc file dữ liệu: {e}") 
    return db_file

db_path = setup_database()

# --- 3. KIỂM TRA KEY & KHỞI TẠO AGENT ---
if not api_key:
    st.warning("👈 Vui lòng nhập API Key ở thanh menu bên trái để bắt đầu!")
    st.stop() 

os.environ["GOOGLE_API_KEY"] = api_key

@st.cache_resource
def get_ai_agent(key):
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    
    # BỘ NÃO ĐÃ ĐƯỢC NÂNG CẤP TỐI ĐA
    instructions = """
    Bạn là một Chuyên gia phân tích dữ liệu (Data Analyst) kiêm Chiến lược gia kinh doanh (Business Strategist). 
    Bạn có quyền truy cập vào CSDL SQLite chứa các bảng: customers, order_items, orders, payments, products.
    
    Quy tắc hoạt động:
    1. Chỉ dùng lệnh SELECT để truy vấn dữ liệu.
    2. KHÔNG tự tạo mã code Python để vẽ.
    3. Trình bày câu trả lời bằng tiếng Việt, bắt buộc chia thành 3 phần rõ ràng:
       - 📊 Số liệu thực tế: Trả lời trực tiếp câu hỏi bằng con số chính xác.
       - 💡 Insights: Đánh giá số liệu này nói lên điều gì về hành vi hoặc xu hướng.
       - 🚀 Chiến lược đề xuất: Đưa ra các hành động cụ thể để cải thiện kinh doanh.
    4. QUAN TRỌNG: Bắt buộc phải bắt đầu toàn bộ câu trả lời bằng "Final Answer: "
    5. ĐẶC BIỆT: Nếu người dùng yêu cầu "vẽ biểu đồ" hoặc "thống kê trực quan", bạn BẮT BUỘC phải cung cấp một Bảng Markdown chứa dữ liệu đó trước khi đưa ra Insights.
    """
    
    agent_executor = create_sql_agent(
        llm=llm, db=db, agent_type="zero-shot-react-description",
        verbose=True, handle_parsing_errors=True
    )
    return agent_executor, instructions

agent, instructions = get_ai_agent(api_key)

# --- 4. QUẢN LÝ LỊCH SỬ HỘI THOẠI & VẼ BIỂU ĐỒ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Hỏi bất kỳ điều gì về dữ liệu...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        try:
            response = agent.invoke(
                {"input": f"{instructions}\n\nCâu hỏi: {user_input}"},
                {"callbacks": [st_callback]}
            )
            answer = response["output"]
            
            # Xóa chữ Final Answer đi cho đẹp giao diện
            display_answer = answer.replace("Final Answer:", "").strip()
            st.write(display_answer)
            st.session_state.messages.append({"role": "assistant", "content": display_answer})
            
            # --- MẮT THẦN TỰ ĐỘNG VẼ BIỂU ĐỒ ---
            if "|" in answer and "-|-" in answer:
                try:
                    lines = [line.strip() for line in answer.split('\n') if line.strip().startswith('|')]
                    if len(lines) > 2:
                        csv_data = '\n'.join(lines)
                        df_plot = pd.read_csv(io.StringIO(csv_data), sep='|').dropna(axis=1, how='all')
                        df_plot.columns = df_plot.columns.str.strip()
                        
                        st.markdown("### 📈 Biểu đồ trực quan")
                        st.bar_chart(df_plot.set_index(df_plot.columns[0]))
                except Exception as e:
                    pass
            # -----------------------------------
            
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {str(e)}")