import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
import re
import pandas as pd
from sqlalchemy import create_engine
import json
import os
import uuid
import sqlite3

# ==========================================
# 0. KHỞI TẠO ĐƯỜNG DẪN & DATABASE SQLITE
# ==========================================
# Ép đường dẫn tuyệt đối để tránh lỗi "no such table"
DB_PATH = os.path.abspath("ecommerce.db").replace('\\', '/')
DB_URI_SQLITE = f"sqlite:///{DB_PATH}"

def init_sqlite_db():
    """Tự động tạo file ecommerce.db và 5 bảng dữ liệu nếu file chưa tồn tại"""
    if not os.path.exists("ecommerce.db"):
        conn = sqlite3.connect("ecommerce.db")
        
        # Tạo data giả lập
        df_customers = pd.DataFrame({
            'customer_id': ['C1', 'C2', 'C3', 'C4', 'C5'], 
            'customer_city': ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'São Paulo', 'Curitiba']
        })
        df_orders = pd.DataFrame({
            'order_id': ['O1', 'O2', 'O3', 'O4', 'O5'], 
            'customer_id': ['C1', 'C2', 'C3', 'C4', 'C5'], 
            'order_status': ['delivered', 'canceled', 'delivered', 'delivered', 'delivered']
        })
        df_payments = pd.DataFrame({
            'order_id': ['O1', 'O2', 'O3', 'O4', 'O5'], 
            'payment_value': [150.5, 200.0, 99.9, 350.0, 45.0]
        })
        df_products = pd.DataFrame({
            'product_id': ['P1', 'P2', 'P3'], 
            'product_category_name': ['Electronics', 'Fashion', 'Home']
        })
        df_orderitems = pd.DataFrame({
            'order_id': ['O1', 'O2', 'O3', 'O4', 'O5'], 
            'product_id': ['P1', 'P2', 'P3', 'P1', 'P2'], 
            'price': [150.5, 200.0, 99.9, 350.0, 45.0]
        })
        
        # Đổ data vào Database
        df_customers.to_sql('df_customers', conn, index=False, if_exists='replace')
        df_orders.to_sql('df_orders', conn, index=False, if_exists='replace')
        df_payments.to_sql('df_payments', conn, index=False, if_exists='replace')
        df_products.to_sql('df_products', conn, index=False, if_exists='replace')
        df_orderitems.to_sql('df_orderitems', conn, index=False, if_exists='replace')
        conn.close()

init_sqlite_db()

# ==========================================
# 0. QUẢN LÝ LỊCH SỬ HỘI THOẠI (MULTI-SESSION)
# ==========================================
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    if len(data) > 0:
                        return {str(uuid.uuid4()): data}
                    return {}
                return data
        except:
            return {}
    return {}

def save_history(all_chats):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chats, f, ensure_ascii=False, indent=4)

def clear_all_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.session_state.all_chats = {}
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.all_chats[st.session_state.current_session_id] = []

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="My AI agent", page_icon="🛒", layout="wide")
st.title("🛒 My AI agent")
st.markdown("Trợ lý AI phân tích dữ liệu, săn Insight & Hoạch định Chiến lược")
st.markdown("🔥 **Agent phát triển bởi: Group 3 - TINE313** 🔥")

if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_history()

if "current_session_id" not in st.session_state:
    if st.session_state.all_chats:
        st.session_state.current_session_id = list(st.session_state.all_chats.keys())[-1]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.all_chats[new_id] = []

# ==========================================
# 2. KHU VỰC CẤU HÌNH & SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 🔥 Group 3 - TINE313")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Chat Mới", type="primary", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.current_session_id = new_id
            st.session_state.all_chats[new_id] = []
            st.rerun()
    with col2:
        with st.popover("⚙️ Cấu hình"):
            google_api_key = st.text_input("Gemini API Key:", type="password", key="api_key")
            st.markdown("[👉 Lấy API Key tại đây](https://aistudio.google.com/app/apikey)")
            st.markdown("---")
            st.caption("Để trống MySQL nếu muốn dùng file ecommerce.db")
            mysql_host = st.text_input("MySQL Host:", key="db_host")
            mysql_user = st.text_input("Username:", key="db_user")
            mysql_pass = st.text_input("Password:", type="password", key="db_pass")
            mysql_db = st.text_input("Database:", key="db_name")

    st.markdown("---")
    st.markdown("📂 **Danh mục Bảng Dữ liệu**")
    with st.expander("Hiển thị chi tiết bảng"):
        st.markdown("""
        - **df_customers** (Khách hàng)
        - **df_orders** (Đơn hàng trung tâm)
        - **df_orderitems** (Chi tiết giao hàng)
        - **df_products** (Sản phẩm)
        - **df_payments** (Thanh toán)
        """)

    st.markdown("---")
    st.markdown("🕒 **Lịch sử Hội thoại**")
    
    has_history = False
    for session_id, chat_messages in reversed(st.session_state.all_chats.items()):
        if len(chat_messages) > 0:
            has_history = True
            title = "Tin nhắn mới..."
            for m in chat_messages:
                if m["role"] == "user":
                    title = m["content"][:22] + "..."
                    break
            
            is_active = (session_id == st.session_state.current_session_id)
            btn_label = f"👉 {title}" if is_active else f"💬 {title}"
            
            if st.button(btn_label, key=f"hist_{session_id}", use_container_width=True):
                st.session_state.current_session_id = session_id
                st.rerun()
            
    if not has_history:
        st.info("Chưa có lịch sử trò chuyện.")

    if st.button("🗑️ Dọn dẹp TOÀN BỘ lịch sử", use_container_width=True):
        clear_all_history()
        st.rerun()

# ==========================================
# 3. BỘ NÃO CHIẾN LƯỢC & ÉP KHUÔN ĐẦU RA 
# ==========================================
instructions_raw = """
# VAI TRÒ
Bạn là Giám đốc Vận hành (COO) & Kỹ sư Dữ liệu cấp cao tại một E-commerce Marketplace.

# BẢN ĐỒ CƠ SỞ DỮ LIỆU
- df_orders: Bảng trung tâm. (order_id, customer_id, order_status).
- df_customers: (customer_id, customer_city). Join với df_orders qua customer_id.
- df_payments: (order_id, payment_value). Join với df_orders qua order_id.
- df_products: (product_id, product_category_name).
- df_orderitems: (order_id, product_id, price). Join df_orders qua order_id, df_products qua product_id.

# QUY TRÌNH VẬN HÀNH BẮT BUỘC (SOP)
1. TÌM KIẾM SỰ THẬT: BẠN BẮT BUỘC phải dùng công cụ sql_db_query để truy vấn CSDL. TUYỆT ĐỐI KHÔNG tự bịa số liệu.
2. NỐI BẢNG: Luôn dùng df_orders làm trung tâm. Tính doanh thu bằng SUM(payment_value), loại trừ đơn Cancelled.

# ĐỊNH DẠNG ĐẦU RA BẮT BUỘC (FINAL ANSWER):
Khi bạn đã có kết quả cuối cùng, bạn BẮT BUỘC phải bắt đầu bằng cụm từ "Final Answer: " sau đó mới đến các thẻ. Không được thiếu thẻ nào.

Final Answer:
[BIỂU ĐỒ]
(BẮT BUỘC gộp toàn bộ code khai báo dữ liệu và vẽ biểu đồ vào DUY NHẤT 1 khối '''python. TUYỆT ĐỐI KHÔNG chia nhỏ thành nhiều khối!)
'''python
# code streamlit, matplotlib gom hết vào đây
'''

[PHÂN TÍCH]
(Trình bày phân tích bằng Markdown sắc nét, chia làm 2 ý rõ ràng:
1. Insight cơ bản: Đọc vị các con số tổng quan, xu hướng chính, phân bổ tỷ trọng bề nổi.
2. Insight nghịch lý/chuyên sâu: Phát hiện điểm bất thường, rủi ro ngầm, hoặc cơ hội ẩn giấu đằng sau những con số đó.)

[CHIẾN LƯỢC]
(BẮT BUỘC trình bày bằng Bullet Points, chia thành 3 mục rõ ràng dựa trên insight:
* Chiến lược Ngắn hạn (Cấp bách): ...
* Chiến lược Trung hạn: ...
* Chiến lược Dài hạn: ...)

[SQL]
'''sql
-- Dán câu lệnh SQL đã chạy thành công
'''
"""
instructions = instructions_raw.replace("'''", "```")

# ==========================================
# 4. WORKFLOW KIỂM ĐỊNH & KẾT NỐI DATABASE
# ==========================================
def get_db_uri():
    host = st.session_state.get("db_host", "")
    user = st.session_state.get("db_user", "")
    pwd = st.session_state.get("db_pass", "")
    db_name = st.session_state.get("db_name", "")
    
    if host and user and db_name:
        pwd_part = f":{pwd}" if pwd else ""
        return f"mysql+pymysql://{user}{pwd_part}@{host}:3306/{db_name}"
    return DB_URI_SQLITE

def run_data_audit(db_uri):
    engine = create_engine(db_uri)
    audit_logs = []
    try:
        df_pay = pd.read_sql("SELECT order_id, payment_value FROM df_payments WHERE payment_value < 0", engine)
        if not df_pay.empty:
            audit_logs.append({"status": "error", "msg": f"❌ df_payments: Phát hiện {len(df_pay)} giao dịch có giá trị âm (Lỗi hệ thống ghi nhận)."})
        else:
            audit_logs.append({"status": "success", "msg": "✅ df_payments: 100% giao dịch có giá trị dương hợp lệ."})
            
        df_ord = pd.read_sql("SELECT order_id FROM df_orders WHERE order_status IS NULL OR order_status = ''", engine)
        if not df_ord.empty:
            audit_logs.append({"status": "warning", "msg": f"⚠️ df_orders: Phát hiện {len(df_ord)} đơn hàng bị trống (Null) trạng thái."})
        else:
            audit_logs.append({"status": "success", "msg": "✅ df_orders: Toàn vẹn dữ liệu trạng thái đơn hàng."})
            
    except Exception as e:
        audit_logs.append({"status": "warning", "msg": f"⚠️ Bỏ qua kiểm định sâu do CSDL chưa khởi tạo đầy đủ."})
        
    return audit_logs

# ==========================================
# 5. KHỞI TẠO TÁC NHÂN
# ==========================================
def get_agent():
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        return None, "Vui lòng nhập API Key trong mục Cấu hình."
    try:
        db_uri = get_db_uri()
        db = SQLDatabase.from_uri(db_uri)
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0.1)
        
        agent_executor = create_sql_agent(
            llm=llm, 
            toolkit=None, 
            db=db,
            agent_type="zero-shot-react-description", 
            prefix=instructions, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=15 
        )
        return agent_executor, "OK"
    except Exception as e:
        return None, str(e)

# ==========================================
# 6. GIAO DIỆN HIỂN THỊ
# ==========================================
def render_assistant_response(answer, audit_logs=None):
    answer = answer.replace("`", "") if answer.startswith("`") else answer
    
    code_blocks = re.findall(r'