import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
import re

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="My AI agent", page_icon="🛒", layout="wide")
st.title("🛒 My AI agent")
st.markdown("Trợ lý AI phân tích dữ liệu, săn Insight & Hoạch định Chiến lược")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. KHU VỰC CẤU HÌNH & SIDEBAR (CÓ BRANDING NHÓM)
# ==========================================
with st.sidebar:
    # Đặt Brand name lên đỉnh cao nhất của thanh trái
    st.markdown("### 🔥 Group 3 - TINE313")
    st.markdown("---")
    
    # Hàng nút bấm trên cùng
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Chat Mới", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        with st.popover("⚙️ Cấu hình"):
            google_api_key = st.text_input("Gemini API Key:", type="password")
            st.markdown("[👉 Lấy API Key tại đây](https://aistudio.google.com/app/apikey)")
            
            st.markdown("---")
            mysql_host = st.text_input("MySQL Host:", value="")
            mysql_user = st.text_input("Username:", value="")
            mysql_pass = st.text_input("Password:", type="password")
            mysql_db = st.text_input("Database:", value="")

    st.markdown("---")
    
    # Danh mục Bảng dữ liệu
    st.markdown("📂 **Danh mục Bảng Dữ liệu**")
    st.caption("Cơ sở dữ liệu gồm 5 danh mục nghiệp vụ:")
    with st.expander("Hiển thị chi tiết bảng"):
        st.markdown("""
        - **df_customers** (Khách hàng)
        - **df_orders** (Đơn hàng trung tâm)
        - **df_orderitems** (Chi tiết giao hàng)
        - **df_products** (Sản phẩm)
        - **df_payments** (Thanh toán)
        """)

    st.markdown("---")
    
    # Lịch sử hội thoại
    st.markdown("🕒 **Lịch sử Hội thoại**")
    st.caption("Bấm vào câu hỏi để xem lại kết quả tức thì")
    
    has_history = False
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            has_history = True
            st.button(f"💬 {msg['content'][:30]}...", key=f"hist_{i}", use_container_width=True)
            
    if not has_history:
        st.info("Chưa có lịch sử trò chuyện.")
        
    if st.button("🗑️ Dọn dẹp lịch sử", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 3. BỘ NÃO CHIẾN LƯỢC & ÉP KHUÔN ĐẦU RA
# ==========================================
instructions = """
# VAI TRÒ
Bạn là một Giám đốc Vận hành (COO) và Kỹ sư Dữ liệu cấp cao làm việc cho một hệ thống E-commerce Marketplace.

# RÀNG BUỘC KỸ THUẬT
Nối bảng bắt buộc dùng df_orders làm cầu nối. Ưu tiên SUM(payment_value) cho doanh thu, loại trừ đơn Cancelled. Luôn dùng SQL để trích xuất số liệu thực tế.

# ĐỊNH DẠNG ĐẦU RA BẮT BUỘC
Để hệ thống render UI, bạn BẮT BUỘC xuất kết quả theo cấu trúc sau:

[BIỂU ĐỒ]
(Cung cấp mã python dùng streamlit, pandas, matplotlib. Bọc code trong ```python...```. KHÔNG giải thích thêm ở phần này)

[PHÂN TÍCH]
(Trình bày số liệu tổng quan và Insight nghịch lý tại đây)

[CHIẾN LƯỢC]
(Đề xuất chiến lược Cấp bách, Trung hạn, Dài hạn)

[SQL]
(Cung cấp câu lệnh SQL bạn đã sử dụng, bọc trong ```sql...```)
"""

# ==========================================
# 4. KHỞI TẠO TÁC NHÂN (FIX 429 & 404)
# ==========================================
def get_agent():
    if not google_api_key:
        return None, "Vui lòng nhập API Key trong mục Cấu hình."
    try:
        if mysql_host and mysql_user and mysql_db:
            pwd_part = f":{mysql_pass}" if mysql_pass else ""
            db_uri = f"mysql+pymysql://{mysql_user}{pwd_part}@{mysql_host}:3306/{mysql_db}"
        else:
            db_uri = "sqlite:///ecommerce.db"
            
        db = SQLDatabase.from_uri(db_uri)
        
        # Phiên bản lõi ổn định, không lỗi 404
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, temperature=0.2)
        
        agent_executor = create_sql_agent(
            llm=llm, 
            toolkit=None, 
            db=db,
            agent_type="zero-shot-react-description", 
            prefix=instructions, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=4  # Giới hạn số vòng suy nghĩ để không chạm trần API
        )
        return agent_executor, "OK"
    except Exception as e:
        return None, str(e)

# ==========================================
# 5. GIAO DIỆN HIỂN THỊ 
# ==========================================
def render_assistant_response(answer):
    code_blocks = re.findall(r'```python(.*?)```', answer, re.DOTALL)
    sql_blocks = re.findall(r'```sql(.*?)