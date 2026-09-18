import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
import re

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT (UI/UX)
# ==========================================
st.set_page_config(page_title="E-Commerce AI Agent", page_icon="🛒", layout="wide")
st.title("🛒 E-Commerce Supply Chain AI Agent")
st.markdown("Trợ lý AI phân tích dữ liệu, săn Insight & Hoạch định Chiến lược")

# Khởi tạo bộ nhớ Lịch sử Chat (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. KHU VỰC CẤU HÌNH (SIDEBAR & EXPANDERS)
# ==========================================
with st.sidebar:
    # Bọc toàn bộ ô nhập liệu vào trong 1 cái Expander có thể click
    with st.expander("⚙️ Cấu hình Hệ thống (Click để mở/đóng)", expanded=True):
        google_api_key = st.text_input("Google Gemini API Key:", type="password")
        
        st.markdown("---")
        st.write("🔌 **KẾT NỐI DATABASE**")
        st.caption("Nếu để trống, hệ thống tự dùng bản Offline (SQLite)")
        mysql_host = st.text_input("Host (VD: localhost):", value="")
        mysql_user = st.text_input("Username:", value="")
        mysql_pass = st.text_input("Password:", type="password")
        mysql_db = st.text_input("Tên Database:", value="")
    
    st.markdown("---")
    if st.button("🗑️ Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()

# Sách Hướng Dẫn Dữ Liệu
with st.expander("📖 Xem cấu trúc Dữ liệu (ERD & Từ điển) - Click để mở"):
    st.markdown("""
    - **df_customers:** `customer_id` (PK), `customer_zip_code_prefix`, `customer_city`, `customer_state`.
    - **df_orders:** `order_id` (PK), `customer_id` (FK), `order_purchase_timestamp`, `order_approved_at`, `order_delivered_timestamp`, `order_estimated_delivery_date`.
    - **df_orderitems:** `order_id` (PK/FK), `product_id` (PK/FK), `seller_id`, `price` (giá SP), `shipping_charges` (phí ship).
    - **df_products:** `product_id` (PK), `product_category_name`, `product_weight_g`, `product_length_cm`, v.v.
    - **df_payments:** `order_id` (PK/FK), `payment_sequential`, `payment_type`, `payment_installments`, `payment_value`.
    *Chú ý: Đơn hàng kết nối với Sản phẩm và Thanh toán thông qua bảng trung tâm `df_orders`.*
    """)

# ==========================================
# 3. BỘ NÃO CHIẾN LƯỢC (SYSTEM PROMPT)
# ==========================================
instructions = """
# VAI TRÒ
Bạn là một Giám đốc Vận hành (COO) và Kỹ sư Dữ liệu cấp cao làm việc cho một hệ thống E-commerce Marketplace. Nhiệm vụ của bạn là viết truy vấn SQL chính xác, tìm Insight nghịch lý và Đề xuất chiến lược thực chiến.

# BỐI CẢNH & LUỒNG VẬN HÀNH (8 BƯỚC)
- Mô hình: Marketplace trung gian, không tự sản xuất.
- Luồng: (1) Đặt hàng -> (2) Check tồn kho -> (3) Tạo đơn -> (4) Xử lý ngoại lệ -> (5) Hủy/Hoàn tiền -> (6) Đóng gói tại kho -> (7) Bàn giao 3PL -> (8) Phân tích.

# TỪ ĐIỂN DỮ LIỆU & RÀNG BUỘC (GUARDRAILS)
1. Cú pháp: Hệ thống sử dụng SQL chuẩn. Nếu trường thời gian là chuỗi, hãy dùng hàm chuyển đổi phù hợp.
2. Nối bảng: Bắt buộc dùng df_orders làm cầu nối, tuyệt đối không JOIN trực tiếp df_customers với df_products/payments.
3. Doanh thu: Ưu tiên SUM(payment_value). Loại trừ các đơn Cancelled hoặc Failed_Delivery khi tính doanh thu kinh doanh. Tuyệt đối không dùng SUM/AVG lên zipcode.

# TƯ DUY PHÂN TÍCH ĐỘT PHÁ (SĂN NGHỊCH LÝ)
Khung phân tích 3 trụ cột: (1) Trải nghiệm KH; (2) Hiệu suất Sản phẩm; (3) Tài chính dòng tiền.
Bạn BẮT BUỘC phải tìm các "Nghịch lý" (Counter-intuitive Insights) và trình bày theo 3 bước:
- Logic thường: "Thông thường..."
- Sự thật từ Data: "Tuy nhiên, dữ liệu thực tế..."
- Nguyên nhân: "Giả thuyết cốt lõi là..."

# ĐỀ XUẤT CHIẾN LƯỢC KINH DOANH (THỰC CHIẾN, KHÔNG ẢO GIÁC)
Dựa trên Insight, BẮT BUỘC đề xuất giải pháp THỰC TẾ, CỤ THỂ gắn liền với Data vừa tìm được (Ví dụ: Không nói "Tối ưu phí ship", mà phải nói "Đàm phán lại phí 3PL cho mặt hàng nội thất trên 10kg tại bang NY"). Phân loại chặt chẽ theo 3 mốc thời gian:
1. Chiến lược Cấp bách (Ngay lập tức/1-3 tháng): Hành động dập lửa, vá lỗ hổng doanh thu, xử lý các seller/danh mục đang gây thiệt hại trực tiếp.
2. Chiến lược Trung hạn (3-6 tháng): Tối ưu quy trình, dòng tiền (VD: Sửa đổi chính sách trả góp, tái phân bổ kho bãi địa phương).
3. Chiến lược Dài hạn (6-12 tháng): Chuyển dịch mô hình, đầu tư hạ tầng mở rộng, hoặc phát triển công nghệ dự báo.

# QUY TẮC TRỰC QUAN HÓA
Nếu yêu cầu vẽ biểu đồ, xuất mã Python dùng `streamlit`, `pandas`, `matplotlib`/`seaborn`.
- Line chart cho xu hướng. Bar chart cho so sánh. Pie/Stacked bar cho tỷ trọng. Scatter cho tương quan.
"""

# ==========================================
# 4. KHỞI TẠO KẾT NỐI (CÓ CƠ CHẾ DỰ PHÒNG) & AGENT
# ==========================================
def get_agent():
    if not google_api_key:
        return None, "Vui lòng nhập Google Gemini API Key bên thanh cấu hình."
    
    try:
        # Cơ chế ưu tiên: Nếu nhập đủ thông tin MySQL thì dùng MySQL
        if mysql_host and mysql_user and mysql_db:
            pwd_part = f":{mysql_pass}" if mysql_pass else ""
            db_uri = f"mysql+pymysql://{mysql_user}{pwd_part}@{mysql_host}:3306/{mysql_db}"
            st.sidebar.success("Đang kết nối: MySQL Server")
        else:
            # Fallback: Nếu không nhập MySQL, tự động dùng SQLite nội bộ
            db_uri = "sqlite:///ecommerce.db"
            st.sidebar.info("Đang kết nối: SQLite Local (Dự phòng)")
            
        db = SQLDatabase.from_uri(db_uri)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=google_api_key,
            temperature=0.2 
        )
        
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=None,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            prefix=instructions,
            verbose=True,
            handle_parsing_errors=True
        )
        return agent_executor, "OK"
    except Exception as e:
        return None, f"Lỗi kết nối cơ sở dữ liệu: {e}"

# ==========================================
# 5. GIAO DIỆN TRÒ CHUYỆN (CHAT INTERFACE)
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "```python" in msg["content"]:
            code_blocks = re.findall(r'```python(.*?)```', msg["content"], re.DOTALL)
            for code in code_blocks:
                try:
                    exec(code)
                except Exception:
                    pass

if prompt := st.chat_input("VD: Phân tích nghịch lý giao hàng và đề xuất chiến lược 3 giai đoạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent, status = get_agent()
    
    with st.chat_message("assistant"):
        if agent is None:
            st.error(status)
        else:
            with st.spinner("Đang truy xuất Database, tìm nghịch lý và lên chiến lược..."):
                try:
                    response = agent.invoke({"input": prompt})
                    answer = response["output"]
                    
                    st.markdown(answer)
                    
                    code_blocks = re.findall(r'```python(.*?)```', answer, re.DOTALL)
                    for code in code_blocks:
                        try:
                            exec(code)
                        except Exception as code_error:
                            st.error(f"Lỗi khi vẽ biểu đồ: {code_error}")
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Đã có lỗi xảy ra: {e}")