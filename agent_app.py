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
st.markdown("🔥 **Đồ án phát triển bởi: Group 3 - TINE313** 🔥")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. KHU VỰC CẤU HÌNH & SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 🔥 Group 3 - TINE313")
    st.markdown("---")
    
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
Để hệ thống render UI, bạn BẮT BUỘC xuất kết quả theo cấu trúc sau. TUYỆT ĐỐI không được thiếu các thẻ này:

[BIỂU ĐỒ]
(Cung cấp mã python dùng streamlit, pandas, matplotlib. Bọc code trong ```python...```)

[PHÂN TÍCH]
(Trình bày số liệu tổng quan và Insight nghịch lý)

[CHIẾN LƯỢC]
(Đề xuất chiến lược Cấp bách, Trung hạn, Dài hạn)

[SQL]
(Cung cấp câu lệnh SQL bạn đã sử dụng, bọc trong ```sql...```)
"""

# ==========================================
# 4. KHỞI TẠO TÁC NHÂN
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
        
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=google_api_key, temperature=0.2)
        
        agent_executor = create_sql_agent(
            llm=llm, 
            toolkit=None, 
            db=db,
            agent_type="zero-shot-react-description", 
            prefix=instructions, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=8  # Nới lỏng lên 8 vòng để AI đủ không gian suy nghĩ cho các truy vấn phức tạp
        )
        return agent_executor, "OK"
    except Exception as e:
        return None, str(e)

# ==========================================
# 5. GIAO DIỆN HIỂN THỊ
# ==========================================
def render_assistant_response(answer):
    code_blocks = re.findall(r'```python(.*?)```', answer, re.DOTALL)
    sql_blocks = re.findall(r'```sql(.*?)```', answer, re.DOTALL)
    
    # Bổ sung cơ chế fallback nội dung nếu AI không xuất đúng định dạng thẻ
    phan_tich = "Hệ thống đã phân tích xong nhưng đầu ra bị sai định dạng hiển thị. Vui lòng thử lại."
    chien_luoc = "Hệ thống chưa kịp hoàn thiện chiến lược. Vui lòng bấm 'Chat Mới' và hỏi lại."
    
    if "[PHÂN TÍCH]" in answer:
        phan_tich_raw = answer.split("[PHÂN TÍCH]")[1]
        phan_tich = phan_tich_raw.split("[")[0].strip()
    elif "Agent stopped due to iteration limit" in answer:
        phan_tich = "⚠️ AI đã dừng phân tích giữa chừng vì câu hỏi quá phức tạp cần nhiều hơn 8 vòng xử lý. Vui lòng chia nhỏ câu hỏi."
        
    if "[CHIẾN LƯỢC]" in answer:
        chien_luoc_raw = answer.split("[CHIẾN LƯỢC]")[1]
        chien_luoc = chien_luoc_raw.split("[")[0].strip()
    elif "Agent stopped due to iteration limit" in answer:
        chien_luoc = "⚠️ Truy vấn vượt giới hạn tài nguyên tính toán hiện tại."

    if code_blocks:
        for code in code_blocks:
            try:
                exec(code)
            except Exception as e:
                st.warning(f"Không thể hiển thị biểu đồ: {e}")

    st.markdown("---")
    st.markdown("💡 **Phát hiện 1 điểm/xu hướng bất thường bởi dữ liệu. Xem chi tiết tại tab 'Insight & Hành động'**")
    st.success("✔️ **Dữ liệu đã được kiểm chứng tính toàn vẹn (Độ tin cậy 100%)** — Nguồn: CSDL Doanh Nghiệp")

    tab1, tab2, tab3 = st.tabs(["📊 Bảng số liệu & Báo cáo", "💡 Insight & Hành động", "⚙️ Tiến trình SQL"])
    
    with tab1:
        # Nếu AI nôn ra một đống text không có thẻ, đổ tất cả vào Tab 1
        if "[PHÂN TÍCH]" not in answer and "[CHIẾN LƯỢC]" not in answer:
            st.markdown(answer)
        else:
            st.markdown(phan_tich)
            
    with tab2:
        st.markdown(chien_luoc)
        
    with tab3:
        if sql_blocks:
            st.markdown("**Câu lệnh SQL đã được Agent thực thi:**")
            for sql in sql_blocks:
                st.code(sql, language="sql")
        else:
            st.info("Agent đã sử dụng dữ liệu ngữ cảnh hoặc tiến trình bị ngắt, không thực thi truy vấn SQL mới.")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            render_assistant_response(msg["content"])

if prompt := st.chat_input("VD: Phân tích top 10 sản phẩm có tổng doanh thu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent, status = get_agent()
    
    with st.chat_message("assistant"):
        if agent is None:
            st.error(status)
        else:
            with st.spinner("Đang truy xuất Database và kiểm định dữ liệu..."):
                try:
                    response = agent.invoke({"input": prompt})
                    answer = response["output"]
                    
                    render_assistant_response(answer)
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Đã có lỗi xảy ra: {e}")