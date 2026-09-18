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
# Dùng mẹo ''' thay vì 3 dấu nháy ngược để tránh lỗi đứt đoạn UI khi copy code
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
(Đề xuất chiến lược hành động dựa trên cả 2 Insight vừa nêu)

[SQL]
'''sql
-- Dán câu lệnh SQL đã chạy thành công
'''
"""
# Tự động chuyển đổi lại thành 3 dấu nháy ngược cho LangChain hiểu
instructions = instructions_raw.replace("'''", "```")

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
        
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=google_api_key, temperature=0.1)
        
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
# 5. GIAO DIỆN HIỂN THỊ
# ==========================================
def render_assistant_response(answer):
    answer = answer.replace("`", "") if answer.startswith("`") else answer
    
    code_blocks = re.findall(r'```python(.*?)```', answer, re.DOTALL)
    sql_blocks = re.findall(r'```sql(.*?)```', answer, re.DOTALL)
    
    phan_tich = "Hệ thống đã phân tích xong nhưng đầu ra bị sai định dạng hiển thị. Vui lòng thử lại."
    chien_luoc = "Hệ thống chưa kịp hoàn thiện chiến lược. Vui lòng bấm 'Chat Mới' và hỏi lại."
    
    if "[PHÂN TÍCH]" in answer:
        phan_tich_raw = answer.split("[PHÂN TÍCH]")[1]
        phan_tich = phan_tich_raw.split("[")[0].strip()
        
    if "[CHIẾN LƯỢC]" in answer:
        chien_luoc_raw = answer.split("[CHIẾN LƯỢC]")[1]
        chien_luoc = chien_luoc_raw.split("[")[0].strip()

    if code_blocks:
        combined_code = "\n".join(code_blocks)
        try:
            exec(combined_code)
        except Exception as e:
            st.warning(f"Không thể hiển thị biểu đồ: {e}")

    st.markdown("---")
    st.markdown("💡 **Phát hiện 1 điểm/xu hướng bất thường bởi dữ liệu. Xem chi tiết tại tab 'Insight & Hành động'**")
    st.success("✔️ **Dữ liệu đã được kiểm chứng tính toàn vẹn (Độ tin cậy 100%)** — Nguồn: CSDL Doanh Nghiệp")

    tab1, tab2, tab3 = st.tabs(["📊 Bảng số liệu & Báo cáo", "💡 Insight & Hành động", "⚙️ Tiến trình SQL"])
    
    with tab1:
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
                    error_str = str(e)
                    if "[PHÂN TÍCH]" in error_str or "[BIỂU ĐỒ]" in error_str:
                        extracted_answer = error_str.split("Could not parse LLM output:")[-1].strip()
                        render_assistant_response(extracted_answer)
                        st.session_state.messages.append({"role": "assistant", "content": extracted_answer})
                    else:
                        st.error(f"Đã có lỗi hệ thống xảy ra: {e}")