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


# ==========================================
# 0. QUẢN LÝ LỊCH SỬ HỘI THOẠI (MULTI-SESSION NHƯ GEMINI)
# ==========================================
HISTORY_FILE = "chat_history.json"


def load_history():
    """Tải lịch sử chat. Hỗ trợ chuyển đổi nếu đang xài bản cũ (dạng List) sang bản mới (dạng Dict)"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Nếu file cũ là dạng List, tự động bọc nó vào một session mới để không bị lỗi
                if isinstance(data, list):
                    if len(data) > 0:
                        return {str(uuid.uuid4()): data}
                    return {}
                return data
        except:
            return {}
    return {}


def save_history(all_chats):
    """Lưu toàn bộ các phiên chat vào file"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chats, f, ensure_ascii=False, indent=4)


def clear_all_history():
    """Xóa trắng toàn bộ dữ liệu"""
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
st.markdown("🔥 **Đồ án phát triển bởi: Group 3 - TINE313** 🔥")


# Khởi tạo dữ liệu Multi-session
if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_history()


if "current_session_id" not in st.session_state:
    # Nếu có lịch sử, chọn cái cuối cùng làm phiên hiện tại. Nếu không, tạo mới.
    if st.session_state.all_chats:
        st.session_state.current_session_id = list(st.session_state.all_chats.keys())[-1]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.all_chats[new_id] = []


# ==========================================
# 2. KHU VỰC CẤU HÌNH & SIDEBAR (CÓ CHUYỂN TAB CHAT)
# ==========================================
with st.sidebar:
    st.markdown("### 🔥 Group 3 - TINE313")
    st.markdown("---")
   
    col1, col2 = st.columns([1, 1])
    with col1:
        # Bấm Chat Mới -> Tạo ID mới -> Lưu vào danh sách -> Làm mới màn hình
        if st.button("➕ Chat Mới", type="primary", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.current_session_id = new_id
            st.session_state.all_chats[new_id] = []
            st.rerun()
    with col2:
        with st.popover("⚙️ Cấu hình"):
            google_api_key = st.text_input("Gemini API Key:", type="password")
            st.markdown("---")
            mysql_host = st.text_input("MySQL Host:", value="")
            mysql_user = st.text_input("Username:", value="")
            mysql_pass = st.text_input("Password:", type="password")
            mysql_db = st.text_input("Database:", value="")


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
    # Duyệt ngược danh sách để các đoạn chat mới nhất lên đầu (như Gemini)
    for session_id, chat_messages in reversed(st.session_state.all_chats.items()):
        if len(chat_messages) > 0:
            has_history = True
            # Lấy câu hỏi đầu tiên làm Tiêu đề cho nút bấm
            title = "Tin nhắn mới..."
            for m in chat_messages:
                if m["role"] == "user":
                    title = m["content"][:22] + "..."
                    break
           
            # Đổi icon để báo hiệu phiên chat nào đang được mở
            is_active = (session_id == st.session_state.current_session_id)
            btn_label = f"👉 {title}" if is_active else f"💬 {title}"
           
            # Nếu người dùng bấm vào lịch sử -> Đổi Session ID hiện tại -> Render lại màn hình
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
# 4. WORKFLOW KIỂM ĐỊNH DỮ LIỆU (DATA AUDIT - CODE CỨNG)
# ==========================================
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
def get_db_uri():
    if mysql_host and mysql_user and mysql_db:
        pwd_part = f":{mysql_pass}" if mysql_pass else ""
        return f"mysql+pymysql://{mysql_user}{pwd_part}@{mysql_host}:3306/{mysql_db}"
    return "sqlite:///ecommerce.db"


def get_agent():
    if not google_api_key:
        return None, "Vui lòng nhập API Key trong mục Cấu hình."
    try:
        db_uri = get_db_uri()
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
# 6. GIAO DIỆN HIỂN THỊ
# ==========================================
def render_assistant_response(answer, audit_logs=None):
    answer = answer.replace("`", "") if answer.startswith("`") else answer
   
    code_blocks = re.findall(r'```python(.*?)```', answer, re.DOTALL)
    sql_blocks = re.findall(r'```sql(.*?)```', answer, re.DOTALL)
   
    phan_tich = "Hệ thống đã phân tích xong nhưng đầu ra bị sai định dạng hiển thị. Vui lòng thử lại."
    chien_luoc = "Hệ thống chưa kịp hoàn thiện chiến lược. Vui lòng bấm 'Chat Mới' và hỏi lại."
   
    if "[PHÂN TÍCH]" in answer:
        phan_tich = answer.split("[PHÂN TÍCH]")[1].split("[")[0].strip()
       
    if "[CHIẾN LƯỢC]" in answer:
        chien_luoc = answer.split("[CHIẾN LƯỢC]")[1].split("[")[0].strip()


    if code_blocks:
        combined_code = "\n".join(code_blocks)
        try:
            exec(combined_code)
        except Exception as e:
            st.warning(f"Không thể hiển thị biểu đồ: {e}")


    st.markdown("---")
   
    if audit_logs:
        with st.expander("🔍 Biên bản Kiểm định Dữ liệu (Auto-Audit Workflow)", expanded=False):
            for log in audit_logs:
                if log["status"] == "error":
                    st.error(log["msg"])
                elif log["status"] == "warning":
                    st.warning(log["msg"])
                else:
                    st.success(log["msg"])
    else:
        st.success("✔️ **Dữ liệu đã được trích xuất an toàn từ CSDL Doanh Nghiệp**")


    tab1, tab2, tab3 = st.tabs(["📊 Báo cáo Phân tích (Insight)", "💡 Đề xuất Chiến lược", "⚙️ Tiến trình SQL"])
   
    with tab1:
        st.markdown(phan_tich if "[PHÂN TÍCH]" in answer else answer)
           
    with tab2:
        st.markdown(chien_luoc)
       
    with tab3:
        if sql_blocks:
            st.markdown("**Câu lệnh SQL đã được Agent thực thi:**")
            for sql in sql_blocks:
                st.code(sql, language="sql")
        else:
            st.info("Agent đã sử dụng dữ liệu ngữ cảnh hoặc tiến trình bị ngắt.")


# Hiển thị tin nhắn CỦA PHIÊN CHAT HIỆN TẠI
current_messages = st.session_state.all_chats.get(st.session_state.current_session_id, [])


for msg in current_messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            render_assistant_response(msg["content"])


if prompt := st.chat_input("VD: Phân tích top 10 sản phẩm..."):
    # Thêm câu hỏi vào phiên hiện tại & LƯU LẠI
    st.session_state.all_chats[st.session_state.current_session_id].append({"role": "user", "content": prompt})
    save_history(st.session_state.all_chats)
   
    with st.chat_message("user"):
        st.markdown(prompt)


    agent, status = get_agent()
   
    with st.chat_message("assistant"):
        if agent is None:
            st.error(status)
        else:
            with st.spinner("Đang chạy luồng kiểm định chất lượng dữ liệu..."):
                current_audit = run_data_audit(get_db_uri())
               
            with st.spinner("Agent đang xử lý phân tích và tổng hợp Insight..."):
                try:
                    response = agent.invoke({"input": prompt})
                    answer = response["output"]
                   
                    render_assistant_response(answer, current_audit)
                   
                    # Thêm câu trả lời vào phiên & LƯU LẠI
                    st.session_state.all_chats[st.session_state.current_session_id].append({"role": "assistant", "content": answer})
                    save_history(st.session_state.all_chats)
                   
                except Exception as e:
                    error_str = str(e)
                    if "[PHÂN TÍCH]" in error_str or "[BIỂU ĐỒ]" in error_str:
                        extracted_answer = error_str.split("Could not parse LLM output:")[-1].strip()
                        render_assistant_response(extracted_answer, current_audit)
                       
                        st.session_state.all_chats[st.session_state.current_session_id].append({"role": "assistant", "content": extracted_answer})
                        save_history(st.session_state.all_chats)
                    else:
                        st.error(f"Đã có lỗi hệ thống xảy ra: {e}")

