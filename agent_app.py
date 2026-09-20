import os
import re
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent

# ==========================================
# 0. KHỞI TẠO ĐƯỜNG DẪN & DATABASE SQLITE
# ==========================================
DB_PATH = os.path.abspath("ecommerce.db").replace('\\', '/')
DB_URI_SQLITE = f"sqlite:///{DB_PATH}"

@st.cache_resource(show_spinner="Đang nạp Data & Dọn dẹp rác Duplicate... Vui lòng đợi!")
def init_sqlite_db():
    conn = sqlite3.connect("ecommerce.db", check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='df_orders'")
    if cursor.fetchone()[0] == 0:
        try:
            read_opts = {'sep': None, 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'utf-8'}
            pd.read_csv("df_Customers.csv", **read_opts).drop_duplicates().to_sql('df_customers', conn, index=False, if_exists='replace')
            pd.read_csv("df_Orders.csv", **read_opts).drop_duplicates().to_sql('df_orders', conn, index=False, if_exists='replace')
            pd.read_csv("df_Payments.csv", **read_opts).drop_duplicates().to_sql('df_payments', conn, index=False, if_exists='replace')
            pd.read_csv("df_Products.csv", **read_opts).drop_duplicates().to_sql('df_products', conn, index=False, if_exists='replace')
            pd.read_csv("df_OrderItems.csv", **read_opts).drop_duplicates().to_sql('df_orderitems', conn, index=False, if_exists='replace')
        except Exception as e:
            st.error(f"Lỗi đọc file CSV: {e} - Hãy chắc chắn 5 file CSV đang nằm chung thư mục với app.py")
    conn.close()
    return True

init_sqlite_db()

# ==========================================
# 1. QUẢN LÝ LỊCH SỬ HỘI THOẠI (MULTI-SESSION)
# ==========================================
HISTORY_FILE = "chat_history_v1.json"

def load_chats():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {}
    return {}

def save_chats():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chats, f, ensure_ascii=False, indent=4)

def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"title": "New chat", "messages": []}
    st.session_state.current_chat_id = chat_id
    save_chats()

def delete_chat(chat_id):
    if chat_id in st.session_state.chats: del st.session_state.chats[chat_id]
    if not st.session_state.chats: create_new_chat()
    elif st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = next(reversed(st.session_state.chats))
    save_chats()

def clear_all_history():
    if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
    st.session_state.chats = {}
    create_new_chat()

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if not st.session_state.chats: create_new_chat()
elif "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[-1]

current_chat = st.session_state.chats[st.session_state.current_chat_id]

# ==========================================
# 2. CẤU HÌNH MAIN PAGE
# ==========================================
st.set_page_config(page_title="My AI agent", page_icon="🛒", layout="wide")
st.title("🛒 My AI agent")
st.markdown("Trợ lý AI phân tích dữ liệu, săn Insight & Hoạch định Chiến lược")
st.markdown("🔥 **Agent phát triển bởi: Group 3 - TINE313** 🔥")

# ==========================================
# 3. KHU VỰC SIDEBAR & CẤU HÌNH
# ==========================================
with st.sidebar:
    st.markdown("### 🔥 Group 3 - TINE313")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Chat Mới", type="primary", use_container_width=True):
            create_new_chat()
            st.rerun()
    with col2:
        with st.popover("⚙️ Cấu hình", use_container_width=True):
            st.text_input("Gemini API Key:", type="password", key="api_key")
            st.markdown("[👉 Lấy API Key tại đây](https://aistudio.google.com/app/apikey)")
            st.markdown("---")
            st.markdown("**Kết nối MySQL Workbench**")
            st.caption("Để trống nếu dùng CSDL SQLite mặc định.")
            st.text_input("Host (VD: localhost):", key="db_host")
            st.text_input("Username (VD: root):", key="db_user")
            st.text_input("Password:", type="password", key="db_pass")
            st.text_input("Database Name:", key="db_name")

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
    for chat_id, chat_data in reversed(list(st.session_state.chats.items())):
        if len(chat_data.get("messages", [])) > 0:
            has_history = True
            cols = st.columns([0.82, 0.18])
            is_current = (chat_id == st.session_state.current_chat_id)
            title = chat_data.get("title", "New chat")
            label = f"👉 {title}" if is_current else f"💬 {title}"
            
            with cols[0]:
                if st.button(label, key=f"open_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            with cols[1]:
                if st.button("🗑️", key=f"delete_{chat_id}", help="Xóa chat này", use_container_width=True):
                    delete_chat(chat_id)
                    st.rerun()
                    
    if not has_history: st.info("Chưa có lịch sử trò chuyện.")
    if st.button("🗑️ Dọn dẹp TOÀN BỘ lịch sử", use_container_width=True):
        clear_all_history()
        st.rerun()

# ==========================================
# 4. BỘ NÃO CHIẾN LƯỢC & PROMPT
# ==========================================
instructions_raw = """
Bạn là Giám đốc Chiến lược Dữ liệu (Chief Data Officer). 
1. BẮT BUỘC dùng tool để chạy SQL lấy kết quả thực tế từ CSDL. KHÔNG ĐƯỢC TỰ BỊA SỐ LIỆU.
2. KHỬ TRÙNG LẶP: Dữ liệu thực tế thường bị nhân bản khi JOIN bảng. BẠN BẮT BUỘC phải dùng COUNT(DISTINCT cột_id) thay vì COUNT() thông thường (Ví dụ: COUNT(DISTINCT order_id)).
3. Trả lời cuối cùng BẮT BUỘC phải theo đúng cấu trúc sau:

[BIỂU ĐỒ]
[CHART:loại_biểu_đồ|tên_cột_x|tên_cột_y|tiêu_đề]

[PHÂN TÍCH]
TUYỆT ĐỐI KHÔNG liệt kê lại các dòng dữ liệu thô. Chỉ tập trung viết đúng 2 ý sau:
- 1. Insight cơ bản: Đúc kết ngắn gọn xu hướng hoặc nguyên nhân cốt lõi từ số liệu.
- 2. Insight nghịch lý/chuyên sâu: BẮT BUỘC chỉ ra một điểm bất thường, trái logic thông thường, hoặc một góc khuất ẩn sâu đằng sau số liệu (Ví dụ: Doanh thu cao nhưng tỷ lệ hủy đơn lại cao nhất; Nhóm sản phẩm bán chạy nhất nhưng biên độ lợi nhuận lại thấp...). 

[CHIẾN LƯỢC]
- Đề xuất 2-3 chiến thuật cụ thể để xử lý hoặc tận dụng chính cái "Insight nghịch lý" vừa tìm thấy.

[SQL]
'''sql
-- Dán câu lệnh SQL đã chạy thành công vào đây
'''
"""
instructions = instructions_raw.replace("'''", "```")

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
        df_pay = pd.read_sql("SELECT payment_value FROM df_payments WHERE payment_value < 0 LIMIT 1", engine)
        if not df_pay.empty: audit_logs.append({"status": "error", "msg": "❌ df_payments: Phát hiện giao dịch giá trị âm."})
        else: audit_logs.append({"status": "success", "msg": "✅ df_payments: 100% giao dịch có giá trị dương hợp lệ."})
            
        df_ord = pd.read_sql("SELECT order_id FROM df_orders WHERE order_status IS NULL LIMIT 1", engine)
        if not df_ord.empty: audit_logs.append({"status": "warning", "msg": "⚠️ df_orders: Phát hiện đơn hàng bị trống trạng thái."})
        else: audit_logs.append({"status": "success", "msg": "✅ df_orders: Toàn vẹn dữ liệu trạng thái đơn hàng."})
    except Exception:
        audit_logs.append({"status": "warning", "msg": "⚠️ Bỏ qua kiểm định sâu do CSDL chưa khởi tạo đầy đủ."})
    return audit_logs

def get_agent():
    api_key = st.session_state.get("api_key", "")
    if not api_key: return None, "Vui lòng nhập Gemini API Key trong mục ⚙️ Cấu hình."
    try:
        # CẤP NHÌN DATA MẪU: Đổi thành 2 để AI thấy tên cột chuẩn xác, giảm lỗi SQL
        db = SQLDatabase.from_uri(get_db_uri(), sample_rows_in_table_info=2)
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0.1)
        # NỚI LỎNG VÒNG LẶP: Tăng lên 8 để AI có cơ hội sửa lỗi SQL
        agent_executor = create_sql_agent(llm=llm, db=db, agent_type="zero-shot-react-description", prefix=instructions, verbose=True, handle_parsing_errors=True, max_iterations=8)
        return agent_executor, "OK"
    except Exception as e: return None, str(e)

# ==========================================
# 5. RENDER RESPONSE
# ==========================================
def render_assistant_response(answer, audit_logs=None):
    answer = answer.strip()
    
    # BÓC TÁCH SQL
    sql_blocks = re.findall(r'```(?:sql)?\s*(.*?)\s*```', answer, re.DOTALL | re.IGNORECASE)
    sql_to_run = sql_blocks[-1] if sql_blocks else None

    # BÓC TÁCH CHART
    chart_spec = None
    chart_match = re.search(r'\[CHART:(.*?)\]', answer, re.IGNORECASE)
    if chart_match:
        parts = chart_match.group(1).split('|')
        if len(parts) >= 3:
            chart_spec = {
                "type": parts[0].strip().lower(),
                "x": parts[1].strip().replace('\n', '').replace('\r', ''),
                "y": parts[2].strip().replace('\n', '').replace('\r', ''),
                "title": parts[3].strip().replace('\n', '') if len(parts) > 3 else "Biểu đồ Phân tích"
            }

    # BÓC TÁCH INSIGHT & CHIẾN LƯỢC
    phan_tich = answer
    chien_luoc = "Hệ thống chưa kịp hoàn thiện chiến lược."

    pt_match = re.search(r'\[PHÂN TÍCH\](.*?)(\[CHIẾN LƯỢC\]|\[SQL\]|```|$)', answer, re.DOTALL | re.IGNORECASE)
    if pt_match: phan_tich = pt_match.group(1).strip()

    cl_match = re.search(r'\[CHIẾN LƯỢC\](.*?)(\[SQL\]|```|$)', answer, re.DOTALL | re.IGNORECASE)
    if cl_match: chien_luoc = cl_match.group(1).strip()

    if not pt_match:
        phan_tich = re.sub(r'```.*?```', '', answer, flags=re.DOTALL)
        phan_tich = re.sub(r'\[CHART:.*?\]', '', phan_tich, flags=re.IGNORECASE).replace("Final Answer:", "").strip()

    st.markdown("---")
    st.markdown("💡 **Hệ thống AI đã bóc tách thành công Insight từ CSDL.**")
    
    if audit_logs:
        with st.expander("🔍 Biên bản Kiểm định Dữ liệu (Auto-Audit Workflow)", expanded=False):
            for log in audit_logs:
                if log["status"] == "error": st.error(log["msg"])
                elif log["status"] == "warning": st.warning(log["msg"])
                else: st.success(log["msg"])

    # THỰC THI SQL
    df_preview = None
    if sql_to_run and "SELECT" in sql_to_run.upper():
        try:
            engine = create_engine(get_db_uri())
            df_preview = pd.read_sql(sql_to_run, engine)
        except Exception as e:
            st.error(f"⚠️ Lỗi CSDL: {e}")

    tab1, tab2, tab3 = st.tabs(["📊 Báo cáo Phân tích (Insight)", "💡 Đề xuất Chiến lược", "⚙️ Tiến trình SQL"])
    
    with tab1:
        if phan_tich: st.markdown(phan_tich)
        else: st.info("Hệ thống chưa tìm thấy Insight đủ sâu cho câu hỏi này.")
        
        if df_preview is not None and not df_preview.empty and chart_spec:
            st.markdown("---")
            c_type = chart_spec.get("type", "none")
            if c_type != "none":
                x_col = chart_spec["x"]
                y_col = chart_spec["y"]
                title = chart_spec["title"]
                try:
                    if c_type == "bar": st.plotly_chart(px.bar(df_preview, x=x_col, y=y_col, title=title), use_container_width=True)
                    elif c_type == "pie": st.plotly_chart(px.pie(df_preview, names=x_col, values=y_col, title=title), use_container_width=True)
                    elif c_type == "line": st.plotly_chart(px.line(df_preview, x=x_col, y=y_col, title=title), use_container_width=True)
                    elif c_type == "scatter": st.plotly_chart(px.scatter(df_preview, x=x_col, y=y_col, title=title), use_container_width=True)
                except Exception:
                    st.warning("⚠️ Biểu đồ không thể hiển thị do AI chọn sai tên cột. Vui lòng xem bảng dữ liệu thô ở tab 'Tiến trình SQL'.")
            
    with tab2: st.markdown(chien_luoc)
        
    with tab3:
        if sql_to_run:
            st.markdown("**Câu lệnh SQL đã thực thi:**")
            st.code(sql_to_run, language="sql")
            if df_preview is not None:
                st.markdown("**🗄️ Bảng kết quả (Data Preview):**")
                st.dataframe(df_preview, use_container_width=True)
        else: st.info("Không có tiến trình SQL nào được ghi nhận.")

# Hiển thị Chat
for msg in current_chat["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"): st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"): render_assistant_response(msg["content"])

# Xử lý Chat mới
if prompt := st.chat_input("VD: Cho tôi insights về doanh thu theo danh mục..."):
    current_chat["messages"].append({"role": "user", "content": prompt})
    save_chats()
    
    with st.chat_message("user"): st.markdown(prompt)

    agent, status = get_agent()
    with st.chat_message("assistant"):
        if agent is None: st.error(status)
        else:
            with st.spinner("Đang chạy luồng kiểm định chất lượng dữ liệu..."):
                current_audit = run_data_audit(get_db_uri())
                
            with st.spinner("Agent đang phân tích sâu dữ liệu với Gemini 3.6..."):
                try:
                    response = agent.invoke({"input": prompt})
                    answer = response["output"]
                    render_assistant_response(answer, current_audit)
                    current_chat["messages"].append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_str = str(e)
                    # Bắt lỗi Parsing
                    if "Could not parse LLM output:" in error_str:
                        extracted_answer = error_str.split("Could not parse LLM output:")[-1].strip()
                        render_assistant_response(extracted_answer, current_audit)
                        current_chat["messages"].append({"role": "assistant", "content": extracted_answer})
                    # Bắt lỗi Iteration Limit
                    elif "Agent stopped due to iteration limit or time limit" in error_str:
                        err_msg = "⚠️ AI đã thử truy xuất dữ liệu nhiều lần nhưng liên tục gặp lỗi SQL nên phải tự động dừng để bảo vệ API. Sếp thử đặt câu hỏi với tên bảng/cột cụ thể hơn nhé!"
                        st.error(err_msg)
                        current_chat["messages"].append({"role": "assistant", "content": err_msg})
                    else:
                        st.error(f"Đã có lỗi hệ thống xảy ra: {e}")
                        
        if current_chat.get("title") == "New chat":
            clean_title = " ".join(prompt.strip().split())
            if len(clean_title) > 34: clean_title = clean_title[:34] + "..."
            current_chat["title"] = clean_title
            
        save_chats()
        st.rerun()