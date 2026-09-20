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
import plotly.express as px

# ==========================================
# 0. KHỞI TẠO ĐƯỜNG DẪN & DATABASE SQLITE
# ==========================================
DB_PATH = os.path.abspath("ecommerce.db").replace('\\', '/')
DB_URI_SQLITE = f"sqlite:///{DB_PATH}"

@st.cache_resource(show_spinner="Đang nạp dữ liệu... Vui lòng đợi!")
def init_sqlite_db():
    conn = sqlite3.connect("ecommerce.db", check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='df_orders'")
    if cursor.fetchone()[0] == 0:
        try:
            read_opts = {'sep': None, 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'utf-8'}
            pd.read_csv("df_Customers.csv", **read_opts).to_sql('df_customers', conn, index=False)
            pd.read_csv("df_Orders.csv", **read_opts).to_sql('df_orders', conn, index=False)
            pd.read_csv("df_Payments.csv", **read_opts).to_sql('df_payments', conn, index=False)
            pd.read_csv("df_Products.csv", **read_opts).to_sql('df_products', conn, index=False)
            pd.read_csv("df_OrderItems.csv", **read_opts).to_sql('df_orderitems', conn, index=False)
        except Exception as e:
            st.error(f"Lỗi đọc file CSV: {e}")
    conn.close()
    return True

init_sqlite_db()

# ==========================================
# 1. QUẢN LÝ LỊCH SỬ HỘI THOẠI
# ==========================================
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return {str(uuid.uuid4()): data}
                return data
        except: pass
    return {}

def save_history(all_chats):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chats, f, ensure_ascii=False, indent=4)

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
# 2. GIAO DIỆN & SIDEBAR
# ==========================================
st.set_page_config(page_title="My AI agent", page_icon="🛒", layout="wide")
st.title("🛒 My AI agent")
st.markdown("Trợ lý AI phân tích dữ liệu, săn Insight & Hoạch định Chiến lược")

with st.sidebar:
    st.markdown("### ⚙️ Cấu hình API")
    gemini_api_key = st.text_input("Gemini API Key:", type="password", key="api_key")
    st.markdown("---")
    if st.button("➕ Chat Mới", type="primary", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.all_chats[new_id] = []
        st.rerun()

    st.markdown("---")
    st.markdown("🕒 **Lịch sử Hội thoại**")
    for session_id, chat_messages in reversed(st.session_state.all_chats.items()):
        if len(chat_messages) > 0:
            title = chat_messages[0]["content"][:20] + "..." if chat_messages[0]["role"] == "user" else "Chat..."
            is_active = (session_id == st.session_state.current_session_id)
            if st.button(f"👉 {title}" if is_active else f"💬 {title}", key=f"hist_{session_id}", use_container_width=True):
                st.session_state.current_session_id = session_id
                st.rerun()

    if st.button("🗑️ Xóa toàn bộ lịch sử", use_container_width=True):
        if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
        st.session_state.all_chats = {}
        st.rerun()

# ==========================================
# 3. PROMPT & AGENT (TỐI ƯU API)
# ==========================================
instructions_raw = """
Bạn là Data Analyst. Dùng công cụ sql_db_query để truy vấn CSDL. KHÔNG tự bịa số liệu.
Khi có kết quả, BẮT BUỘC bắt đầu bằng "Final Answer: " và xuất theo định dạng sau:

[BIỂU ĐỒ]
[CHART:loại_biểu_đồ|cột_x|cột_y|tiêu_đề] (loại: bar, line, pie, scatter. TUYỆT ĐỐI KHÔNG CODE PYTHON)

[PHÂN TÍCH]
Viết 2-3 câu phân tích ngắn gọn.

[CHIẾN LƯỢC]
Viết 2 bullet points đề xuất hành động.

[SQL]
'''sql
-- Lệnh SQL ở đây
'''
"""
instructions = instructions_raw.replace("'''", "```")

def get_agent():
    api_key = st.session_state.get("api_key", "")
    if not api_key: return None, "Vui lòng nhập Gemini API Key ở thanh bên trái."
    try:
        db = SQLDatabase.from_uri(DB_URI_SQLITE)
        # Sử dụng model gemini-3.6-flash theo yêu cầu của sếp
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0)
        
        agent_executor = create_sql_agent(
            llm=llm, 
            db=db,
            agent_type="zero-shot-react-description", 
            prefix=instructions, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=4 # GIỚI HẠN VÒNG LẶP ĐỂ CHỐNG CHÁY API
        )
        return agent_executor, "OK"
    except Exception as e:
        return None, str(e)

# ==========================================
# 4. RENDER GIAO DIỆN
# ==========================================
def render_assistant_response(answer):
    # Trích xuất biểu đồ
    chart_spec = None
    chart_match = re.search(r'\[CHART:(.*?)\]', answer)
    if chart_match:
        parts = chart_match.group(1).split('|')
        if len(parts) >= 3:
            chart_spec = {"type": parts[0].strip(), "x": parts[1].strip(), "y": parts[2].strip(), "title": parts[3].strip() if len(parts)>3 else ""}

    # Trích xuất SQL
    sql_blocks = re.findall(r'```sql\s*(.*?)\s*```', answer, re.DOTALL | re.IGNORECASE)
    sql_to_run = sql_blocks[-1] if sql_blocks else None

    # Làm sạch text
    phan_tich = answer
    phan_tich = re.sub(r'```.*?```', '', phan_tich, flags=re.DOTALL)
    phan_tich = re.sub(r'\[CHART:.*?\]', '', phan_tich)
    phan_tich = phan_tich.replace("Final Answer:", "").strip()

    st.markdown("---")
    
    # Render dữ liệu
    df_preview = None
    if sql_to_run and "SELECT" in sql_to_run.upper():
        try:
            engine = create_engine(DB_URI_SQLITE)
            df_preview = pd.read_sql(sql_to_run, engine)
        except: pass

    tab1, tab2 = st.tabs(["📊 Báo cáo Insight", "⚙️ SQL"])
    with tab1:
        st.markdown(phan_tich)
        if df_preview is not None and not df_preview.empty and chart_spec:
            c_type = chart_spec["type"]
            try:
                if c_type == "bar": st.plotly_chart(px.bar(df_preview, x=chart_spec["x"], y=chart_spec["y"], title=chart_spec["title"]), use_container_width=True)
                elif c_type == "pie": st.plotly_chart(px.pie(df_preview, names=chart_spec["x"], values=chart_spec["y"], title=chart_spec["title"]), use_container_width=True)
                elif c_type == "line": st.plotly_chart(px.line(df_preview, x=chart_spec["x"], y=chart_spec["y"], title=chart_spec["title"]), use_container_width=True)
                elif c_type == "scatter": st.plotly_chart(px.scatter(df_preview, x=chart_spec["x"], y=chart_spec["y"], title=chart_spec["title"]), use_container_width=True)
            except: pass

    with tab2:
        if sql_to_run:
            st.code(sql_to_run, language="sql")
            if df_preview is not None: st.dataframe(df_preview, use_container_width=True)

# Hiển thị lịch sử
current_messages = st.session_state.all_chats.get(st.session_state.current_session_id, [])
for msg in current_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user": st.markdown(msg["content"])
        else: render_assistant_response(msg["content"])

# Xử lý Input
if prompt := st.chat_input("Hỏi tôi về dữ liệu..."):
    st.session_state.all_chats[st.session_state.current_session_id].append({"role": "user", "content": prompt})
    save_history(st.session_state.all_chats)
    
    with st.chat_message("user"): st.markdown(prompt)

    agent, status = get_agent()
    with st.chat_message("assistant"):
        if not agent:
            st.error(status)
        else:
            with st.spinner("Đang phân tích (Tiết kiệm API mode)..."):
                try:
                    response = agent.invoke({"input": prompt})
                    answer = response["output"]
                    render_assistant_response(answer)
                    st.session_state.all_chats[st.session_state.current_session_id].append({"role": "assistant", "content": answer})
                    save_history(st.session_state.all_chats)
                except Exception as e:
                    error_msg = str(e)
                    # Nếu lỗi parse (Gemini hay gặp lỗi này nhưng vẫn ra kết quả)
                    if "Could not parse LLM output" in error_msg:
                        extracted = error_msg.split("Could not parse LLM output:")[-1].strip()
                        render_assistant_response(extracted)
                        st.session_state.all_chats[st.session_state.current_session_id].append({"role": "assistant", "content": extracted})
                        save_history(st.session_state.all_chats)
                    else:
                        st.error(f"Lỗi: {e}")