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

@st.cache_resource(show_spinner="Đang nạp 89.000+ đơn hàng từ CSV vào hệ thống... Vui lòng đợi!")
def init_sqlite_db():
    conn = sqlite3.connect("ecommerce.db", check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='df_orders'")
    if cursor.fetchone()[0] == 0:
        try:
            read_opts = {'sep': None, 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'utf-8'}
            pd.read_csv("df_Customers.csv", **read_opts).to_sql('df_customers', conn, index=False, if_exists='replace')
            pd.read_csv("df_Orders.csv", **read_opts).to_sql('df_orders', conn, index=False, if_exists='replace')
            pd.read_csv("df_Payments.csv", **read_opts).to_sql('df_payments', conn, index=False, if_exists='replace')
            pd.read_csv("df_Products.csv", **read_opts).to_sql('df_products', conn, index=False, if_exists='replace')
            pd.read_csv("df_OrderItems.csv", **read_opts).to_sql('df_orderitems', conn, index=False, if_exists='replace')
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
instructions = """
Bạn là Kỹ sư Dữ liệu cấp cao. 
1. BẮT BUỘC dùng tool để chạy SQL lấy kết quả thực tế. KHÔNG ĐƯỢC TỰ BỊA SỐ LIỆU.
2. Trả lời cuối cùng BẮT BUỘC phải theo đúng cấu trúc sau:

[BIỂU ĐỒ]
[CHART:loại_biểu_đồ|tên_cột_x|tên_cột_y|tiêu_đề]

[PHÂN TÍCH]
- Trình bày chi tiết dữ liệu vừa truy xuất được.
- Đưa ra ít nhất 2 insight kinh doanh cụ thể từ con số trên. (Viết chi tiết, tuyệt đối không được bỏ trống phần này)

[CHIẾN LƯỢC]
- Đề xuất 2-3 hành động cụ thể để cải thiện.

[SQL]
```sql
-- Dán câu SQL đã chạy thành công vào đây