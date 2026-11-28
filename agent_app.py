import datetime
import io
import math
import re
import traceback
import numpy as np
import pandas as pd
import streamlit as st
from google import genai

st.set_page_config(page_title="Excel AI Agent", page_icon="📊", layout="wide")

# ==========================================
# 1. KHÓA MẬT KHẨU BẢO VỆ QUOTA API
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>🔒 EXCEL AI AGENT</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Nhập mật khẩu để truy cập hệ thống</p>", unsafe_allow_html=True)
            pwd = st.text_input("Mật khẩu", type="password", label_visibility="collapsed", placeholder="Nhập mật khẩu...")
            if pwd:
                if pwd == "lamineyamal":
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("🚫 Sai mật khẩu!")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# 2. KẾT NỐI GOOGLE AI STUDIO (XOAY VÒNG KEY)
# ==========================================
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

def get_api_keys():
    raw_keys = st.session_state.get("custom_api_key", "").strip()
    if not raw_keys and "GEMINI_API_KEY" in st.secrets:
        raw_keys = st.secrets["GEMINI_API_KEY"]
    return [k.strip() for k in raw_keys.split(",") if k.strip()]

def ask_gemini(prompt: str) -> str:
    keys = get_api_keys()
    if not keys:
        raise ValueError("Chưa cấu hình GEMINI_API_KEY trong Streamlit Secrets hoặc Sidebar.")

    last_err = None
    for key in keys:
        client = genai.Client(api_key=key)
        for model_name in FALLBACK_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return (response.text or "").strip()
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if any(tok in err_str for tok in ["429", "quota", "rate", "resource_exhausted"]):
                    continue
                raise e
    raise last_err

# ==========================================
# 3. ĐỌC, SOI ĐỀ BÀI & XUẤT FILE EXCEL
# ==========================================
@st.cache_data(show_spinner=False)
def load_spreadsheet_sheets(file_bytes: bytes, file_name: str) -> dict:
    if file_name.lower().endswith(".csv"):
        return {"Sheet1": pd.read_csv(io.BytesIO(file_bytes))}
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)

def inspect_excel(sheets: dict) -> str:
    lines = []
    for name, df in sheets.items():
        cols_info = ", ".join(f"{col} ({dtype})" for col, dtype in df.dtypes.items())
        null_counts = df.isna().sum()
        null_cols = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
        null_summary = str(null_cols) if null_cols else "Không có ô trống"

        if len(df) <= 40:
            content_preview = f"- Toàn bộ nội dung sheet:\n{df.to_string(index=False)}"
        else:
            head_str = df.head(8).to_string(index=False)
            tail_str = df.tail(5).to_string(index=False)
            blank_rows = df[df.isna().any(axis=1)].head(5)
            blank_str = blank_rows.to_string(index=False) if not blank_rows.empty else "Không có"
            content_preview = (
                f"- 8 dòng đầu:\n{head_str}\n"
                f"- 5 dòng cuối:\n{tail_str}\n"
                f"- 5 dòng mẫu đang chứa ô trống (NaN) cần kiểm tra:\n{blank_str}"
            )

        lines.append(
            f"### SHEET: '{name}' (Kích thước: {len(df)} dòng x {len(df.columns)} cột)\n"
            f"- Danh sách cột: {cols_info}\n"
            f"- Thống kê ô trống (NaN) theo cột: {null_summary}\n"
            f"{content_preview}\n"
        )
    return "\n".join(lines)

def to_excel_bytes(sheets_dict: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets_dict.items():
            safe_name = str(sheet_name)[:31] or "Sheet1"
            df.to_excel(writer, sheet_name=safe_name, index=False)
            worksheet = writer.sheets[safe_name]
            for idx, col in enumerate(df.columns, 1):
                max_len = max(
                    df[col].astype(str).map(len).max() if not df.empty else 0,
                    len(str(col))
                ) + 4
                col_letter = worksheet.cell(row=1, column=idx).column_letter
                worksheet.column_dimensions[col_letter].width = min(max_len, 45)
    return output.getvalue()

# ==========================================
# 4. SANDBOX THỰC THI CODE AN TOÀN & AUTO-FIX
# ==========================================
FORBIDDEN_PATTERNS = [
    r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
    r"\bimport\s+shutil\b", r"\bopen\s*\(", r"\beval\s*\(",
    r"\bexec\s*\(", r"\bsecrets\b", r"\benviron\b"
]

ALLOWED_MODULES = {
    "pandas": pd,
    "numpy": np,
    "math": math,
    "datetime": datetime,
    "re": re,
}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ALLOWED_MODULES:
        return ALLOWED_MODULES[name]
    raise ImportError(f"Module '{name}' không được phép sử dụng trong Sandbox.")

def extract_python_code(text: str) -> str:
    blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.S | re.I)
    if not blocks:
        blocks = re.findall(r"```\s*(.*?)```", text, flags=re.S)
    return blocks[0].strip() if blocks else ""

def extract_plan(text: str) -> str:
    m = re.search(r"##\s*PLAN\s*(.*?)(?=##\s*CODE|```|$)", text, flags=re.S | re.I)
    return m.group(1).strip() if m else "Tự động phân tích và xử lý dữ liệu."

def run_sandbox(code: str, raw_sheets: dict):
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            return False, None, "", f"Lệnh bị chặn vì lý do bảo mật: {pat}"

    safe_builtins = {
        "__import__": safe_import,
        "len": len, "range": range, "min": min, "max": max, "sum": sum,
        "round": round, "abs": abs, "int": int, "float": float, "str": str,
        "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
        "enumerate": enumerate, "zip": zip, "sorted": sorted, "reversed": reversed,
        "map": map, "filter": filter, "any": any, "all": all, "print": print,
        "isinstance": isinstance, "type": type,
        "Exception": Exception, "ValueError": ValueError, "KeyError": KeyError, "TypeError": TypeError
    }
    safe_globals = {
        "pd": pd,
        "np": np,
        "math": math,
        "datetime": datetime,
        "re": re,
        "__builtins__": safe_builtins,
    }
    local_vars = {
        "sheets": {k: v.copy() for k, v in raw_sheets.items()},
        "output_sheets": {},
        "analysis_note": ""
    }

    try:
        exec(code, safe_globals, local_vars)
        out = local_vars.get("output_sheets")
        note = str(local_vars.get("analysis_note", ""))

        if not out or not isinstance(out, dict):
            out = local_vars["sheets"]

        clean_out = {}
        for k, v in out.items():
            if isinstance(v, pd.DataFrame):
                clean_out[str(k)] = v
            elif isinstance(v, pd.Series):
                clean_out[str(k)] = v.reset_index()

        if not clean_out:
            return False, None, "", "Code chạy xong nhưng không tạo ra bảng DataFrame nào trong `output_sheets`."

        return True, clean_out, note, ""
    except Exception:
        return False, None, "", traceback.format_exc()

def run_excel_agent(task: str, raw_sheets: dict, business_rules: str):
    schema_info = inspect_excel(raw_sheets)

    coder_prompt = f"""
Bạn là Chuyên gia Tự động hóa Dữ liệu Excel bằng Python Pandas.

CẤU TRÚC & NỘI DUNG FILE EXCEL ĐẦU VÀO:
{schema_info}

QUY TẮC NGHIỆP VỤ BỔ SUNG:
{business_rules}

NHIỆM VỤ CỦA NGƯỜI DÙNG:
{task}

YÊU CẦU ĐẦU RA:
1. Viết mục `## PLAN` nêu ngắn gọn các bước xử lý (ghi rõ nếu phát hiện đề bài nằm sẵn trong file Excel hoặc quy luật điền ô trống).
2. Viết mục `## CODE` chứa 1 khối ```python ... ``` thực hiện nhiệm vụ.

LUẬT VIẾT CODE & XỬ LÝ Ô TRỐNG (AUTO-FILL) BẮT BUỘC:
- Biến `sheets` (dict[str, pd.DataFrame]) đã có sẵn, chứa toàn bộ các sheet của file Excel.
- Không dùng `pd.read_excel()` hay `to_excel()`, không import `os`, `sys`. Chỉ dùng `pd`, `np`, `math`, `datetime`, `re`.
- NẾU ĐỀ BÀI NẰM TRONG FILE EXCEL: Hãy đọc kỹ các text/ghi chú trong các sheet (như sheet Hướng dẫn, Yêu cầu, hoặc dòng ghi chú đầu/cuối bảng) để thực thi chính xác từng yêu cầu.
- NẾU PHẢI ĐIỀN Ô TRỐNG (BLANK / NaN):
  + Đối chiếu các dòng đã có sẵn dữ liệu với các dòng bị `NaN` để tìm ra quy luật (công thức tính toán giữa các cột, ánh xạ/VLOOKUP từ sheet danh mục sang sheet chính, phân loại điều kiện, hoặc nội suy dữ liệu thiếu).
  + Giữ nguyên các giá trị hợp lệ đã có sẵn (trừ khi đề bài yêu cầu tính lại toàn bộ), chỉ điền vào các ô `NaN` / trống.
- Lưu toàn bộ bảng kết quả vào dictionary `output_sheets` (ví dụ: `output_sheets = {{"Ket_qua": df_kq}}`).
- Gán tóm tắt những gì đã làm và các con số nổi bật vào biến `analysis_note` (kiểu string).
"""
    raw_reply = ask_gemini(coder_prompt)
    plan = extract_plan(raw_reply)
    code = extract_python_code(raw_reply)

    if not code:
        return {"error": True, "message": "AI không sinh được mã Python hợp lệ. Hãy thử lại."}

    # VÒNG LẶP THỰC THI & TỰ SỬA LỖI (TỐI ĐA 3 LẦN)
    trace = []
    success, out_sheets, note, err_msg = run_sandbox(code, raw_sheets)
    trace.append({"attempt": 1, "status": "OK" if success else "ERROR", "code": code, "error": err_msg})

    attempt = 1
    while not success and attempt < 3:
        attempt += 1
        fix_prompt = f"""
Đoạn code Pandas sau bị lỗi khi chạy trên file Excel:

CẤU TRÚC EXCEL:
{schema_info}

NHIỆM VỤ:
{task}

CODE BỊ LỖI:
```python
{code}
        THÔNG BÁO LỖI PYTHON:
{err_msg}

Hãy sửa lại code cho chạy thành công 100%. Chỉ trả về duy nhất 1 khối python ...  và đảm bảo gán kết quả vào output_sheets và analysis_note.
"""
fixed_reply = ask_gemini(fix_prompt)
new_code = extract_python_code(fixed_reply)
if not new_code:
break
code = new_code
success, out_sheets, note, err_msg = run_sandbox(code, raw_sheets)
trace.append({"attempt": attempt, "status": "OK" if success else "ERROR", "code": code, "error": err_msg})
if not success:
    return {
        "error": True,
        "message": f"Agent đã thử sửa {attempt} lần nhưng vẫn gặp lỗi thực thi:\n```\n{err_msg}\n```",
        "trace": trace
    }

# AGENT TỔNG HỢP BÁO CÁO
preview_out = "\n".join(
    f"- Sheet '{k}' ({len(v)} dòng): {v.head(5).to_string(index=False)}"
    for k, v in out_sheets.items()
)
reporter_prompt = f"""
Bạn là Trợ lý Phân tích Dữ liệu. Nhiệm vụ vừa chạy trên file dữ liệu: "{task}"
Ghi chú từ hệ thống (analysis_note): {note}
Bảng dữ liệu đầu ra (5 dòng đầu):
{preview_out}

Hãy viết báo cáo ngắn gọn, rõ ràng bằng tiếng Việt:

Nêu rõ những gì đã phát hiện trong file (nếu có đề bài sẵn hoặc quy luật điền ô trống) và đã xử lý xong.

Nêu các con số/insight nổi bật rút ra được.

Nhắc người dùng xem bảng kết quả và bấm nút tải file Excel bên dưới.
"""
summary = ask_gemini(reporter_prompt)

return {
"error": False,
"plan": plan,
"summary": summary,
"output_sheets": out_sheets,
"excel_bytes": to_excel_bytes(out_sheets),
"trace": trace
}

==========================================
5. GIAO DIỆN STREAMLIT
==========================================
if "excel_messages" not in st.session_state:
st.session_state.excel_messages = []

with st.sidebar:
st.markdown("### 📊 Excel AI Agent")
st.caption("Powered by Google AI Studio (Gemini)")
st.markdown("---")
uploaded_excel = st.file_uploader("📂 1. Tải lên file Excel / CSV", type=["xlsx", "xls", "csv"])

business_rules = st.text_area(
    "🧠 2. Quy tắc nghiệp vụ (Tùy chọn)",
    placeholder="VD: Cột Trang_thai = 'Hủy' thì không tính doanh thu...",
    height=110
)

with st.expander("🔑 Cấu hình API Key dự phòng"):
    st.text_input(
        "Google AI Studio API Key",
        type="password",
        key="custom_api_key",
        placeholder="AIzaSy... (Để trống nếu đã cài trong Secrets)"
    )

if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
    st.session_state.excel_messages = []
    st.rerun()
    st.title("📊 Trợ lý AI Xử lý & Phân tích Excel")

if not uploaded_excel:
st.info("👈 Hãy tải lên một file Excel (.xlsx, .xls) hoặc CSV ở thanh bên trái để bắt đầu giao việc cho AI.")
st.stop()

raw_sheets = load_spreadsheet_sheets(uploaded_excel.getvalue(), uploaded_excel.name)

with st.expander(f"👀 Xem nhanh dữ liệu gốc: {uploaded_excel.name} ({len(raw_sheets)} sheet)", expanded=False):
tabs = st.tabs(list(raw_sheets.keys()))
for tab, (s_name, s_df) in zip(tabs, raw_sheets.items()):
with tab:
st.caption(f"Kích thước: {len(s_df)} dòng × {len(s_df.columns)} cột | Ô trống (NaN): {int(s_df.isna().sum().sum())}")
st.dataframe(s_df.head(50), use_container_width=True, hide_index=True)

Hiển thị lịch sử hội thoại
for idx, msg in enumerate(st.session_state.excel_messages):
with st.chat_message(msg["role"]):
if msg["role"] == "user":
st.markdown(msg["content"])
else:
res = msg["result"]
if res.get("error"):
st.error(res["message"])
else:
st.markdown(res["summary"])
st.download_button(
label="📥 Tải file Excel kết quả (.xlsx)",
data=res["excel_bytes"],
file_name=f"AI_Processed_{uploaded_excel.name.rsplit('.', 1)[0]}.xlsx",
mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
key=f"dl_btn_{idx}",
type="primary"
)
out_sheets = res["output_sheets"]
out_tabs = st.tabs([f"📄 {k}" for k in out_sheets.keys()])
for t, (k, df_out) in zip(out_tabs, out_sheets.items()):
with t:
st.caption(f"{len(df_out)} dòng × {len(df_out.columns)} cột")
st.dataframe(df_out.head(100), use_container_width=True, hide_index=True)
if res.get("trace"):
            with st.expander("⚙️ Xem kế hoạch & Code Python Agent đã chạy"):
                if res.get("plan"):
                    st.markdown(f"**Kế hoạch:**\n{res['plan']}")
                for step in res["trace"]:
                    st.caption(f"Lần chạy #{step['attempt']} — Trạng thái: {step['status']}")
                    st.code(step["code"], language="python")
                    if step["error"]:
                        st.error(step["error"])
                        NÚT CHẠY TỰ ĐỘNG (KHÔNG CẦN GÕ LỆNH) + KHUNG NHẬP THỦ CÔNG
auto_run = st.button(
"⚡ Tự động đọc đề bài trong file & Điền ô trống (Không cần nhập lệnh)",
type="primary",
use_container_width=True
)
manual_input = st.chat_input("Hoặc nhập yêu cầu cụ thể tại đây...")

task_input = None
if auto_run:
task_input = (
"⚡ [CHẾ ĐỘ TỰ ĐỘNG]: Hãy quét toàn bộ các sheet trong file Excel này: "
"(1) Tìm xem có đề bài, hướng dẫn hoặc yêu cầu nào được ghi sẵn bên trong file Excel không để thực hiện; "
"(2) Xác định tất cả các cột/ô đang bị bỏ trống (Blank / NaN), tự suy luận quy luật tính toán hoặc ánh xạ từ các dòng mẫu/sheet tham chiếu để điền đầy đủ vào các ô trống đó."
)
elif manual_input:
task_input = manual_input

if task_input:
st.session_state.excel_messages.append({"role": "user", "content": task_input})
with st.chat_message("user"):
st.markdown(task_input)
with st.spinner("🤖 Agent đang quét đề bài, suy luận quy luật điền ô trống và kiểm thử code..."):
    try:
        result = run_excel_agent(task_input, raw_sheets, business_rules)
    except Exception as e:
        result = {"error": True, "message": f"Lỗi kết nối Google AI Studio: {e}"}

st.session_state.excel_messages.append({"role": "assistant", "result": result})
st.rerun()
