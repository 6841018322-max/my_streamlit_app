import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time # For exponential backoff
from typing import Optional, Tuple, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Configuration ---
MODEL_NAME = "gemini-2.5-flash"
MAX_ARTICLE_LENGTH = 15000
# ---------------------

# 1. Utility Functions (ไม่เปลี่ยนแปลงจากโค้ดที่คุณให้มา)
def get_article_text(url) -> Tuple[Optional[str], Optional[str]]:
    """ดึงข้อความหลักจาก URL ข่าว."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
        article_text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        if len(article_text) < 100:
             article_text = soup.body.get_text(separator='\n', strip=True)
             
        if not article_text:
            return None, "ไม่พบเนื้อหาที่ชัดเจนในหน้านี้"

        if len(article_text) > MAX_ARTICLE_LENGTH:
            article_text = article_text[:MAX_ARTICLE_LENGTH] + "..."
            st.warning(f"ข้อความข่าวที่ดึงมา (ก่อนการกรอง) ถูกตัดให้เหลือเพียง {MAX_ARTICLE_LENGTH} ตัวอักษรเพื่อความรวดเร็วในการประมวลผล")

        return article_text, None

    except requests.exceptions.RequestException as e:
        return None, f"ไม่สามารถดึงข้อมูลจาก URL ได้: {e}"
    except Exception as e:
        return None, f"เกิดข้อผิดพลาดในการประมวลผล: {e}"


def extract_main_content_with_gemini(client: genai.Client, noisy_text: str) -> Tuple[Optional[str], Optional[str]]:
    """ใช้ Gemini เพื่อดึงเฉพาะเนื้อหาหลักของบทความจากข้อความที่อาจมีสิ่งรบกวน."""
    extraction_prompt = f"""
    คุณคือผู้ช่วยดึงเนื้อหาหลัก (Core Article Extractor)
    จงวิเคราะห์ข้อความต่อไปนี้ซึ่งถูกดึงมาจากหน้าเว็บข่าว
    ข้อความนี้อาจมีเนื้อเนื้อหาที่ไม่เกี่ยวข้อง เช่น เมนูนำทาง, โฆษณา, คำบรรยายรูปภาพ, หรือส่วนท้ายของเว็บไซต์
    หน้าที่ของคุณคือ:
    1.  คัดเลือก **เฉพาะเนื้อหาหลักของบทความข่าว** (บทนำ, ย่อหน้าเนื้อหา, บทสรุป)
    2.  ละเว้นส่วนที่ไม่ใช่เนื้อหาหลัก เช่น ส่วนหัว, ส่วนท้าย, เมนู, ลิงก์ที่เกี่ยวข้อง, และคำอธิบายภาพที่ไม่ใช่เนื้อหา
    3.  ตอบกลับด้วยเนื้อหาหลักที่ถูกคัดเลือกมาเท่านั้น

    --- ข้อความที่ถูกดึงมา ---
    {noisy_text}
    """
    
    extraction_system_instruction = "คุณคือ Core Article Extractor ที่แม่นยำและตอบกลับด้วยข้อความที่สะอาดเท่านั้น"
    extraction_config = types.GenerateContentConfig(
        system_instruction=extraction_system_instruction
    )

    response = make_gemini_call_with_retry(
        client,
        contents=[extraction_prompt],
        config=extraction_config
    )

    if response and response.text:
        return response.text.strip(), None
    else:
        return None, "ไม่สามารถดึงเนื้อหาหลักด้วย Gemini ได้"


def make_gemini_call_with_retry(client: genai.Client, contents: Any, config: Optional[types.GenerateContentConfig]=None, max_retries: int=3):
    """เรียกใช้ Gemini API พร้อมกลไก Exponential Backoff."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
            return response
        except APIError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt 
                st.warning(f"เกิดข้อผิดพลาดจาก API ({e}) ลองใหม่ใน {wait_time} วินาที...")
                time.sleep(wait_time)
            else:
                st.error(f"การเรียก API ล้มเหลวหลังจาก {max_retries} ครั้ง: {e}")
                return None
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิดในการเรียก API: {e}")
            return None
    return None

# 2. Custom CSS สำหรับสีพื้นหลัง (สีฟ้าอ่อน / Light Blue)
def set_custom_theme():
    """ตั้งค่า CSS เพื่อเปลี่ยนสีพื้นหลังและปรับแต่งส่วนต่าง ๆ"""
    
    # สีพื้นหลัง Alice Blue (#F0F8FF) พร้อมลายจุดสีฟ้าอ่อน
    custom_css = """
    <style>
    /* ตั้งค่าสีพื้นหลังของหน้าหลัก (body) */
    .stApp {
        background-color: #F0F8FF; /* Alice Blue */
        opacity: 1;
        background-image:  radial-gradient(#b6d5f7 1.05px, transparent 1.05px), radial-gradient(#b6d5f7 1.05px, #F0F8FF 1.05px);
        background-size: 42px 42px;
        background-position: 0 0, 21px 21px;
    }
    
    /* ปรับปรุง container หลักที่อยู่ตรงกลาง (main) */
    .main > div {
        background-color: rgba(255, 255, 255, 0.9); /* ทำให้เนื้อหาหลักมีความโปร่งใสเล็กน้อย */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* ปรับแต่ง Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ADD8E6; /* Light Blue Sidebar */
        color: #00008B; /* Dark Blue Text */
        border-radius: 0 10px 10px 0;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# 3. Streamlit App Layout and Logic

# --- Page Config ---
st.set_page_config(
    page_title="เรียนภาษาอังกฤษจากข่าว - Powered by Gemini",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ใช้ CSS เพื่อตกแต่ง
set_custom_theme()

# --- Header Section ---
st.title("📰✨ English Learner Hub: เรียนภาษาอังกฤษจากข่าว")
st.markdown("---")
st.markdown(
    "👉 **ใส่ URL ข่าวภาษาอังกฤษ** ที่คุณสนใจ เพื่อรับการสรุปภาษาไทยและตารางคำศัพท์สำหรับฝึกฝน! **ขับเคลื่อนโดย Google Gemini**"
)

# --- Sidebar for API Key ---
with st.sidebar:
    st.header("🔑 การตั้งค่า Gemini API")
    gemini_api_key = st.text_input(
        "ใส่ Gemini API Key:",
        type="password",
        key="gemini_api_key"
    )
    st.info("💡 API Key จำเป็นสำหรับการประมวลผลโดย Gemini")
    st.markdown("หากยังไม่มีคีย์ สามารถรับได้จาก [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)")
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit & Gemini API")


# --- Main Input Container ---
input_container = st.container(border=True)
with input_container:
    st.subheader("🔗 ขั้นตอนที่ 1: ป้อน URL ข่าว")
    
    col1, col2 = st.columns([4, 1]) 
    
    with col1:
        news_url = st.text_input(
            "URL ของบทความข่าวภาษาอังกฤษ:",
            key="news_url",
            placeholder="เช่น https://www.bbc.com/news/world-us-canada-67616140",
            label_visibility="collapsed"
        )
    
    with col2:
        process_button = st.button("🚀 ประมวลผล", type="primary", key="process_news_button", use_container_width=True)


# --- Processing Logic ---

if process_button:
    # 1. Input Validation
    if not news_url:
        st.error("❌ กรุณาใส่ URL ของบทความข่าว")
        st.stop()
        
    if not gemini_api_key:
        st.error("🔑 กรุณาใส่ **Gemini API Key** ในแถบด้านข้างก่อน!")
        st.stop()
    
    if not news_url.startswith(('http://', 'https://')):
        st.error("❌ URL ไม่ถูกต้อง กรุณาใส่ URL ที่ขึ้นต้นด้วย http:// หรือ https://")
        st.stop()

    # --- Start Processing ---
    st.markdown("---")
    st.subheader("⚙️ กำลังประมวลผล...")
    
    # 1.1 Extract Article Text (Noisy)
    status_extract = st.status("กำลังดึงข้อความจาก URL...", expanded=True)
    noisy_article_text, error = get_article_text(news_url)
    status_extract.update(label="ดึงข้อความจาก URL", state="complete" if not error else "error", expanded=False)

    if error:
        st.error(f"❌ ดึงข้อมูลล้มเหลว: {error}")
        st.stop()

    if not noisy_article_text or len(noisy_article_text) < 50:
        st.error("❌ ไม่สามารถดึงข้อความข่าวที่มีความหมายได้ กรุณาลอง URL อื่น.")
        st.stop()

    try:
        client = genai.Client(api_key=gemini_api_key)
    except Exception as e:
        st.error(f"❌ ไม่สามารถเริ่มต้น Gemini Client ได้: {e}. ตรวจสอบ API Key ของคุณ.")
        st.stop()
        
    # 1.2 Clean Article Text with Gemini
    status_clean = st.status("กำลังให้ Gemini กรองเฉพาะเนื้อหาข่าวหลัก...", expanded=True)
    clean_article_text, extraction_error = extract_main_content_with_gemini(client, noisy_article_text)
    status_clean.update(label="กรองเนื้อหาหลัก", state="complete" if not extraction_error else "error", expanded=False)

    if extraction_error:
        st.warning("⚠️ เนื่องจากเกิดข้อผิดพลาดในการกรองเนื้อหาหลัก ระบบจะใช้ข้อความที่ดึงมาทั้งหมดแทน (อาจมีสิ่งรบกวน)")
        clean_article_text = noisy_article_text 
    
    st.success("✅ การดึงและกรองข้อมูลเสร็จสมบูรณ์! แสดงผลลัพธ์ด้านล่าง")
    st.markdown("---")

    # --- Display Part 1: Cleaned Text ---
    with st.expander("📖 ข้อความข่าวภาษาอังกฤษฉบับหลักที่ถูกกรองแล้ว (คลิกเพื่อดู)", expanded=False):
        st.code(clean_article_text, language='text')
    
    st.markdown("---")

    # --- Step 2: Generate Thai Summary (Part 2 - Full Width) ---
    st.header("🇹🇭 สรุปข่าวเป็นภาษาไทย")
    with st.spinner("กำลังให้ Gemini สรุปข่าวเป็นภาษาไทย..."):
        
        summary_prompt = f"สรุปเนื้อหาข่าวภาษาอังกฤษต่อไปนี้ให้เป็นภาษาไทยที่กระชับและเข้าใจง่าย ในรูปแบบย่อหน้าเดียว:\n\n---\n\n{clean_article_text}"
        summary_system_instruction = "คุณคือผู้ช่วยสรุปข่าวที่เชี่ยวชาญภาษาไทย"

        summary_config = types.GenerateContentConfig(
            system_instruction=summary_system_instruction
        )

        summary_response = make_gemini_call_with_retry(
            client, 
            contents=[summary_prompt], 
            config=summary_config
        )
        
        if summary_response and summary_response.text:
            st.info(summary_response.text) 
        else:
            st.error("ไม่สามารถสร้างบทสรุปได้")
            
    st.markdown("---") 

    # --- Step 3: Generate Vocabulary Table (Part 3 - Full Width) ---
    st.header("📝 ตารางคำศัพท์ที่น่าสนใจ (10 คำ)")
    st.caption("คำศัพท์ **10 คำ** พร้อมคำแปลและตัวอย่างประโยคจากข่าว เพื่อการฝึกฝนภาษาอังกฤษ")
    with st.spinner("กำลังให้ Gemini สร้างตารางคำศัพท์ 10 คำ..."):
        
        # Define the JSON Schema for structured output (Schema ยังคงเดิม)
        vocab_schema = types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "English_Word": types.Schema(type=types.Type.STRING, description="ศัพท์ภาษาอังกฤษระดับมัธยมจากข่าว"),
                    "Thai_Translation": types.Schema(type=types.Type.STRING, description="คำแปลภาษาไทย"),
                    "Example_Sentence": types.Schema(type=types.Type.STRING, description="ประโยคเต็มที่ใช้คำนั้นจากข้อความข่าวเดิม")
                },
                required=["English_Word", "Thai_Translation", "Example_Sentence"]
            )
        )
        
        vocab_system_instruction = "คุณคือครูสอนภาษาอังกฤษที่เชี่ยวชาญการสร้างบทเรียนจากเนื้อหาจริง คุณต้องตอบกลับเป็น JSON ที่ตรงตาม Schema ที่กำหนดเท่านั้น"

        vocab_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=vocab_schema,
            system_instruction=vocab_system_instruction
        )
        
        # *** เปลี่ยน prompt เพื่อขอ 10 คำ ***
        vocab_prompt = f"จากข้อความข่าวต่อไปนี้ ให้คุณสร้างรายการคำศัพท์ **10 คำ** ที่เหมาะสำหรับนักเรียนระดับมัธยมปลาย พร้อมคำแปลภาษาไทย และตัวอย่างประโยคที่ใช้คำนั้น ซึ่งต้องมาจากข้อความข่าวเดิมเท่านั้น:\n\n---\n\n{clean_article_text}"

        vocab_response = make_gemini_call_with_retry(
            client, 
            contents=[vocab_prompt], 
            config=vocab_config
        )

        if vocab_response and vocab_response.text:
            try:
                vocab_data = json.loads(vocab_response.text)
                
                # ตรวจสอบและจำกัดจำนวนคำศัพท์ไม่ให้เกิน 10 หาก Gemini ให้มาเกิน
                if len(vocab_data) > 10:
                    vocab_data = vocab_data[:10]
                    st.warning(f"⚠️ ระบบแสดงผลเพียง 10 คำแรกจากที่ Gemini สร้างมาทั้งหมด {len(vocab_data)} คำ")
                    
                vocab_df = pd.DataFrame(vocab_data)
                vocab_df.columns = ["ศัพท์ภาษาอังกฤษ", "คำแปลภาษาไทย", "ตัวอย่างประโยค (จากข่าว)"]

                # Display the DataFrame
                st.dataframe(
                    vocab_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "ตัวอย่างประโยค (จากข่าว)": st.column_config.TextColumn(
                            "ตัวอย่างประโยค (จากข่าว)",
                            width="large"
                        )
                    }
                )

            except json.JSONDecodeError:
                st.error("❌ ข้อผิดพลาด: Gemini ตอบกลับเป็นรูปแบบ JSON ที่ไม่ถูกต้อง")
            except Exception as e:
                st.error(f"❌ ข้อผิดพลาดในการแสดงผลตาราง: {e}")
        else:
            st.error("❌ ไม่สามารถสร้างตารางคำศัพท์ได้")

    st.markdown("---")
    st.caption("การประมวลผลเสร็จสิ้นแล้ว! ลองใส่ URL ข่าวอื่น ๆ เพื่อฝึกฝนต่อ")