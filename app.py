#import streamlit as st
#import pandas as pd
#import numpy as np

# Use caching to efficiently load data only once
#@st.cache_data
#def load_data():
   # Load 10,000 rows of data
   #data = pd.DataFrame(
       #np.random.randn(10000, 2) / [50, 50] + [37.76, -122.4],
       #columns=['lat', 'lon']
   
   #return data

#st.title('Simple Data Explorer')

# Load the data
#df = load_data()

# 1. Add a Widget to control the data
#st.subheader('Filter Data')
#num_points = st.slider('Number of points to display', 100, 10000, 1000)

# 2. Filter the data based on the widget value
#filtered_df = df.head(num_points)

# 3. Display the results
#st.subheader(f'Displaying the first {num_points} data points')
#st.dataframe(filtered_df)

# 4. Visualize the results on a map
#st.map(filtered_df)



#ในการใช้ API Key และฟังก์ชัน $response = model.generate\_content(prompt)$ กับโค้ด Streamlit ที่ได้ให้ไว้ก่อนหน้านี้ จะเป็นการ **แทนที่ส่วนของการวิเคราะห์เนื้อหาจำลอง (Mock Data)** ด้วยการเรียกใช้งานจริงครับ

#นี่คือโครงสร้างโค้ดที่รวมการใช้งาน **Gemini API** เข้ากับ Streamlit และ Web Scraping โดยใช้ $response = client.models.generate\_content()$ :

## 💻 โค้ด Python ฉบับสมบูรณ์ที่ใช้ API Call

#โค้ดนี้จะใช้ `google-genai` library (สมมติว่าคุณติดตั้งและมี API Key แล้ว)

#```python
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import os # ใช้สำหรับเรียก API Key จาก Environment Variable หรือ st.secrets
from typing import List, Dict, Any

st.set_page_config(page_title="News Vocab Extractor with LLM", layout="wide")
# ********** ส่วนการตั้งค่า API และ Model **********

# **1. การจัดการ API Key อย่างปลอดภัย**
# แนะนำให้ใช้ st.secrets หรือ Environment Variables (เช่น os.environ.get("GEMINI_API_KEY"))
# สำหรับตัวอย่างนี้ จะใช้ st.secrets ซึ่งเป็นวิธีมาตรฐานของ Streamlit
try:
    from google import genai
    # ใช้คีย์จาก st.secrets หากรันบน Streamlit Cloud หรือตั้งค่า local secrets
    # หากรัน local ต้องตั้งค่า GEMINI_API_KEY ในไฟล์ .streamlit/secrets.toml
    API_KEY = st.secrets.get("AIzaSyC8XBq4qiuar9rWFJt5JZttX0Fxy7ffpg0") 
    if not API_KEY:
        st.error("API Key ไม่ถูกตั้งค่าใน st.secrets กรุณาตรวจสอบการตั้งค่า.")
        # หากไม่มี ให้ดึงจาก Environment Variable ทั่วไป
        API_KEY = os.environ.get("AIzaSyC8XBq4qiuar9rWFJt5JZttX0Fxy7ffpg0")

    if API_KEY:
        client = genai.Client(api_key=API_KEY)
        MODEL_NAME = "gemini-2.5-flash"
    else:
        client = None
        st.warning("ไม่สามารถเชื่อมต่อ Gemini API ได้ (API Key หายไป)")
except ImportError:
    st.error("กรุณาติดตั้งไลบรารี google-genai: pip install google-genai")
    client = None
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการตั้งค่า API Client: {e}")
    client = None
    
# ********** ส่วนการกำหนด Prompt และ Schema **********

def get_response_schema():
    """กำหนด JSON Schema เพื่อให้ Output เป็นตารางที่ถูกต้อง"""
    return {
        "type": "object",
        "properties": {
            "vocabulary_list": {
                "type": "array",
                "description": "รายการคำศัพท์ที่วิเคราะห์ได้",
                "items": {
                    "type": "object",
                    "properties": {
                        "คำศัพท์ (English)": {"type": "string", "description": "คำศัพท์ภาษาอังกฤษที่ถูกระบุ"},
                        "คำแปล (Thai)": {"type": "string", "description": "คำแปลหลักเป็นภาษาไทย"},
                        "ประโยคที่ปรากฏในข่าว": {"type": "string", "description": "ประโยคเต็มจากเนื้อหาที่ปรากฏคำศัพท์นั้น"},
                        "ตัวอย่างประโยค": {"type": "string", "description": "ประโยคตัวอย่างใหม่ที่สร้างขึ้นโดยใช้คำศัพท์"}
                    },
                    "required": ["คำศัพท์ (English)", "คำแปล (Thai)", "ประโยคที่ปรากฏในข่าว", "ตัวอย่างประโยค"]
                }
            }
        },
        "required": ["vocabulary_list"]
    }

def create_prompt(text_content: str) -> str:
    """สร้าง Prompt สำหรับการวิเคราะห์เนื้อหาข่าว"""
    return f"""
    คุณเป็นผู้เชี่ยวชาญด้านภาษาอังกฤษที่ทำหน้าที่วิเคราะห์ข่าว 
    
    1. **วิเคราะห์**เนื้อหาข่าวภาษาอังกฤษที่ให้มาด้านล่างนี้
    2. **ระบุ**คำศัพท์ภาษาอังกฤษที่สำคัญหรือน่าสนใจที่สุด 5-7 คำ
    3. สำหรับแต่ละคำศัพท์:
        a. **หา**ประโยคเต็มที่ปรากฏคำศัพท์นั้นในเนื้อหา
        b. **แปล**คำศัพท์นั้นเป็นภาษาไทย (คำแปลหลักที่เหมาะสมกับบริบท)
        c. **สร้าง**ประโยคตัวอย่างใหม่ (ที่ไม่ใช่ประโยคในข่าว) ที่ใช้คำศัพท์นั้น
    4. **ส่งออก**ผลลัพธ์ทั้งหมดในรูปแบบ JSON ตาม JSON Schema ที่กำหนดไว้เท่านั้น
    
    **เนื้อหาข่าวสำหรับวิเคราะห์:**
    ---
    {text_content}
    ---
    """
    
# ********** ฟังก์ชันหลักสำหรับ Web Scraping และ API Call **********

def extract_and_analyze_content_with_api(url: str) -> List[Dict[str, Any]]:
    """
    ดึงเนื้อหาจาก URL และใช้ Gemini API ในการวิเคราะห์คำศัพท์
    """
    if not client:
        return []
        
    st.info(f"กำลังดึงเนื้อหาจาก: {url}...")
    
    try:
        # 1. Web Scraping
        headers = {'User-Agent': 'Mozilla/5.0'}
        response_scrape = requests.get(url, headers=headers, timeout=15)
        response_scrape.raise_for_status()
        soup = BeautifulSoup(response_scrape.content, 'html.parser')
        
        # ดึงเนื้อหาที่เป็นข้อความหลัก (ปรับ selector ตามเว็บข่าวจริง)
        text_content = soup.find('body').get_text(separator=' ', strip=True) 
        
        # จำกัดขนาดเนื้อหาเพื่อประหยัด Token (สำคัญ!)
        context_text = text_content[:10000] 
        st.caption(f"ความยาวเนื้อหาที่ถูกส่งไปวิเคราะห์: {len(context_text)} ตัวอักษร")
        
        # 2. API Call ด้วย model.generate_content(prompt)
        prompt = create_prompt(context_text)
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                # บังคับให้ Output เป็น JSON ตาม Schema
                response_mime_type="application/json",
                response_schema=get_response_schema()
            )
        )
        
        # 3. ประมวลผลผลลัพธ์จาก API
        json_output = json.loads(response.text)
        
        if 'vocabulary_list' in json_output:
            st.success("✅ วิเคราะห์คำศัพท์ด้วย API สำเร็จ!")
            return json_output['vocabulary_list']
        else:
            st.error("❌ API Response มีปัญหาหรือไม่เป็นไปตามรูปแบบที่คาดหวัง")
            return []

    except requests.exceptions.RequestException as e:
        st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลจาก URL: {e}")
        return []
    except json.JSONDecodeError:
        st.error("❌ API ตอบกลับมา แต่มีข้อผิดพลาดในการถอดรหัส JSON (Response ไม่เป็น JSON ที่ถูกต้อง)")
        st.code(response.text, language="json") # แสดง Output ดิบ
        return []
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดทั่วไป: {e}")
        return []

# ********** Streamlit App Layout **********

def main():
    st.title("📰 News Vocabulary Extractor (LLM Powered)")
    
    st.markdown("""
        แอปพลิเคชันนี้ใช้ **Web Scraping** ดึงเนื้อหาและใช้ **Gemini API** พร้อม **Prompt** ในการวิเคราะห์และสร้างตารางคำศัพท์.
    """)
    
    news_url = st.text_input(
        "ใส่ลิงค์ (URL) ของเว็บไซต์ข่าวภาษาอังกฤษ:",
        placeholder="เช่น https://www.bbc.com/news/world",
        key="url_input"
    )
    
    if st.button("🔍 เริ่มวิเคราะห์คำศัพท์"):
        if client is None:
            st.error("ไม่สามารถดำเนินการได้ เนื่องจาก API Client ไม่ได้ถูกตั้งค่าอย่างถูกต้อง.")
            return

        if news_url:
            with st.spinner('กำลังดึงข้อมูลและเรียก API เพื่อวิเคราะห์...'):
                # เรียกใช้ฟังก์ชันที่ใช้ API Call
                analysis_data = extract_and_analyze_content_with_api(news_url)
            
            if analysis_data:
                # สร้าง DataFrame และแสดงผล
                df_vocab = pd.DataFrame(analysis_data)
                
                st.subheader("📊 ตารางคำศัพท์สำคัญจากข่าว (สร้างโดย AI)")
                st.dataframe(df_vocab, use_container_width=True)
            else:
                st.warning("ไม่สามารถวิเคราะห์คำศัพท์ได้ (โปรดดูข้อความ Error ด้านบน)")
        else:
            st.error("กรุณาใส่ลิงค์ (URL) ก่อนเริ่มการวิเคราะห์")

if __name__ == "__main__":
    main()
#```
