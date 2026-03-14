import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

# ==========================================
# ⚙️ PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Thailand Real Estate Intelligence",
    layout="wide",
    page_icon="🐦‍🔥"
)

st.markdown("""
<style>
.metric-card {
    background-color:#111827;
    padding:20px;
    border-radius:12px;
    text-align:center;
}
.insight-box {
    background-color: #1e293b;
    border-left: 5px solid #3b82f6;
    padding: 15px;
    border-radius: 5px;
    margin-top: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# พิกัด 77 จังหวัดของไทย สำหรับพล็อตลงแผนที่
THAI_COORDS = {
    'กรุงเทพมหานคร': (13.7563, 100.5018), 'สมุทรปราการ': (13.5993, 100.5968), 'นนทบุรี': (13.8591, 100.5217), 'ปทุมธานี': (14.0208, 100.5250),
    'พระนครศรีอยุธยา': (14.3510, 100.5704), 'อ่างทอง': (14.5896, 100.4550), 'ลพบุรี': (14.7995, 100.6534), 'สิงห์บุรี': (14.8936, 100.4013),
    'ชัยนาท': (15.1852, 100.1251), 'สระบุรี': (14.5289, 100.9101), 'ชลบุรี': (13.3611, 100.9847), 'ระยอง': (12.6814, 101.2816),
    'จันทบุรี': (12.6114, 102.1039), 'ตราด': (12.2428, 102.5175), 'ฉะเชิงเทรา': (13.6904, 101.0780), 'ปราจีนบุรี': (14.0510, 101.3736),
    'นครนายก': (14.2069, 101.2131), 'สระแก้ว': (13.8240, 102.0646), 'นครราชสีมา': (14.9799, 102.0978), 'บุรีรัมย์': (14.9930, 103.1029),
    'สุรินทร์': (14.8818, 103.4936), 'ศรีสะเกษ': (15.1186, 104.3220), 'อุบลราชธานี': (15.2448, 104.8473), 'ยโสธร': (15.7926, 104.1453),
    'ชัยภูมิ': (15.8068, 102.0315), 'อำนาจเจริญ': (15.8597, 104.6258), 'หนองบัวลำภู': (17.2045, 102.4339), 'ขอนแก่น': (16.4322, 102.8236),
    'อุดรธานี': (17.4138, 102.7872), 'เลย': (17.4860, 101.7223), 'หนองคาย': (17.8785, 102.7413), 'มหาสารคาม': (16.1852, 103.3007),
    'ร้อยเอ็ด': (16.0538, 103.6520), 'กาฬสินธุ์': (16.4328, 103.5061), 'สกลนคร': (17.1664, 104.1486), 'นครพนม': (17.3920, 104.7907),
    'มุกดาหาร': (16.5436, 104.7210), 'เชียงใหม่': (18.7883, 98.9853), 'ลำพูน': (18.5745, 99.0087), 'ลำปาง': (18.2888, 99.4930),
    'อุตรดิตถ์': (17.6201, 100.0993), 'แพร่': (18.1446, 100.1403), 'น่าน': (18.7756, 100.7730), 'พะเยา': (19.1666, 99.9022),
    'เชียงราย': (19.9105, 99.8406), 'แม่ฮ่องสอน': (19.3020, 97.9654), 'นครสวรรค์': (15.7005, 100.1356), 'อุทัยธานี': (15.3835, 100.0246),
    'กำแพงเพชร': (16.4828, 99.5258), 'ตาก': (16.8840, 99.1258), 'สุโขทัย': (17.0193, 99.8265), 'พิษณุโลก': (16.8211, 100.2659),
    'พิจิตร': (16.4416, 100.3488), 'เพชรบูรณ์': (16.4150, 101.1569), 'ราชบุรี': (13.5283, 99.8134), 'กาญจนบุรี': (14.0228, 99.5328),
    'สุพรรณบุรี': (14.4742, 100.1123), 'นครปฐม': (13.8140, 100.0373), 'สมุทรสาคร': (13.5475, 100.2736), 'สมุทรสงคราม': (13.4098, 100.0023),
    'เพชรบุรี': (13.1119, 99.9462), 'ประจวบคีรีขันธ์': (11.8021, 99.7977), 'นครศรีธรรมราช': (8.4304, 99.9631), 'กระบี่': (8.0863, 98.9063),
    'พังงา': (8.4501, 98.5283), 'ภูเก็ต': (7.8804, 98.3923), 'สุราษฎร์ธานี': (9.1342, 99.3334), 'ระนอง': (9.9658, 98.6348),
    'ชุมพร': (10.4930, 99.1800), 'สงขลา': (7.1988, 100.5951), 'สตูล': (6.6238, 100.0659), 'ตรัง': (7.5563, 99.6114),
    'พัทลุง': (7.6167, 100.0740), 'ปัตตานี': (6.8682, 101.2504), 'ยะลา': (6.5411, 101.2804), 'นราธิวาส': (6.4255, 101.8253),
    'บึงกาฬ': (18.3608, 103.6465)
}

# ==========================================
# 🗂️ LOAD & PREPROCESS DATA
# ==========================================
@st.cache_data
def load_data():
    file_name = "thai_property_mock_data.csv"
    if not os.path.exists(file_name):
        return pd.DataFrame()
        
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.lower().str.strip()
    
    # 💡 แก้ไข: เพิ่มระบบทำความสะอาดชื่อจังหวัดและอำเภอ
    if "province" in df.columns:
        df["province"] = df["province"].astype(str).str.replace("จังหวัด", "").str.replace("จ.", "").str.strip()
    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.replace("อำเภอ", "").str.replace("อ.", "").str.replace("เขต", "").str.strip()

    if "price" in df.columns:
        if pd.api.types.is_numeric_dtype(df["price"]):
            df["price_num"] = df["price"]
        else:
            df["price_num"] = (
                df["price"].astype(str)
                .str.replace("$","",regex=False)
                .str.replace(",","",regex=False)
                .astype(float)
            )
            
    df = df.dropna(subset=['district', 'province', 'price_num'])
    
    for col in ['bed_rooms', 'bath_rooms', 'carport']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
    # Imputation แยกตามกลุ่มจังหวัด
    for col in ['land_area', 'building_area']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df.groupby('province')[col].transform(lambda x: x.fillna(x.median())).fillna(df[col].median())

    # กรอง Outlier 
    if not df.empty:
        q_low = df["price_num"].quantile(0.001) 
        q_hi  = df["price_num"].quantile(0.995) 
        df = df[(df["price_num"] >= q_low) & (df["price_num"] <= q_hi)]
        
        df = df[(df["bed_rooms"] >= 0) & (df["bed_rooms"] <= 20)]
        df = df[(df["bath_rooms"] >= 0) & (df["bath_rooms"] <= 20)]
            
    # ปรับบริบทไทย (พื้นที่ดินเป็นตารางวา)
    if "land_area" in df.columns:
        df["land_area_wa"] = df["land_area"] / 4.0  
        
    if "building_area" in df.columns and "price_num" in df.columns:
        df["price_per_sqm"] = np.where(df["building_area"] > 0, df["price_num"] / df["building_area"], 0)
        
    if "price_num" in df.columns and not df.empty:
        p33 = df["price_num"].quantile(0.33)
        p67 = df["price_num"].quantile(0.67)
        
        def categorize_price(price):
            if price > p67: return "ระดับหรูหรา"
            elif price > p33: return "ระดับกลาง"
            else: return "ราคาประหยัด"
                
        df["price_segment"] = df["price_num"].apply(categorize_price)

    return df

@st.cache_resource
def train_ml_model(df_for_ml):
    features = ["district", "province", "bed_rooms", "bath_rooms", "carport", "land_area", "building_area"]
    missing_cols = [f for f in features if f not in df_for_ml.columns]
    
    if len(missing_cols) > 0 or len(df_for_ml) < 50:
        return None, 0, features
        
    if len(df_for_ml) > 20000:
        model_df = df_for_ml.dropna(subset=features + ["price_num"]).sample(n=20000, random_state=42)
    else:
        model_df = df_for_ml.dropna(subset=features + ["price_num"])
        
    X = model_df[features]
    y = model_df["price_num"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cat_cols = ["district", "province"]
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )
    
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1))
    ])
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    return rf_model, score, features

df = load_data()

if df.empty or "price_num" not in df.columns:
    st.error("⚠️ ไม่พบข้อมูล หรือไฟล์ thai_property_mock_data.csv ไม่มีคอลัมน์ price")
    st.stop()

rf_model, ml_score, ml_features = train_ml_model(df)

# ดึงค่า Percentile เพื่อนำไปโชว์ในแท็บแบ่งกลุ่มราคา
global_p33 = df["price_num"].quantile(0.33)
global_p67 = df["price_num"].quantile(0.67)

# ==========================================
# 🎛️ SIDEBAR
# ==========================================
st.sidebar.title("🛠️ เครื่องมือจัดการ")
st.sidebar.markdown("---")
with st.sidebar.expander("➕ เพิ่มข้อมูลบ้านใหม่", expanded=True):
    
    # 1. ดึงรายชื่อจังหวัดทั้งหมดจาก DataFrame ที่โหลดมาแล้ว เพื่อทำเป็นตัวเลือก
    available_provinces = sorted(df['province'].dropna().unique().tolist())
    
    # ใช้ st.selectbox แทน st.text_input และไม่อยู่ใน st.form
    new_province = st.selectbox("📍 จังหวัด", options=available_provinces)
    
    # 2. กรองรายชื่ออำเภอ ให้แสดงเฉพาะอำเภอที่อยู่ในจังหวัดที่เลือก
    if new_province:
        available_districts = sorted(df[df['province'] == new_province]['district'].dropna().unique().tolist())
    else:
        available_districts = []
        
    new_district = st.selectbox("🏠 อำเภอ", options=available_districts)
    
    # 3. ข้อมูลอื่นๆ รับค่าตามปกติ
    new_price = st.number_input("ราคา (บาท)", min_value=0.0, step=100000.0, format="%.2f")
    new_bed = st.number_input("จำนวนห้องนอน", min_value=1, step=1)
    new_bath = st.number_input("จำนวนห้องน้ำ", min_value=1, step=1)
    new_carport = st.number_input("ที่จอดรถ", min_value=0, step=1)
    new_land = st.number_input("พื้นที่ดิน (ตร.ม.)", min_value=1.0, step=1.0)
    new_building = st.number_input("พื้นที่ใช้สอย (ตร.ม.)", min_value=1.0, step=1.0)
    
    # 4. เปลี่ยนมาใช้ st.button ธรรมดาแทน st.form_submit_button
    if st.button("💾 บันทึกข้อมูล", use_container_width=True):
        if new_district and new_province and new_price > 0:
            new_data = pd.DataFrame({
                'province': [new_province], 'district': [new_district], 
                'bed_rooms': [int(new_bed)], 'bath_rooms': [int(new_bath)],
                'carport': [int(new_carport)], 'land_area': [new_land], 
                'building_area': [new_building], 'price': [new_price]
            })
            # บันทึกลงไฟล์ CSV
            new_data.to_csv('thai_property_mock_data.csv', mode='a', header=False, index=False)
            
            # ล้างแคชเพื่อให้กราฟดึงข้อมูลใหม่
            st.cache_data.clear() 
            st.cache_resource.clear() 
            st.success("✅ บันทึกสำเร็จ! กำลังรีเฟรชหน้าจอ...")
            st.rerun()
        else:
            st.error("⚠️ กรุณากรอกข้อมูลให้ครบ และราคาต้องมากกว่า 0")

# ==========================================
# 📊 HEADER & KPI CONTAINER
# ==========================================
st.title("🐦‍🔥 ระบบวิเคราะห์อสังหาริมทรัพย์ไทย")
st.caption("แพลตฟอร์มวิเคราะห์ตลาดและคาดการณ์ราคาบ้านด้วย AI ⚠️ ข้อมูลจำลองสำหรับการสาธิตเท่านั้น")

THAI_LABELS = {
    "province": "จังหวัด", "district": "อำเภอ", "price_num": "ราคา (บาท)", 
    "price_per_sqm": "ราคาต่อ ตร.ม. (บาท)", "land_area": "พื้นที่ดิน (ตร.ม.)",
    "land_area_wa": "พื้นที่ดิน (ตร.ว.)", "building_area": "พื้นที่ใช้สอย (ตร.ม.)", 
    "bed_rooms": "ห้องนอน", "bath_rooms": "ห้องน้ำ", "carport": "ที่จอดรถ", 
    "price_segment": "กลุ่มราคา", "count": "จำนวน (หลัง)", "Count": "จำนวน (หลัง)", "Segment": "กลุ่มราคา"
}

SEGMENT_ORDER = ["ระดับหรูหรา", "ระดับกลาง", "ราคาประหยัด"]
COLOR_MAP = {"ระดับหรูหรา": "#e74c3c", "ระดับกลาง": "#f1c40f", "ราคาประหยัด": "#2ecc71"}

kpi_container = st.container()

st.markdown("---")

tab_province, tab_district, tab_segment, tab_deep, tab_ml, tab_data = st.tabs([
    "🌆 1. ภาพรวมระดับจังหวัด", 
    "📊 2. ภาพรวมตลาดระดับอำเภอ", 
    "💎 3. แบ่งกลุ่มราคา", 
    "🎯 4. วิเคราะห์เชิงลึก", 
    "🤖 5. AI ทำนายราคา",
    "📋 6. ข้อมูลดิบ"
])

# ==========================================
# 🌆 TAB 1: Province Level & 🎯 DATA FILTER
# ==========================================
with tab_province:
    with st.expander("🎯 ตัวกรองข้อมูลอัจฉริยะ (Data Filter) - คลิกเพื่อพับ/ขยาย", expanded=False):
        st.caption("เลือกพื้นที่จังหวัดและอำเภอ เพื่อดูการวิเคราะห์ข้อมูลที่คุณสนใจ")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            provinces = sorted(df["province"].dropna().unique())
            select_all_prov = st.checkbox("✅ เลือกทุกจังหวัด", value=True)
            selected_provinces = st.multiselect("📍 เลือกจังหวัด", provinces, default=provinces if select_all_prov else [])
            filtered_by_prov = df[df["province"].isin(selected_provinces)]

        with col_f2:
            districts = sorted(filtered_by_prov["district"].dropna().unique())
            select_all_dist = st.checkbox("✅ เลือกทุกอำเภอในจังหวัดที่เลือก", value=True)
            selected_districts = st.multiselect("🏠 เลือกอำเภอ", districts, default=districts if select_all_dist else [])

    filtered_df = filtered_by_prov[filtered_by_prov["district"].isin(selected_districts)]

    if filtered_df.empty:
        st.warning("⚠️ กรุณาเลือกจังหวัดและอำเภออย่างน้อย 1 รายการเพื่อแสดงผลกราฟ")
        st.stop()
        
    st.markdown("---")

    num_prov = filtered_df["province"].nunique()
    num_dist = filtered_df["district"].nunique()
    st.subheader(f"ภาพรวมระดับจังหวัด (จาก {num_prov} จังหวัด | {num_dist} อำเภอ)")
    
    # 💡 จุดที่แก้ไข: ระบบแปลภาษาอังกฤษเป็นไทย
    def get_real_lat_lon(prov_name):
        name = str(prov_name).strip().lower()
        en_to_th_full = {
            'bangkok': 'กรุงเทพมหานคร', 'samut prakan': 'สมุทรปราการ', 'samutprakan': 'สมุทรปราการ',
            'nonthaburi': 'นนทบุรี', 'pathum thani': 'ปทุมธานี', 'pathumthani': 'ปทุมธานี',
            'phra nakhon si ayutthaya': 'พระนครศรีอยุธยา', 'ayutthaya': 'พระนครศรีอยุธยา',
            'ang thong': 'อ่างทอง', 'angthong': 'อ่างทอง', 'lop buri': 'ลพบุรี', 'lopburi': 'ลพบุรี',
            'sing buri': 'สิงห์บุรี', 'singburi': 'สิงห์บุรี', 'chai nat': 'ชัยนาท', 'chainat': 'ชัยนาท',
            'saraburi': 'สระบุรี', 'chon buri': 'ชลบุรี', 'chonburi': 'ชลบุรี', 'rayong': 'ระยอง',
            'chanthaburi': 'จันทบุรี', 'trat': 'ตราด', 'chachoengsao': 'ฉะเชิงเทรา',
            'prachin buri': 'ปราจีนบุรี', 'prachinburi': 'ปราจีนบุรี', 'nakhon nayok': 'นครนายก', 'nakhonnayok': 'นครนายก',
            'sa kaeo': 'สระแก้ว', 'sakaeo': 'สระแก้ว', 'nakhon ratchasima': 'นครราชสีมา', 'nakhonratchasima': 'นครราชสีมา',
            'buri ram': 'บุรีรัมย์', 'buriram': 'บุรีรัมย์', 'surin': 'สุรินทร์', 'si sa ket': 'ศรีสะเกษ', 'sisaket': 'ศรีสะเกษ',
            'ubon ratchathani': 'อุบลราชธานี', 'ubonratchathani': 'อุบลราชธานี', 'yasothon': 'ยโสธร',
            'chaiyaphum': 'ชัยภูมิ', 'amnat charoen': 'อำนาจเจริญ', 'amnatcharoen': 'อำนาจเจริญ',
            'nong bua lamphu': 'หนองบัวลำภู', 'nongbualamphu': 'หนองบัวลำภู', 'khon kaen': 'ขอนแก่น', 'khonkaen': 'ขอนแก่น',
            'udon thani': 'อุดรธานี', 'udonthani': 'อุดรธานี', 'loei': 'เลย', 'nong khai': 'หนองคาย', 'nongkhai': 'หนองคาย',
            'maha sarakham': 'มหาสารคาม', 'mahasarakham': 'มหาสารคาม', 'roi et': 'ร้อยเอ็ด', 'roiet': 'ร้อยเอ็ด',
            'kalasin': 'กาฬสินธุ์', 'sakon nakhon': 'สกลนคร', 'sakonnakhon': 'สกลนคร',
            'nakhon phanom': 'นครพนม', 'nakhonphanom': 'นครพนม', 'mukdahan': 'มุกดาหาร',
            'chiang mai': 'เชียงใหม่', 'chiangmai': 'เชียงใหม่', 'lamphun': 'ลำพูน', 'lampang': 'ลำปาง',
            'uttaradit': 'อุตรดิตถ์', 'phrae': 'แพร่', 'nan': 'น่าน', 'phayao': 'พะเยา',
            'chiang rai': 'เชียงราย', 'chiangrai': 'เชียงราย', 'mae hong son': 'แม่ฮ่องสอน', 'maehongson': 'แม่ฮ่องสอน',
            'nakhon sawan': 'นครสวรรค์', 'nakhonsawan': 'นครสวรรค์', 'uthai thani': 'อุทัยธานี', 'uthaithani': 'อุทัยธานี',
            'kamphaeng phet': 'กำแพงเพชร', 'kamphaengphet': 'กำแพงเพชร', 'tak': 'ตาก', 'sukhothai': 'สุโขทัย',
            'phitsanulok': 'พิษณุโลก', 'phichit': 'พิจิตร', 'phetchabun': 'เพชรบูรณ์', 'ratchaburi': 'ราชบุรี',
            'kanchanaburi': 'กาญจนบุรี', 'suphan buri': 'สุพรรณบุรี', 'suphanburi': 'สุพรรณบุรี',
            'nakhon pathom': 'นครปฐม', 'nakhonpathom': 'นครปฐม', 'samut sakhon': 'สมุทรสาคร', 'samutsakhon': 'สมุทรสาคร',
            'samut songkhram': 'สมุทรสงคราม', 'samutsongkhram': 'สมุทรสงคราม', 'phetchaburi': 'เพชรบุรี',
            'prachuap khiri khan': 'ประจวบคีรีขันธ์', 'prachuapkhirikhan': 'ประจวบคีรีขันธ์',
            'nakhon si thammarat': 'นครศรีธรรมราช', 'nakhonsithammarat': 'นครศรีธรรมราช',
            'krabi': 'กระบี่', 'phangnga': 'พังงา', 'phang nga': 'พังงา', 'phuket': 'ภูเก็ต',
            'surat thani': 'สุราษฎร์ธานี', 'suratthani': 'สุราษฎร์ธานี', 'ranong': 'ระนอง',
            'chumphon': 'ชุมพร', 'songkhla': 'สงขลา', 'satun': 'สตูล', 'trang': 'ตรัง',
            'phatthalung': 'พัทลุง', 'pattani': 'ปัตตานี', 'yala': 'ยะลา', 'narathiwat': 'นราธิวาส',
            'bueng kan': 'บึงกาฬ', 'buengkan': 'บึงกาฬ'
        }
        
        thai_name = en_to_th_full.get(name, str(prov_name).strip())
        return pd.Series(THAI_COORDS.get(thai_name, (13.0, 101.0)))

    prov_summary = filtered_df.groupby("province").agg({"price_num":"mean", "land_area_wa":"mean", "district":"count"}).reset_index()
    prov_summary.rename(columns={"district":"count"}, inplace=True)
    prov_summary[['lat', 'lon']] = prov_summary['province'].apply(get_real_lat_lon)

    # เช็คและเตือนกรณีมีคำแปลกๆ หลุดรอดมา
    missing_coords = prov_summary[prov_summary['lat'] == 13.0]['province'].tolist()
    if missing_coords:
        st.warning(f"⚠️ ยังมีบางจังหวัดที่ระบบหาไม่เจอ (อาจสะกดผิด): {', '.join(missing_coords)}")

    col_side, col_map = st.columns([2, 2])
    with col_side:
        prov_count = filtered_df["province"].value_counts().reset_index()
        prov_count.columns = ["province", "count"]
        fig_prov1 = px.pie(prov_count, values="count", names="province", hole=0.4,
                           title="1. สัดส่วนประกาศขาย", labels=THAI_LABELS)
        fig_prov1.update_layout(margin={"r":0,"t":40,"l":0,"b":0}) 
        st.plotly_chart(fig_prov1, use_container_width=True)

    with col_map:
        fig_map = px.scatter_mapbox(prov_summary, lat="lat", lon="lon", size="count", color="price_num",
                                    hover_name="province", zoom=4.5, center={"lat": 13.0, "lon": 101.0},
                                    mapbox_style="open-street-map", 
                                    title="🗺️ แผนที่กระจายตัวระดับจังหวัด",
                                    color_continuous_scale="Viridis",
                                    labels={"count": "จำนวน (หลัง)", "price_num": "ราคาเฉลี่ย (บาท)"})
        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    colC2, colC3, colC4 = st.columns(3)
    
    with colC2:
        prov_avg_price = prov_summary.sort_values("price_num", ascending=False)
        fig_prov2 = px.bar(prov_avg_price, x="price_num", y="province", orientation="h",
                           title="2. ราคาเฉลี่ยรายจังหวัด", labels=THAI_LABELS, color="price_num", color_continuous_scale="Viridis")
        fig_prov2.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig_prov2, use_container_width=True)
    
    with colC3:
        fig_prov3 = px.box(filtered_df, x="province", y="price_num", color="province",
                           title="3. ช่วงความกว้างของราคา", labels=THAI_LABELS, points="outliers")
        fig_prov3.update_layout(showlegend=False)
        st.plotly_chart(fig_prov3, use_container_width=True)

    with colC4:
        prov_avg_sqm = filtered_df.groupby("province")["price_per_sqm"].mean().reset_index().sort_values("price_per_sqm", ascending=False)
        fig_prov4 = px.bar(prov_avg_sqm, x="province", y="price_per_sqm",
                           title="4. ราคาเฉลี่ย/ตร.ม.", labels=THAI_LABELS, color="price_per_sqm", color_continuous_scale="Plasma")
        fig_prov4.update_layout(showlegend=False)
        st.plotly_chart(fig_prov4, use_container_width=True)

# ==========================================
# 💡 POPULATE KPI 
# ==========================================
with kpi_container:
    st.subheader("📌 สรุปข้อมูลสำคัญ")
    current_avg_price = filtered_df['price_num'].mean()
    current_avg_sqm = filtered_df['price_per_sqm'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏠 จำนวนบ้านทั้งหมด", f"{len(filtered_df):,} หลัง")
    col2.metric("💰 ราคาเฉลี่ย", f"{current_avg_price/1e6:,.2f} ล้านบาท")
    col3.metric("🎯 ราคากลาง (Median)", f"{filtered_df['price_num'].median()/1e6:,.2f} ล้านบาท")
    col4.metric("📐 ราคาเฉลี่ย/ตร.ม.", f"{current_avg_sqm:,.0f} บาท")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("📈 ราคาสูงสุด", f"{filtered_df['price_num'].max()/1e6:,.2f} ล้านบาท")
    col6.metric("📉 ราคาต่ำสุด", f"{filtered_df['price_num'].min()/1e6:,.2f} ล้านบาท")

    mode_bed = int(filtered_df['bed_rooms'].mode()[0]) if not filtered_df.empty else 0
    mode_bath = int(filtered_df['bath_rooms'].mode()[0]) if not filtered_df.empty else 0
    mode_carport = int(filtered_df['carport'].mode()[0]) if not filtered_df.empty else 0
    col7.metric("⭐ สเปคยอดฮิต", f"🛏️ {mode_bed} | 🛁 {mode_bath} | 🚗 {mode_carport}")
    col8.metric("🏗️ พื้นที่เฉลี่ย (ดิน/ใช้สอย)", f"{filtered_df['land_area_wa'].mean():.0f} ตร.ว. / {filtered_df['building_area'].mean():.0f} ตร.ม.")

# ==========================================
# 📊 TAB 2: District Level 
# ==========================================
with tab_district:
    st.subheader("วิเคราะห์ภาพรวมตลาดระดับอำเภอ")
    
    colA, colB = st.columns(2)
    with colA:
        top_districts_count = filtered_df["district"].value_counts().head(10).reset_index()
        fig10 = px.bar(top_districts_count, x="count", y="district", orientation="h", 
                       title="1. Top 10 อำเภอที่มีประกาศขายมากที่สุด", labels=THAI_LABELS, color="count", color_continuous_scale="Blues")
        fig10.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig10, use_container_width=True)
    with colB:
        avg_price = filtered_df.groupby("district")["price_num"].mean().reset_index().sort_values("price_num", ascending=False).head(10)
        fig2 = px.bar(avg_price, x="price_num", y="district", orientation="h", 
                      title="2. Top 10 อำเภอที่ราคาเฉลี่ยแพงที่สุด", labels=THAI_LABELS, color="price_num", color_continuous_scale="Reds")
        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)
    
    colC, colD = st.columns(2)
    with colC:
        fig1 = px.histogram(filtered_df, x="price_num", nbins=50, 
                            title="3. การกระจายตัวของราคาบ้านทั้งหมด", labels=THAI_LABELS, color_discrete_sequence=['#9b59b6'])
        st.plotly_chart(fig1, use_container_width=True)
    with colD:
        fig3 = px.scatter(filtered_df, x="land_area_wa", y="price_num", color="district", 
                          title="4. ความสัมพันธ์ระหว่างพื้นที่ดิน (ตร.ว.) และราคา", labels=THAI_LABELS)
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    top_5_dist_names = top_districts_count['district'].head(5).tolist()
    df_top_5_dist = filtered_df[filtered_df['district'].isin(top_5_dist_names)]
    fig_dist_box = px.box(df_top_5_dist, x="district", y="price_num", color="district",
                          title="5. ช่วงความกว้างราคา เฉพาะ 5 อำเภอยอดฮิต", labels=THAI_LABELS)
    st.plotly_chart(fig_dist_box, use_container_width=True)

# ==========================================
# 💎 TAB 3: Segmentation 
# ==========================================
with tab_segment:
    st.subheader("การแบ่งกลุ่มช่วงราคา (หรูหรา / กลาง / ประหยัด)")
    
    st.info(f"**ℹ️ เกณฑ์การแบ่งกลุ่มราคา (อิงจากเปอร์เซ็นไทล์ของข้อมูลทั้งหมด):**\n\n"
            f"🟢 **ราคาประหยัด:** ไม่เกิน {global_p33/1e6:,.2f} ล้านบาท | "
            f"🟡 **ระดับกลาง:** มากกว่า {global_p33/1e6:,.2f} ถึง {global_p67/1e6:,.2f} ล้านบาท | "
            f"🔴 **ระดับหรูหรา:** มากกว่า {global_p67/1e6:,.2f} ล้านบาท ขึ้นไป")
    
    prov_segment = filtered_df.groupby(["province", "price_segment"]).size().reset_index(name="count")
    fig_prov5 = px.bar(prov_segment, x="province", y="count", color="price_segment", barmode="group",
                       category_orders={"price_segment": SEGMENT_ORDER},
                       title="1. สัดส่วนกลุ่มราคาแยกตามจังหวัด", labels=THAI_LABELS, color_discrete_map=COLOR_MAP)
    st.plotly_chart(fig_prov5, use_container_width=True)

    colE, colF = st.columns(2)
    with colE:
        segment_count = filtered_df["price_segment"].value_counts().reset_index()
        segment_count.columns = ["Segment", "Count"]
        segment_count['Segment'] = pd.Categorical(segment_count['Segment'], categories=SEGMENT_ORDER, ordered=True)
        segment_count = segment_count.sort_values('Segment')
        fig8 = px.pie(segment_count, values="Count", names="Segment", hole=0.4,
                      title="2. สัดส่วนโดยรวม", labels=THAI_LABELS, color="Segment", color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig8, use_container_width=True)

    with colF:
        top_15_districts = filtered_df["district"].value_counts().head(15).index
        df_top_15 = filtered_df[filtered_df["district"].isin(top_15_districts)]
        fig9 = px.histogram(df_top_15, y="district", color="price_segment", barmode="stack", orientation="h",
                            category_orders={"price_segment": SEGMENT_ORDER},
                            title="3. สัดส่วนกลุ่มราคาในอำเภอยอดนิยม", labels=THAI_LABELS, color_discrete_map=COLOR_MAP)
        fig9.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig9, use_container_width=True)

    st.markdown("### 4. เจาะลึกอำเภอยอดนิยม แยกตามกลุ่มราคา")
    col_l, col_m, col_b = st.columns(3) 
    
    with col_l:
        luxury_df = filtered_df[filtered_df["price_segment"] == "ระดับหรูหรา"]
        if not luxury_df.empty:
            top_luxury = luxury_df["district"].value_counts().head(5).reset_index()
            fig_l = px.bar(top_luxury, x="count", y="district", orientation="h", 
                           title="Top 5 อำเภอ (หรูหรา)", color_discrete_sequence=['#e74c3c'])
            fig_l.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_l, use_container_width=True)
    with col_m:
        mid_df = filtered_df[filtered_df["price_segment"] == "ระดับกลาง"]
        if not mid_df.empty:
            top_mid = mid_df["district"].value_counts().head(5).reset_index()
            fig_m = px.bar(top_mid, x="count", y="district", orientation="h", 
                           title="Top 5 อำเภอ (กลาง)", color_discrete_sequence=['#f1c40f'])
            fig_m.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_m, use_container_width=True)
    with col_b:
        budget_df = filtered_df[filtered_df["price_segment"] == "ราคาประหยัด"]
        if not budget_df.empty:
            top_budget = budget_df["district"].value_counts().head(5).reset_index()
            fig_b = px.bar(top_budget, x="count", y="district", orientation="h", 
                           title="Top 5 อำเภอ (ประหยัด)", color_discrete_sequence=['#2ecc71'])
            fig_b.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_b, use_container_width=True)

    fig_seg_box = px.box(filtered_df, x="price_segment", y="land_area_wa", color="price_segment",
                         category_orders={"price_segment": SEGMENT_ORDER}, color_discrete_map=COLOR_MAP,
                         title="5. เปรียบเทียบพื้นที่ดิน (ตร.ว.) ในแต่ละกลุ่มราคา", labels=THAI_LABELS, points=False)
    st.plotly_chart(fig_seg_box, use_container_width=True)

# ==========================================
# 🎯 TAB 4: Deep Analysis (Revised for Clarity, Language, and Interactivity)
# ==========================================
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- ส่วนนี้ขออนุญาตเพิ่มเข้าไปเพื่อให้แน่ใจว่า THAI_LABELS มีครบ ---
THAI_LABELS_EXT = {
    'bed_rooms': "ห้องนอน", 
    'bath_rooms': "ห้องน้ำ", 
    'carport': "ที่จอดรถ", 
    'land_area': "ที่ดิน (ตร.ว.)", 
    'building_area': "พื้นที่ใช้สอย (ตร.ม.)", 
    'price': "ราคา (บาท)", 
    'price_per_sqm': "ราคาต่อ ตร.ม. (บาท)"
}

with tab_deep:
    st.subheader("วิเคราะห์ข้อมูลเชิงลึก (Deep Dive)")
    
    # --- Feature Engineering พื้นฐาน ---
    if 'price_per_sqm' not in filtered_df.columns:
        filtered_df['price_per_sqm'] = filtered_df['price'] / filtered_df['building_area']

    colG, colH = st.columns(2)
    
    with colG:
        # 1. Heatmap: ภาษาไทย 100%
        corr_cols = ['bed_rooms', 'bath_rooms', 'carport', 'land_area', 'building_area', 'price', 'price_per_sqm']
        corr_df = filtered_df[corr_cols].rename(columns=THAI_LABELS_EXT)
        corr = corr_df.corr()
        
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", 
                             title="ความสัมพันธ์ของปัจจัยต่างๆ (Correlation Matrix)", labels=THAI_LABELS_EXT)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with colH:
        # 2. Bubble Chart: ดูความกว้างของที่ดิน vs พื้นที่ใช้สอย โดยมีราคาเป็นขนาดของฟองสบู่
        fig_area = px.scatter(filtered_df, x="land_area", y="building_area", size="price", 
                              color="province", hover_name="district",
                              title="ความสัมพันธ์: พื้นที่ดิน vs พื้นที่ใช้สอย (ขนาดบับเบิ้ล = ราคา)",
                              labels={"land_area": "ขนาดที่ดิน", "building_area": "พื้นที่ใช้สอย"})
        st.plotly_chart(fig_area, use_container_width=True)
        
    st.markdown("---")
    colI, colJ = st.columns(2)
    
    with colI:
        # 3. Boxplot: การกระจายตัวของราคาในแต่ละจังหวัด
        fig_prov = px.box(filtered_df, x="province", y="price", color="province",
                          title="การกระจายตัวของราคาในแต่ละจังหวัด",
                          labels={"province": "จังหวัด", "price": "ราคา (บาท)"})
        fig_prov.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_prov, use_container_width=True)
        
    with colJ:
        # 4. Scatter Plot + Trendline: เจาะลึกราคาต่อพื้นที่ใช้สอย
        fig_scatter = px.scatter(filtered_df, x="building_area", y="price", color="bed_rooms", 
                                 trendline="ols", title="แนวโน้มราคาตามขนาดพื้นที่ใช้สอย",
                                 labels={"building_area": "พื้นที่ใช้สอย (ตร.ม.)", "price": "ราคา (บาท)", "bed_rooms": "จำนวนห้องนอน"}, 
                                 hover_data=["province", "district"])
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================
    # 🌟 ส่วนที่เพิ่มใหม่: เจาะลึกปัจจัยที่มีผลต่อราคา (Factor Impact Analysis)
    # ==========================================
    st.markdown("---")
    st.subheader("ปัจจัยที่มีผลกระทบต่อราคา (Factor Impact Analysis)")
    
    colK, colL = st.columns(2)
    
    with colK:
        # 5. Bar Chart: ดึงค่า Correlation เฉพาะที่เกี่ยวกับ "ราคา" มาพล็อต
        # เอาตัวแปร 'ราคา (บาท)' ออกจากแกนเพื่อไม่ให้มันเปรียบเทียบกับตัวเอง (ซึ่งจะได้ค่า 1 เสมอ)
        price_corr = corr['ราคา (บาท)'].drop('ราคา (บาท)').sort_values(ascending=True)
        
        # กำหนดสี: ถ้าค่าบวก (แปลว่าเพิ่มแล้วราคาขึ้น) ให้เป็นสีเขียว ถ้าค่าลบเป็นสีแดง
        colors = ['#ff9999' if val < 0 else '#66b3ff' for val in price_corr.values]
        
        fig_impact = px.bar(x=price_corr.values, y=price_corr.index, orientation='h',
                            title="ปัจจัยไหนส่งผลให้ราคาบ้านสูงขึ้นมากที่สุด?",
                            labels={'x': 'ระดับความสัมพันธ์ (Correlation)', 'y': 'ปัจจัย'},
                            color_discrete_sequence=[colors])
        
        fig_impact.update_layout(showlegend=False)
        st.plotly_chart(fig_impact, use_container_width=True)
        
    with colL:
        # 6. Interactive Scatter + Trendline: ให้ผู้ใช้เลือกตัวแปรได้เอง
        st.markdown("**ทดลองปรับเปลี่ยนปัจจัยเพื่อดูแนวโน้มราคา**")
        
        # สร้างตัวเลือก (Dictionary สำหรับทำแมปปิ้งภาษาไทย -> ชื่อคอลัมน์)
        feature_options = {
            "จำนวนห้องน้ำ": "bath_rooms",
            "จำนวนห้องนอน": "bed_rooms",
            "จำนวนที่จอดรถ": "carport",
            "ขนาดที่ดิน": "land_area",
            "พื้นที่ใช้สอย": "building_area"
        }
        
        # กล่องให้ผู้ใช้เลือก
        selected_thai_label = st.selectbox("เลือกปัจจัยที่สนใจ:", list(feature_options.keys()))
        selected_col = feature_options[selected_thai_label]
        
        # สร้างกราฟตามสิ่งที่ผู้ใช้เลือก
        fig_interactive = px.scatter(filtered_df, x=selected_col, y="price", 
                                     opacity=0.5, trendline="ols", trendline_color_override="red",
                                     title=f"ความสัมพันธ์ระหว่าง {selected_thai_label} กับราคา",
                                     labels={selected_col: selected_thai_label, "price": "ราคา (บาท)"},
                                     hover_data=["province", "district"])
        st.plotly_chart(fig_interactive, use_container_width=True)

# ==========================================
# 🤖 TAB 5: ML & 📋 TAB 6: Data
# ==========================================
with tab_ml:
    st.subheader("🤖 AI ประเมินราคาบ้าน")
    
    if rf_model is not None:
        # --- คำนวณ RMSE อัตโนมัติ ป้องกัน Error ---
        try:
            # ใช้โมเดลทำนายข้อมูลที่มีเพื่อหาค่าความคลาดเคลื่อนเฉลี่ย
            y_actual = filtered_df['price']
            y_predicted = rf_model.predict(filtered_df[ml_features])
            rmse_score = np.sqrt(mean_squared_error(y_actual, y_predicted))
        except Exception as e:
            rmse_score = 0 # กันเหนียวเผื่อมี error อื่น

        # --- แสดง Metrics ---
        col_met1, col_met2 = st.columns(2)
        with col_met1:
            st.metric("ความแม่นยำ (R² Score)", f"{ml_score:.3f}", help="เข้าใกล้ 1.0 แปลว่า AI ทายได้แม่นยำมาก")
        with col_met2:
            st.metric("ความคลาดเคลื่อนเฉลี่ย (RMSE)", f"± ฿{rmse_score:,.0f}", help="AI มักจะทายราคาคลาดเคลื่อนเฉลี่ยประมาณนี้")
        
        st.markdown("---")
        
        col_ml1, col_ml2 = st.columns(2)
        with col_ml1:
            model_step = rf_model.named_steps['regressor']
            feature_names = rf_model.named_steps['preprocessor'].get_feature_names_out(ml_features)
            importance = pd.DataFrame({"Feature": feature_names, "Importance": model_step.feature_importances_}).sort_values(by="Importance", ascending=False).head(15)
            importance['Feature'] = importance['Feature'].str.replace('cat__', '').str.replace('remainder__', '')
            importance['Feature'] = importance['Feature'].replace({'bed_rooms': 'ห้องนอน', 'bath_rooms': 'ห้องน้ำ', 'carport': 'ที่จอดรถ', 'land_area': 'พื้นที่ดิน', 'building_area': 'พื้นที่ใช้สอย'})
            fig_ml = px.bar(importance, x="Importance", y="Feature", orientation="h", title="น้ำหนักปัจจัยที่มีผลต่อราคา")
            fig_ml.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="ระดับความสำคัญ", yaxis_title="")
            st.plotly_chart(fig_ml, use_container_width=True)
            
        with col_ml2:
            st.markdown("### 🔮 ลองให้ AI ทำนายราคา")
            input_province = st.selectbox("เลือกจังหวัด", options=sorted(df["province"].unique()))
            districts_for_ai = sorted(df[df["province"] == input_province]["district"].unique())
            input_district = st.selectbox("เลือกอำเภอ", options=districts_for_ai)
            
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                input_land_wa = st.number_input("พื้นที่ดิน (ตร.ว.)", min_value=1.0, value=50.0) 
                input_bed = st.number_input("จำนวนห้องนอน", min_value=1, step=1, value=3)
                input_carport = st.number_input("ที่จอดรถ", min_value=0, step=1, value=1)
            with col_in2:
                input_building = st.number_input("พื้นที่ใช้สอย (ตร.ม.)", min_value=1.0, value=150.0)
                input_bath = st.number_input("จำนวนห้องน้ำ", min_value=1, step=1, value=2)
            
            if st.button("ประเมินราคาเลย!", type="primary"):
                input_land_sqm = input_land_wa * 4 
                input_data = pd.DataFrame([[input_district, input_province, input_bed, input_bath, input_carport, input_land_sqm, input_building]], columns=ml_features)
                
                prediction = rf_model.predict(input_data)[0]
                st.success(f"💰 ราคาประเมินจาก AI: **{prediction/1e6:,.2f} ล้านบาท**")
                
                # --- เทียบราคาตลาด ---
                st.markdown("#### 📊 เทียบกับราคาตลาด (อ้างอิงจากฐานข้อมูล)")
                similar_houses = df[(df['province'] == input_province) & 
                                    (df['district'] == input_district) & 
                                    (df['bed_rooms'] == input_bed)]
                
                if not similar_houses.empty:
                    avg_actual_price = similar_houses['price'].mean()
                    min_price = similar_houses['price'].min()
                    max_price = similar_houses['price'].max()
                    
                    st.info(f"**ราคาเฉลี่ยบ้าน {input_bed} ห้องนอน ใน {input_district}**:\n\n 🏷️ **{avg_actual_price/1e6:,.2f} ล้านบาท** (ช่วงราคา: {min_price/1e6:,.2f} - {max_price/1e6:,.2f} ลบ.)")
                    
                    diff_percent = ((prediction - avg_actual_price) / avg_actual_price) * 100
                    if diff_percent > 0:
                        st.caption(f"📈 *AI ประเมินราคาสูงกว่าตลาดเฉลี่ย {diff_percent:.1f}%*")
                    else:
                        st.caption(f"📉 *AI ประเมินราคาต่ำกว่าตลาดเฉลี่ย {abs(diff_percent):.1f}%*")
                else:
                    st.warning(f"ไม่มีข้อมูลบ้าน {input_bed} ห้องนอน ในพื้นที่ {input_district} เพื่อใช้เปรียบเทียบ")
    else:
        st.error("⚠️ ข้อมูลไม่เพียงพอสำหรับการสร้าง AI Model")

with tab_data:
    st.subheader("ตารางข้อมูลดิบ (ตามเงื่อนไขที่กรอง)")
    display_df = filtered_df.drop(columns=['lat', 'lon'], errors='ignore')
    # แสดงหัวตารางเป็นภาษาไทย
    try:
        st.dataframe(display_df.rename(columns=THAI_LABELS_EXT), use_container_width=True)
    except NameError:
        st.dataframe(display_df, use_container_width=True)