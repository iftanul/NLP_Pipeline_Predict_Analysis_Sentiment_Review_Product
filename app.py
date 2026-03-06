import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import html
import nltk
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

# ==========================================
# 1. KONFIGURASI AWAL
# ==========================================
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Produk", 
    page_icon="🛍️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM CSS (ULTRA PREMIUM DARK MODE & BUG FIXES)
# ==========================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* 1. BACKGROUND UTAMA */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #121212 !important; 
        color: #E0E0E0 !important;
    }

    /* 2. SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #0A0A0A !important;
        border-right: 1px solid #2A2A2A;
    }
    [data-testid="stSidebarNav"] span, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    /* 3. GLOWING BUTTON (THE REAL CONTRAST) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #9EE05B 0%, #68B02A 100%) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 4px 15px rgba(158, 224, 91, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 25px rgba(158, 224, 91, 0.7) !important;
        transform: translateY(-3px) scale(1.02);
    }

    /* 4. HERO SECTION */
    .hero-box {
        background: linear-gradient(145deg, #1e1e1e, #252525);
        border-left: 5px solid #9EE05B;
        padding: 2.5rem 3rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    .hero-box h1 { color: #FFFFFF; font-weight: 800; font-size: 2.5rem; margin-bottom: 0.5rem; }
    .hero-box p { color: #A0A0A0; font-size: 1.15rem; margin-bottom: 0; }

    /* 5. METRIC CARDS */
    .metric-container { display: flex; gap: 1.5rem; }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #333;
        text-align: left;
        flex: 1;
        transition: all 0.3s ease;
        border-top: 4px solid #9EE05B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #9EE05B;
        box-shadow: 0 10px 20px rgba(158, 224, 91, 0.15);
    }
    .metric-card.alert { border-top: 4px solid #FF3B30; }
    .metric-card.warning { border-top: 4px solid #FFCC00; }
    
    .metric-title { color: #888888; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.5rem; }
    .metric-value { color: #FFFFFF; font-size: 2.2rem; font-weight: 800; line-height: 1.1; }

    /* 6. SECTION CONTAINERS */
    .dashboard-section {
        background-color: #1E1E1E;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #333;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* 7. RESULT CARDS */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        animation: fadeIn 0.5s ease-in-out;
    }
    .result-positive {
        background-color: rgba(158, 224, 91, 0.05);
        border: 2px solid #9EE05B;
        box-shadow: 0 0 20px rgba(158, 224, 91, 0.15);
    }
    .result-negative {
        background-color: rgba(255, 59, 48, 0.05);
        border: 2px solid #FF3B30;
        box-shadow: 0 0 20px rgba(255, 59, 48, 0.15);
    }
    .result-title { font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem; }
    .result-positive .result-title { color: #9EE05B; }
    .result-negative .result-title { color: #FF3B30; }

    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

    /* 8. DATAFRAME FIXES */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid #444;
        background-color: #1E1E1E !important;
    }
    
    /* Input Text Area Glowing Focus */
    .stTextArea textarea {
        background-color: #111 !important; color: white !important; border: 1px solid #444 !important; font-size: 1.1rem; padding: 1rem;
    }
    .stTextArea textarea:focus {
        border-color: #9EE05B !important; box-shadow: 0 0 10px rgba(158, 224, 91, 0.3) !important;
    }

    /* 9. FILE UPLOADER FIX */
    [data-testid="stFileUploader"] {
        background-color: #1A1A1A !important; border: 2px dashed #444 !important; border-radius: 12px !important; padding: 1.5rem !important;
    }
    [data-testid="stFileUploader"] section { background-color: transparent !important; }
    [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] small { color: #E0E0E0 !important; }
    [data-testid="stFileUploader"] button {
        background-color: #111 !important; color: #9EE05B !important; border: 1px solid #9EE05B !important; border-radius: 6px !important; font-weight: 700 !important; padding: 0.5rem 1rem !important;
    }
    [data-testid="stFileUploader"] button:hover { background-color: #9EE05B !important; color: #000 !important; }

    /* 10. WARNING & ALERT FIX */
    [data-testid="stAlert"] { background-color: #1E1E1E !important; border: 1px solid #444 !important; border-radius: 8px !important; }
    [data-testid="stAlert"] p, [data-testid="stAlert"] span { color: #FFFFFF !important; font-weight: 500 !important; }

    h1, h2, h3, h4 { color: #FFFFFF !important; }
    .caption-text { color: #A0A0A0 !important; font-size: 0.95rem; margin-bottom: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ==========================================
# 3. KAMUS & STOPWORDS
# ==========================================
slang_dict = {
    "bgt": "banget", "yg": "yang", "jg": "juga", "tdk": "tidak", "sdh": "sudah",
    "blm": "belum", "krn": "karena", "tp": "tapi", "ga": "tidak", "gak": "tidak",
    "gx": "tidak", "gn": "begini", "emang": "memang", "km": "kamu", "sy": "saya",
    "gw": "saya", "ok": "oke", "oke": "oke", "sip": "siap", "recom": "recommended",
    "mantap": "mantap", "cepet": "cepat", "fast": "cepat", "res": "respon",
    "brg": "barang", "pake": "pakai", "pakek": "pakai", "udah": "sudah",
    "utk": "untuk", "aja": "saja", "smua": "semua", "dapet": "dapat",
    "nyampe": "sampai", "sampe": "sampai", "dtg": "datang", "mks": "makasih",
    "trims": "terima kasih", "tks": "terima kasih", "tq": "terima kasih"
}

nltk_sw = stopwords.words('indonesian')
factory = StopWordRemoverFactory()
sastrawi_stopword = factory.get_stop_words()
final_stopwords = list(set(nltk_sw + sastrawi_stopword))

negation_word = [
    'tidak','nggak','gak','ga','tak','kurang','belum','bukan',
    'jangan','enggak','gaklah','gausah','belumlah','bukanlah',
    'tidaklah','tanpa','kurangnya','susah','gagal','kecewa',
    'parah','jelek','buruk'
]
final_stopwords = list(set(final_stopwords) - set(negation_word))

ecommerce_noise = [
    'gan','sis','kak','seller','admin','tokopedia','yg','sdh','dah',
    'dgn','nya','terima','kasih','tks','thx','thanks','terimakasih','tp','makasih'
]
final_stopwords = list(set(final_stopwords) | set(ecommerce_noise))
final_stopwords_set = set(final_stopwords)

# ==========================================
# 4. FUNGSI PREPROCESSING & ANALYTICS
# ==========================================
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    words = [w for w in words if w not in final_stopwords_set]
    return " ".join(words)

def clean_meta_data(text):
    if pd.isna(text): return "Unknown"
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_sold_to_numeric(sold_str):
    if pd.isna(sold_str): return 0
    sold_str = str(sold_str).lower().strip()
    sold_str = re.sub(r'[^0-9a-z\.,]', '', sold_str)
    multiplier = 1
    if 'rb' in sold_str or 'k' in sold_str:
        multiplier = 1000
        sold_str = re.sub(r'[rb|k]', '', sold_str)
    elif 'jt' in sold_str or 'm' in sold_str:
        multiplier = 1000000
        sold_str = re.sub(r'[jt|m]', '', sold_str)
    sold_str = sold_str.replace(',', '.')
    try: return int(float(sold_str) * multiplier)
    except Exception: return 0

def get_risk_scores(data, min_reviews=20):
    risk_df = data.groupby('product_name').agg(
        total_review=('sentiment_label', 'count'),
        neg_count=('sentiment_label', lambda x: (x == 'Negative').sum()),
        total_sold=('sold_numeric', 'max')
    ).reset_index()
    risk_df = risk_df[risk_df['total_review'] >= min_reviews]
    risk_df['pct_negative'] = risk_df['neg_count'] / risk_df['total_review']
    risk_df['risk_score'] = risk_df['pct_negative'] * np.log1p(risk_df['total_review'])
    return risk_df.sort_values('risk_score', ascending=False)

def plot_ngram_wordcloud(data, ngram_level):
    neg_data = data[data['sentiment_label'] == 'Negative']['clean_text'].dropna()
    if len(neg_data) == 0: return None
    try:
        cv = CountVectorizer(ngram_range=(ngram_level, ngram_level), max_features=100)
        matrix = cv.fit_transform(neg_data)
        freqs = dict(zip(cv.get_feature_names_out(), matrix.sum(axis=0).A1))
        
        wc = WordCloud(
            width=800, height=400, background_color='#1E1E1E', 
            colormap='autumn', max_words=100
        ).generate_from_frequencies(freqs)
        
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1E1E1E')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        return fig
    except Exception: return None

# ==========================================
# 5. LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    model = joblib.load('models/model_stacking.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model_stacking, vectorizer = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Gagal memuat model Machine Learning. Error: {e}")
    models_loaded = False

# ==========================================
# 6. SIDEBAR NAVIGATION & SETTINGS
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #9EE05B !important; font-weight: 800;'>Predict Sentiment</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation Menu:",
        ["💬 Sandbox Predictor", "📊 Analytics Dashboard"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Threshold Settings")
    st.caption("Atur probabilitas batas untuk mengklasifikasikan sentimen negatif. Semakin tinggi, model semakin berhati-hati (presisi tinggi).")
    
    # === THRESHOLD SLIDER  ===
    prediction_threshold = st.slider(
        "Negative Threshold", 
        min_value=0.10, 
        max_value=0.95, 
        value=0.70,
        step=0.05
    )
    
    if not models_loaded:
        st.stop()

# ==========================================
# 7. PAGE 1: SINGLE PREDICTOR (WITH PROBABILITY)
# ==========================================
if menu == "💬 Sandbox Predictor":
    
    st.markdown("""
    <div class="hero-box">
        <h1>💬 Live Sandbox Predictor</h1>
        <p>Ketik ulasan untuk mendeteksi keluhan.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("### Uji Coba Model")
        st.markdown(f"<div class='caption-text'>Model mendeteksi sentimen menggunakan ambang batas / threshold <b>{prediction_threshold}</b> yang Anda atur di sidebar.</div>", unsafe_allow_html=True)
        
        user_input = st.text_area("Review Text", height=150, placeholder="Contoh: Barangnya cepat rusak dan sellernya tidak responsif sama sekali...", label_visibility="collapsed")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True)

        if analyze_btn:
            if user_input.strip() == "":
                st.warning("⚠️ Masukkan teks terlebih dahulu.")
            else:
                with st.spinner("🧠 Memproses text..."):
                    clean_text = preprocess_text(user_input)
                    features = vectorizer.transform([clean_text])
                    
                    # === MENGGUNAKAN PREDICT_PROBA ===
                    probabilities = model_stacking.predict_proba(features)[0]
                    
                    # Asumsi: Indeks 1 adalah kelas Negatif (1), Indeks 0 kelas Positif (0)
                    prob_negative = probabilities[1] 
                    
                    sentiment_result = "Negative" if prob_negative >= prediction_threshold else "Positive"
                    
                st.markdown("---")
                
                # Menampilkan Confidence Score di UI
                confidence_percentage = prob_negative * 100
                
                if sentiment_result == "Positive":
                    st.markdown(f"""
                    <div class="result-card result-positive">
                        <div class="result-title">✨ POSITIVE SENTIMENT</div>
                        <p style="color: #E0E0E0; margin: 0;">Sistem mendeteksi ulasan ini <b>aman</b> (Probabilitas Negatif di bawah batas).<br>
                        <span style="color: #9EE05B; font-weight: bold; font-size: 1.1rem;">Negative Probability: {confidence_percentage:.1f}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card result-negative">
                        <div class="result-title">⚠️ NEGATIVE SENTIMENT</div>
                        <p style="color: #E0E0E0; margin: 0;">Peringatan! Ulasan ini terindikasi memiliki <b>keluhan atau ketidakpuasan</b> pelanggan.<br>
                        <span style="color: #FF3B30; font-weight: bold; font-size: 1.1rem;">Negative Probability: {confidence_percentage:.1f}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    

# ==========================================
# 8. PAGE 2: BATCH ANALYTICS DASHBOARD
# ==========================================
elif menu == "📊 Analytics Dashboard":
    
    st.markdown("""
    <div class="hero-box">
        <h1>Analytics Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("### 📂 Input Data")
        st.markdown("Pastikan kolom terdapat 'text, product_name, category dan sold'")
        st.markdown(f"<div class='caption-text'>Unggah dataset ulasan berformat CSV. Analisis menggunakan Threshold Negatif: <b>{prediction_threshold}</b>.</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_cols = ['text', 'product_name', 'category', 'sold']
        
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Kehilangan kolom wajib. Terdeteksi: {list(df.columns)}")
        else:
            if st.button("🚀 Execute Dashboard Rendering", type="primary"):
                with st.status("⚙️ Mengeksekusi Machine Learning Pipeline...", expanded=True) as status:
                    st.write("1️⃣ Membersihkan anomali Metadata...")
                    df['product_name'] = df['product_name'].apply(clean_meta_data)
                    df['category'] = df['category'].apply(clean_meta_data)
                    df['sold_numeric'] = df['sold'].apply(parse_sold_to_numeric)
                    
                    st.write("2️⃣ Ekstraksi Linguistik (Sastrawi, Regex)...")
                    df['clean_text'] = df['text'].apply(preprocess_text)
                    
                    st.write("3️⃣ Memprediksi klasifikasi sentimen dengan Predict_Proba...")
                    X_features = vectorizer.transform(df['clean_text'])
                    
                    # === LOGIKA BATCH PREDICT_PROBA ===
                    probabilities = model_stacking.predict_proba(X_features)
                    prob_negative_array = probabilities[:, 1] # Ambil seluruh kolom probabilitas kelas 1
                    
                    # Gunakan threshold slider dari sidebar
                    df['sentiment_label'] = np.where(prob_negative_array >= prediction_threshold, 'Negative', 'Positive')
                    
                    st.session_state['processed_df'] = df
                    status.update(label="Analytics Engine Berhasil Dirender!", state="complete", expanded=False)
                st.rerun()

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        
        total_ulasan = len(df)
        total_negatif = len(df[df['sentiment_label'] == 'Negative'])
        persentase_negatif = (total_negatif / total_ulasan) * 100 if total_ulasan > 0 else 0
        total_produk_terjual = df['sold_numeric'].sum()

        st.markdown("<br>", unsafe_allow_html=True)

        # --- METRICS ---
        st.markdown("### 📈 Executive Metrics")
        st.markdown("<div class='caption-text'>Indikator performa utama berdasarkan agregasi data sentimen.</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card"><div class="metric-title">Total Reviews</div><div class="metric-value">{total_ulasan:,}</div></div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card alert"><div class="metric-title">Negative Cases</div><div class="metric-value">{total_negatif:,}</div></div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card warning"><div class="metric-title">Dissatisfaction %</div><div class="metric-value">{persentase_negatif:.1f}%</div></div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card"><div class="metric-title">Est. Units Sold</div><div class="metric-value">{total_produk_terjual:,}</div></div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- CATEGORY CHART ---
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Category Risk Distribution")
        st.markdown("<div class='caption-text'>Pemetaan departemen kategori yang menerima proporsi keluhan paling tinggi.</div>", unsafe_allow_html=True)
        
        cat_df = df.groupby('category').agg(
            total_review=('sentiment_label', 'count'),
            neg_count=('sentiment_label', lambda x: (x == 'Negative').sum())
        ).reset_index()
        cat_df['Persentase Negatif (%)'] = (cat_df['neg_count'] / cat_df['total_review']) * 100
        cat_df = cat_df.sort_values('Persentase Negatif (%)', ascending=False)
        
        if not cat_df.empty:
            fig = px.bar(
                cat_df, x='category', y='Persentase Negatif (%)',
                text=cat_df['Persentase Negatif (%)'].apply(lambda x: f'{x:.1f}%'),
                color='Persentase Negatif (%)', color_continuous_scale='Reds',
                labels={'category': 'Departemen Kategori', 'Persentase Negatif (%)': '% Keluhan'}
            )
            fig.update_traces(textposition='outside', textfont_color='white')
            fig.update_layout(
                template="plotly_dark",
                xaxis_tickangle=-45, 
                margin=dict(b=80, t=30, l=0, r=0),
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- LEADERBOARD ---
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Critical Risk Products")
        st.markdown("<div class='caption-text'>Urutan produk dengan skor risiko terekstrem (Kalkulasi: % Negatif × log(Total Review)).</div>", unsafe_allow_html=True)
        
        
        df_risk = get_risk_scores(df, min_reviews=20)
        
        if not df_risk.empty:
            st.dataframe(
                df_risk[['product_name', 'total_sold', 'total_review', 'pct_negative', 'risk_score']].head(10),
                column_config={
                    "product_name": st.column_config.TextColumn("Product Name", width="large"),
                    "total_sold": st.column_config.NumberColumn("Total Sold", format="%d Unit"),
                    "total_review": st.column_config.NumberColumn("Total Reviews"),
                    "pct_negative": st.column_config.NumberColumn("Negative %", format="%.2f"),
                    "risk_score": st.column_config.ProgressColumn(
                        "Severity Score", format="%.2f", 
                        min_value=0, max_value=float(df_risk['risk_score'].max())
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # --- WORDCLOUD ---
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("### ☁️ Root Cause Insights")
        st.markdown("<div class='caption-text'>Analisis WordCloud</div>", unsafe_allow_html=True)
        
        col_wc_settings, col_wc_plot = st.columns([1, 3])
        with col_wc_settings:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("**⚙️ N-Gram Filter:**")
            ngram_choice = st.radio("Level Kedalaman:", [1, 2, 3], format_func=lambda x: "Unigram (1 Kata)" if x==1 else ("Bigram (2 Kata)" if x==2 else "Trigram (3 Kata)"), label_visibility="collapsed")
        
        with col_wc_plot:
            fig_wc = plot_ngram_wordcloud(df, ngram_level=ngram_choice)
            if fig_wc:
                st.pyplot(fig_wc)
        st.markdown('</div>', unsafe_allow_html=True)