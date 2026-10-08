import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==================== رفع مشکل BitLocker ====================
# مسیر nltk_data رو به درایو D منتقل می‌کنیم تا به درایو قفل‌شده E کاری نداشته باشه
NLTK_DIR = r"D:\nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)

# مسیرهای مربوط به درایو E رو حذف کن
nltk.data.path = [p for p in nltk.data.path if not str(p).upper().startswith("E:")]
# مسیر امن D رو اضافه کن
nltk.data.path.insert(0, NLTK_DIR)

nltk.download('stopwords', download_dir=NLTK_DIR, quiet=True)
nltk.download('wordnet', download_dir=NLTK_DIR, quiet=True)
nltk.download('omw-1.4', download_dir=NLTK_DIR, quiet=True)

# ==================== بارگذاری مدل ====================
@st.cache_resource
def load_models():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
    st.info("Make sure `spam_model.pkl` and `vectorizer.pkl` are in the same folder as `app.py`.")
    st.stop()

# ==================== Preprocessing ====================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ==================== UI ====================
st.set_page_config(page_title="SMS Spam Detector", page_icon="📧", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Vazirmatn', sans-serif;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 2px solid #eee;
        color: #888;
        font-size: 0.95rem;
    }
    .footer .name {
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📧 SMS Spam Detector (Advanced)</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#666;'>This app uses <b>SVM + TF-IDF + SMOTE</b> "
    "to detect spam messages with high accuracy.</p>",
    unsafe_allow_html=True
)

user_input = st.text_area("✍️ Enter your SMS message:", height=150,
                          placeholder="e.g. Congratulations! You've won a $1000 gift card. Click here...")

col1, col2 = st.columns([3, 1])
with col1:
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col2:
    clear = st.button("🗑️ Clear", use_container_width=True)

if clear:
    st.rerun()

if analyze:
    if not user_input.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        with st.spinner("⏳ Analyzing..."):
            cleaned = preprocess(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]

            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                confidence = proba[int(pred)] * 100

        if pred == 1:
            st.error("🚨 **SPAM DETECTED**")
        else:
            st.success("✅ **NOT SPAM (Ham)**")

        if confidence is not None:
            st.write(f"**🎯 Confidence:** {confidence:.2f}%")
            st.progress(int(confidence))

        with st.expander("🔬 See details"):
            st.write("**Original:**", user_input)
            st.write("**Cleaned:**", cleaned)
            st.write("**Prediction (raw):**", pred)
            if confidence is not None:
                st.write(f"**Confidence:** {confidence:.2f}%")

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
    <div class="footer">
        Built by <span class="name">A_srt343</span><br>
        Instagram: <span class="name">@project_srt343</span>
    </div>
""", unsafe_allow_html=True)
