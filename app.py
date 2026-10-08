import streamlit as st
import joblib
import numpy as np

# ==================== تنظیمات صفحه ====================
st.set_page_config(
    page_title="تشخیص اسپم SMS",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==================== CSS سفارشی ====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Vazirmatn', sans-serif;
        direction: rtl;
        text-align: right;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .result-box {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
        animation: fadeIn 0.5s ease-in;
    }

    .spam-box {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.3);
    }

    .ham-box {
        background: linear-gradient(135deg, #51cf66, #37b24d);
        color: white;
        box-shadow: 0 8px 20px rgba(81, 207, 102, 0.3);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 2px solid #eee;
        color: #888;
        font-size: 0.9rem;
        line-height: 2;
    }

    .footer .name {
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.05rem;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)


# ==================== بارگذاری مدل ====================
@st.cache_resource
def load_models():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError as e:
    st.error(f"❌ فایل مدل پیدا نشد: {e}")
    st.info("لطفاً مطمئن شوید فایل‌های `spam_model.pkl` و `vectorizer.pkl` در کنار `app.py` هستند.")
    st.stop()


# ==================== هدر ====================
st.markdown('<div class="main-title">📧 تشخیص پیامک اسپم</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">با هوش مصنوعی، متن پیامک خود را تحلیل کنید</div>', unsafe_allow_html=True)


# ==================== سایدبار ====================
with st.sidebar:
    st.header("ℹ️ دربارهٔ این اپ")
    st.write(
        "این اپلیکیشن با استفاده از **یادگیری ماشین** و **پردازش زبان طبیعی**، "
        "پیامک‌های اسپم را از پیام‌های عادی تشخیص می‌دهد."
    )
    st.divider()
    st.subheader("🎯 نمونه‌های آماده")
    st.write("برای تست سریع، روی یکی از دکمه‌ها کلیک کن:")

    if st.button("📢 نمونه اسپم", use_container_width=True):
        st.session_state['sample'] = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!"

    if st.button("💬 نمونه عادی", use_container_width=True):
        st.session_state['sample'] = "سلام، جلسه فردا ساعت ۱۰ صبح برگزار می‌شود. لطفاً حضور داشته باشید."

    st.divider()
    st.caption("ساخته شده با ❤️ با Streamlit")


# ==================== ورودی کاربر ====================
default_text = st.session_state.get('sample', '')

user_input = st.text_area(
    "✍️ متن پیامک خود را وارد کنید:",
    value=default_text,
    height=150,
    placeholder="مثلاً: Congratulations! You've won a free prize..."
)

col1, col2 = st.columns([3, 1])
with col1:
    check = st.button("🔍 بررسی کن", type="primary", use_container_width=True)
with col2:
    clear = st.button("🗑️ پاک کن", use_container_width=True)

if clear:
    st.session_state['sample'] = ''
    st.rerun()


# ==================== پیش‌بینی ====================
if check:
    if user_input.strip() == "":
        st.warning("⚠️ لطفاً یک متن وارد کنید.")
    else:
        with st.spinner("⏳ در حال تحلیل پیامک..."):
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]

            # درصد اطمینان (اگر مدل پشتیبانی کند)
            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_vec)[0]
                confidence = proba[int(prediction)] * 100

        st.divider()
        st.subheader("📊 نتیجهٔ تحلیل")

        if prediction == 1:
            st.markdown(
                '<div class="result-box spam-box">🚨 این پیامک <b>اسپم</b> است!</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box ham-box">✅ این پیامک <b>غیراسپم (Ham)</b> است.</div>',
                unsafe_allow_html=True
            )

        if confidence is not None:
            st.write("")
            st.write(f"**🎯 میزان اطمینان مدل:** {confidence:.2f}%")
            st.progress(int(confidence))

        # جزئیات بیشتر
        with st.expander("🔬 جزئیات بیشتر"):
            st.write(f"**طول متن:** {len(user_input)} کاراکتر")
            st.write(f"**تعداد کلمات:** {len(user_input.split())}")
            st.write(f"**پیش‌بینی خام مدل:** {prediction}")
            if confidence is not None:
                st.write(f"**درصد اطمینان:** {confidence:.2f}%")

        # امکان دانلود نتیجه
        result_text = (
            f"متن ورودی:\n{user_input}\n\n"
            f"نتیجه: {'اسپم 🚨' if prediction == 1 else 'غیراسپم ✅'}\n"
            f"اطمینان: {confidence:.2f}%\n" if confidence else ""
        )
        st.download_button(
            "💾 دانلود نتیجه",
            data=result_text,
            file_name="spam_result.txt",
            mime="text/plain"
        )


# ==================== فوتر ====================
st.markdown("""
    <div class="footer">
        ساخته شده توسط<br>
        <span class="name">a_srt343</span><br>
        <span class="name">project_srt343</span>
    </div>
""", unsafe_allow_html=True)