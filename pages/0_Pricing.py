# pages/0_Pricing.py

# pages/0_Pricing.py
import streamlit as st
import stripe

# ===================== INITIAL SETUP =====================

PRICE_ID = st.secrets["PRICE_ID"]

st.set_page_config(
    page_title="FCPA Sentinel AI – Pricing",
    page_icon="shield",
    layout="centered"
)
st.sidebar.image("logo.jpg", width=200)
# Initialize Stripe (only once)
if "stripe_ready" not in st.session_state:
    stripe.api_key = st.secrets.get("STRIPE_SECRET_KEY")
    if not stripe.api_key:
        st.error("Stripe API key not configured. Check secrets.toml")
        st.stop()
    st.session_state.stripe_ready = True

PRICE_ID = st.secrets["PRICE_ID"]
DOMAIN = st.secrets["DOMAIN"]  # e.g. "http://localhost:8501" or your production URL

# ===================== CHECK IF USER ALREADY PAID =====================
def user_has_paid() -> bool:
    if st.query_params.get("subscribed") == "true":
        st.session_state.paid = True
        st.query_params.clear()
        return True
    return st.session_state.get("paid", False)

if user_has_paid():
    st.switch_page("Dashboard.py")

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    .price-highlight {font-size: 3.8rem; font-weight: bold; color: #06b6d4;}
    .big-button {height: 70px; font-size: 1.5rem !important; font-weight: bold;}
    .stButton > button {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# ===================== HERO IMAGE (centered) =====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        st.image("pricing.jpg", width=520)  # Fixed: no more use_column_width
    except:
        # Fallback image if local file is missing
        st.image("https://i.imgur.com/9mXvG4Q.jpg", width=520)

st.markdown("<br>", unsafe_allow_html=True)

# ===================== MAIN TITLE =====================
st.markdown(
    "<h1 style='text-align: center; font-size: 4.2rem; background: linear-gradient(90deg, #06b6d4, #3b82f6); "
    "-webkit-background-clip: text; -webkit-text-fill-color: transparent;'>FCPA Sentinel AI</h1>",
    unsafe_allow_html=True
)
st.markdown("<h2 style='text-align: center; color: #e2e8f0;'>Enterprise FCPA Forensic Intelligence</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.3rem; color: #94a3b8;'>"
            "Detect hidden bribes, evasion patterns, and FCPA risks in seconds — 100% AI-powered</p>",
            unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ===================== PRICING CARDS =====================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div style="background:#1e293b; padding:35px; border-radius:16px; border:2px solid #06b6d4; text-align:center;">
        <h3>Starter</h3>
        <h2><span class="price-highlight">$49</span><small style="color:#94a3b8;">/month</small></h2>
        <ul style="text-align:left; color:#cbd5e1; line-height:2;">
            <li>Unlimited document analysis</li>
            <li>PDF & Excel reports</li>
            <li>Basic risk scoring</li>
            <li>Email support</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#1e293b; padding:35px; border-radius:16px; border:3px solid #8b5cf6; text-align:center;">
        <h3 style="color:#e9d5ff;">Enterprise <small style="background:#8b5cf6; color:white; padding:4px 12px; border-radius:8px;">Most Popular</small></h3>
        <h2><span class="price-highlight">$399</span><small style="color:#94a3b8;">/month</small></h2>
        <ul style="text-align:left; color:#cbd5e1; line-height:2;">
            <li>Everything in Starter</li>
            <li>Multi-user (10+ seats)</li>
            <li>Full API access</li>
            <li>On-premise / VPC deployment</li>
            <li>Custom compliance rules</li>
            <li>24/7 priority support</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br><br><br>", unsafe_allow_html=True)

# ===================== FINAL CTA – BIG BUTTON =====================
st.markdown("<h2 style='text-align: center; color: #f1f5f9;'>Start Protecting Your Company Today</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.4rem; color: #94a3b8;'>"
            "<strong>14-day free trial</strong> • No credit card required • Cancel anytime</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2.2, 1])
with col2:
    email = st.text_input(
        "Corporate email",
        placeholder="compliance@yourcompany.com",
        key="final_email",
        label_visibility="collapsed"
    )

    if st.button(
        "START 14-DAY FREE TRIAL",
        type="primary",
        use_container_width=True,
        key="final_subscribe"
    ):
        if email and "@" in email and "." in email.split("@")[-1]:
            with st.spinner("Creating your secure account..."):
                try:
                    session = stripe.checkout.Session.create(   # Correct syntax
                        payment_method_types=["card"],
                        line_items=[{"price": PRICE_ID, "quantity": 1}],
                        mode="subscription",
                        customer_email=email,
                        subscription_data={"trial_period_days": 14},
                        success_url=f"{DOMAIN}/?subscribed=true",
                        cancel_url=f"{DOMAIN}/pages/0_Pricing.py",
                        metadata={"plan": "fcpa_sentinel"}
                    )
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={session.url}">', unsafe_allow_html=True)
                    st.success("Redirecting to secure checkout...")
                except Exception as e:
                    st.error(f"Payment error: {e}")
        else:
            st.warning("Please enter a valid corporate email")
st.stop()

# Botón secreto solo para ti (nadie lo ve)
if st.button("👑 Owner Access", help="Solo para Monica"):
    st.session_state.paid = True
    st.switch_page("Dashboard.py")
# ===================== FOOTER =====================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("**Trusted by compliance teams at:** KPMG • Deloitte • PwC • EY • Baker McKenzie • Global Banks", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569;'>© 2025 FCPA Sentinel AI – All rights reserved</p>", unsafe_allow_html=True)

