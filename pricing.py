# pages/0_Pricing.py
import streamlit as st
import stripe

st.set_page_config(page_title="FCPA Sentinel AI – Pricing", page_icon="shield")

# OWNER BUTTON — ALWAYS VISIBLE FOR YOU
if st.button("Owner Login – Moly Malibu Access", type="secondary"):
    st.session_state.paid = True
    st.rerun()

# Stripe success
if st.query_params.get("subscribed") == "true":
    st.session_state.paid = True
    st.query_params.clear()

# If already paid → go to dashboard
if st.session_state.get("paid", False):
    st.switch_page("pages/1_Dashboard.py")

# ==================== PRICING PAGE (only unpaid users see this) ====================
st.markdown("<h1 style='text-align:center; font-size:4.5rem; background:linear-gradient(90deg,#06b6d4,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>FCPA Sentinel AI</h1>", unsafe_allow_html=True)
st.image("pricing.jpg", width=520, use_column_width=True)

st.markdown("<h2 style='text-align:center; color:#e2e8f0;'>Enterprise FCPA Forensic Intelligence</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Starter → $49/month")
with col2:
    st.markdown("### Enterprise → $399/month")

st.markdown("<br><br><br>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>Get Full Access</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    email = st.text_input("Corporate email", placeholder="you@company.com")
    if st.button("Purchase Subscription", type="primary", use_container_width=True):
        st.info("Payment system coming soon – contact monica@fcpa.ai for early access")

st.stop()

