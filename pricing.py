# pricing.py - Sistema de precios y pagos independiente
import streamlit as st
import stripe
import os

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://static.vecteezy.com/system/resources/previews/001/984/361/non_2x/abstract-modern-pattern-background-white-and-grey-geometric-texture-vector-art-illustration.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)


def init_stripe():
    """Inicializa Stripe con secrets"""
    stripe.api_key = st.secrets.get("STRIPE_SECRET_KEY", "")
    PUBLISHABLE_KEY = st.secrets["STRIPE_PUBLISHABLE_KEY"]
    return bool(stripe.api_key)

    if not stripe.api_key:
        st.error("❌ Stripe API key no configurada. Revisa Secrets en Streamlit Cloud.")
    st.stop()

def show_pricing_screen():
    """Muestra landing de precios y retorna True si pagado"""
    if not init_stripe():
        st.error("⚠️ Stripe configuration missing. Contact admin.")
        st.stop()
    
    PRICE_ID = st.secrets.get("PRICE_ID", "")
    DOMAIN = st.secrets.get("DOMAIN", "https://moly-malibu-fcpa-forensic-dashboard-dp1oob.streamlit.app/Pricing")
    
    st.image("pricing.jpg", width=200)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <style>
        .pricing-card {background: rgba(255,255,255,0.1); padding: 3rem; border-radius: 20px; 
                       backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2); 
                       text-align: center; max-width: 600px;}
        .price-highlight {font-size: 3.5rem; font-weight: 900; color: #10b981;}
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="pricing-card">', unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 4rem; font-weight: 900; background: linear-gradient(90deg, #06b6d4, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>FCPA Forensic & Contract Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #94a3b8;'>Enterprise FCPA Forensic Intelligence</h2>", unsafe_allow_html=True)
        st.markdown("**Detect hidden bribes, evasion patterns, and FCPA risks in seconds — 100% automated**")
        
        st.markdown("---")
        col_plan1, col_plan2 = st.columns(2)
        
        with col_plan1:
            st.markdown("### **Starter**")
            st.markdown("<span class='price-highlight'>$49<span style='font-size: 1.5rem;'>/mo</span></span>", unsafe_allow_html=True)
            st.markdown("- Unlimited document analysis<br>- PDF/CSV reports<br>- Basic risk scoring<br>- Email support")
        
        with col_plan2:
            st.markdown("### **Enterprise**")
            st.markdown("<span class='price-highlight'>$399<span style='font-size: 1.5rem;'>/mo</span></span>", unsafe_allow_html=True)
            st.markdown("- Multi-user (10+ seats)<br>- API access<br>- On-premise option<br>- Priority 24/7 support<br>- Custom compliance rules")
        
        st.markdown("---")
        
        email = st.text_input("📧 Professional Email", placeholder="compliance@company.com")
        if st.button("🚀 Subscribe Now - Launch Pricing", type="primary", use_container_width=True, help="Limited time offer"):
            if email:
                try:
                    checkout_session = stripe.checkout.sessions.create(
                        payment_method_types=['card'],
                        line_items=[{
                            'price': PRICE_ID, 
                            'quantity': 1
                        }],
                        mode='subscription',
                        customer_email=email,
                        success_url=DOMAIN + "?session_id={CHECKOUT_SESSION_ID}&subscribed=true",
                        cancel_url=DOMAIN + "?cancelled=true",
                        metadata={'plan': 'FCPA Forensic & Contract Analyzerr'}
                    )
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={checkout_session.url}">', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Payment error: {str(e)}")
            else:
                st.warning("⚠️ Please enter your professional email")
        
        st.markdown("**Trusted by:** KPMG • Deloitte • Baker McKenzie • Global Banks")
        st.markdown('</div>', unsafe_allow_html=True)
    
    return False  # No pagado aún

def check_active_subscription(email=None):
    """Verifica suscripción activa (opcional)"""
    try:
        if email:
            customers = stripe.Customer.list(email=email, limit=1)
            if customers.data:
                subs = stripe.Subscription.list(customer=customers.data[0].id, status='active')
                return len(subs.data) > 0
        return True  # Demo: siempre activo
    except:
        return True  # Fallback para demo
