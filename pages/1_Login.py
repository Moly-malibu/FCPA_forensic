# pages/0_Login.py
import streamlit as st
import pyotp
import qrcode
import base64
from io import BytesIO
import time
import json
import os
import base64

# ====================== background ======================

# def set_background(image_file):
#     """
#     Función para inyectar CSS personalizado con la imagen de fondo codificada en Base64.
#     """
#     with open(image_file, "rb") as f:
#         img_data = f.read()
#     encoded_image = base64.b64encode(img_data).decode()
    
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("data:image/jpg;base64,{encoded_image}");
#             background-size: cover; /* Cubre toda la pantalla */
#             background-repeat: no-repeat;
#             background-attachment: fixed; /* Fija el fondo al hacer scroll */
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
# set_background('assets/backg4.jpg')


# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(page_title="FCPA Sentinel AI - Login", page_icon="lock", layout="centered")

# === ESTILOS ===
st.markdown("""
<style>
    .title { font-size: 42px !important; font-weight: 700; text-align: center; color: #1E3A8A; }
    .subtitle { text-align: center; color: #475569; margin-bottom: 40px; }
    .stButton>button { height: 54px; font-size: 18px; }
    .big-button button { height: 60px; font-size: 20px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# === save ===
SECRET_FILE = "user_secret.json"

def load_secret():
    if os.path.exists(SECRET_FILE):
        with open(SECRET_FILE, "r") as f:
            data = json.load(f)
            st.session_state.totp_secret = data["secret"]
            st.session_state.pending_email = data["email"]

def save_secret(email, secret):
    with open(SECRET_FILE, "w") as f:
        json.dump({"email": email, "secret": secret}, f)

if "totp_secret" not in st.session_state:
    load_secret()

if st.session_state.get("authenticated", False):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.success(f"Welcome back, **{st.session_state.user}**!")
        st.markdown("### You are already logged in")
        if st.button("Logout", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            if os.path.exists(SECRET_FILE):
                os.remove(SECRET_FILE)
            st.success("Logged out")
            time.sleep(1)
            st.rerun()
    st.stop()

# ===  QR ===
def make_qr(secret: str, email: str) -> str:
    uri = pyotp.TOTP(secret).provisioning_uri(name=email, issuer_name="FCPA Sentinel AI")
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# === PANTALLA PRINCIPAL ===
col1, col2, col3 = st.columns([1,3,1])
with col2:
    # ← LOGO CENTRADO
    st.markdown(
        """
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <img src="https://assets.grok.com/users/87c3ab6d-d97f-4805-bf0c-dd8c61e0e5a4/generated/d1d5b03e-ebe5-42ac-9fe7-b91f5286e408/image.jpg" width="220">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <h1 style="
            text-align: center;
            color: #0052CC;
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 8px;
            letter-spacing: -1px;
        ">FCPA Sentinel AI</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style="
            text-align: center;
            color: #475569;
            font-size: 18px;
            font-weight: 500;
            margin-top: 0;
        ">Enterprise Forensic Intelligence • 100% Private • GDPR Compliant</p>
        """,
        unsafe_allow_html=True
    )
    tab1, tab2 = st.tabs(["Login", "First Time Setup"])

    # === PESTAÑA LOGIN ===
    with tab1:
        st.markdown("#### Enter your credentials")
        email = st.text_input("Email", placeholder="you@company.com", key="login_mail")
        code = st.text_input("2FA Code (6 digits)", type="password", max_chars=6, key="code")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Login", type="primary", use_container_width=True):
                if not email or not code:
                    st.error("Fill in all fields")
                elif not st.session_state.get("totp_secret"):
                    st.error("No 2FA configured. Go to 'First Time Setup' first.")
                else:
                    totp = pyotp.TOTP(st.session_state.totp_secret)
                    if code == "123456" or totp.verify(code, valid_window=1):
                        st.session_state.authenticated = True
                        st.session_state.user = email
                        st.success("Access granted!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid or expired code")

        with col_b:
            st.button("Cancel", use_container_width=True)

    # ===  ===
    with tab2:
        st.warning("Use only the first time or to reset 2FA")
        new_email = st.text_input("Your email", placeholder="you@company.com", key="new_mail")

        if st.button("Generate 2FA QR Code", type="primary", use_container_width=True):
            if "@" not in new_email:
                st.error("Enter a valid email")
            else:
                with st.spinner("Generating..."):
                    secret = pyotp.random_base32()
                    st.session_state.totp_secret = secret
                    st.session_state.pending_email = new_email
                    save_secret(new_email, secret)   
                    qr_b64 = make_qr(secret, new_email)

                st.success("2FA Ready!")
                st.image(f"data:image/png;base64,{qr_b64}", width=220)
                st.code(secret, language=None)
                st.info("Scan with Google Authenticator, Authy, etc.")
                st.success(f"Done {new_email}! Now go to Login and enter the 6-digit code (or type 123456 for demo)")

    # === BOTÓN SALIR ===
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Exit Application", type="secondary"):
        st.warning("Closing FCPA Sentinel AI...")
        time.sleep(2)
        st.stop()

st.stop()