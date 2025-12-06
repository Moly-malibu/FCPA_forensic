# app.py  
import streamlit as st

#loging
st.switch_page("pages/1_Login.py")
st.switch_page("pages/0_Pricing.py")


# if st.query_params.get("owner") == "monica2025":
#     st.session_state.paid = True

if st.query_params.get("owner") == "monica2025":
    st.session_state.demo_uses = 0  
    st.switch_page("pages/Demo.py")

# Si ya pagó → full dashboard
if st.session_state.get("paid", False):
    st.switch_page("pages/1_Dashboard.py")