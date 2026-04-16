import streamlit as st
import numpy as np
import math
import re

st.set_page_config(page_title="HUE Numerical Master", layout="wide")

st.title("🔢 Numerical Methods Master - HUE")
st.markdown("### Developed by: Ahmed Alsayed Al-Iraqi")

# Sidebar for Inputs
st.sidebar.header("Input Parameters")
eq_input = st.sidebar.text_input("Function f(x):", "x**3 - 4*x - 9")
a_val = st.sidebar.number_input("Start (a / x0):", value=2.0)
b_val = st.sidebar.number_input("End (b / x1):", value=3.0)
eps = st.sidebar.number_input("Tolerance (ε):", value=0.001, format="%.4f")

# Helper to Evaluate
def safe_eval(expr, x):
    expr = expr.lower().replace('^', '**')
    allowed = {"x": x, "sin": math.sin, "cos": math.cos, "tan": math.tan, "sqrt": math.sqrt, "e": math.e}
    return eval(expr, {"__builtins__": None}, allowed)

method = st.selectbox("Choose Method:", ["Bisection", "False Position", "Newton Interpolation", "Thomas Algorithm"])

if st.button("Calculate"):
    if method == "Bisection":
        a, b, n = a_val, b_val, 1
        c_old = a
        data = []
        while True:
            c = (a + b) / 2
            fc = safe_eval(eq_input, c)
            data.append([n, round(a, 6), round(b, 6), round(c, 6), round(fc, 6)])
            if abs(c - c_old) < eps and n > 1: break
            if safe_eval(eq_input, a) * fc < 0: b = c
            else: a = c
            c_old = c
            n += 1
        st.table(np.array(data, dtype=object))
        st.success(f"Final Root: {c:.6f}")

    elif method == "Thomas Algorithm":
        # المنطق المأخوذ من ورقة الشرح الخاصة بك
        st.info("Applying Thomas Algorithm (y1 = a11) as per HUE rules...")
        # (بقية الكود الخاص بـ Thomas و Doolittle يتم وضعه هنا بنفس المنطق)
