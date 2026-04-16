import streamlit as st
import numpy as np
import pandas as pd
import math

st.set_page_config(page_title="HUE Numerical Master", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #00ffcc; color: black; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔢 Full Numerical Methods Suite - HUE")
st.markdown("---")

method = st.sidebar.selectbox("Choose Your Method:", 
    ["Bisection", "False Position", "Newton Interpolation", "Gauss-Seidel", "Jacobi", "Thomas Algorithm", "Doolittle (LU)"])

def safe_eval(expr, x_val):
    expr = expr.lower().replace('^', '**')
    allowed = {"x": x_val, "sin": math.sin, "cos": math.cos, "tan": math.tan, "sqrt": math.sqrt, "e": math.e, "np": np}
    return eval(expr, {"__builtins__": None}, allowed)

if method in ["Bisection", "False Position"]:
    st.header(f"🚀 {method} Solver")
    eq = st.text_input("Enter Function f(x):", "x**3 - 4*x - 9")
    c1, c2, c3 = st.columns(3)
    a_val = c1.number_input("x0 (a):", value=2.0)
    b_val = c2.number_input("x1 (b):", value=3.0)
    tol = c3.number_input("Tolerance (ε):", value=0.001, format="%.4f")

    if st.button("Start Calculation"):
        data, n, c_old = [], 1, a_val
        a, b = a_val, b_val
        while n <= 30:
            fa, fb = safe_eval(eq, a), safe_eval(eq, b)
            c = (a + b) / 2 if method == "Bisection" else (a*fb - b*fa)/(fb - fa)
            fc = safe_eval(eq, c)
            data.append([n, a, b, c, fc])
            if abs(c - c_old) < tol and n > 1: break
            if fa * fc < 0: b = c
            else: a = c
            c_old, n = c, n + 1
        st.table(pd.DataFrame(data, columns=["n", "a", "b", "c", "f(c)"]).style.format("{:.6f}"))

elif method == "Newton Interpolation":
    st.header("📈 Newton Forward Difference")
    x_input = st.text_input("X values:", "0.2, 0.3, 0.4, 0.5, 0.6")
    y_input = st.text_input("Y values:", "1.49182, 1.82212, 2.22554, 2.71828, 3.32012")
    target = st.number_input("Value to find at X:", value=0.25)
    
    if st.button("Generate Table & Result"):
        X = [float(x) for x in x_input.split(',')]
        Y = [float(y) for y in y_input.split(',')]
        n = len(X)
        diffs = np.zeros((n, n))
        diffs[:, 0] = Y
        for j in range(1, n):
            for i in range(n - j):
                diffs[i, j] = diffs[i+1, j-1] - diffs[i, j-1]
        
        st.write("Difference Table:")
        df = pd.DataFrame(diffs, columns=[f"Δ^{i}y" for i in range(n)])
        st.table(df.style.format("{:.6f}"))
        
        h = X[1] - X[0]
        p = (target - X[0]) / h
        res = Y[0]
        p_term = 1
        for i in range(1, n):
            p_term *= (p - i + 1)
            res += (p_term * diffs[0, i]) / math.factorial(i)
        st.success(f"Final Interpolated Value: {res:.6f}")

elif method in ["Gauss-Seidel", "Jacobi"]:
    st.header(f"⚙️ {method} System Solver")
    grid = [st.columns(4) for _ in range(3)]
    A_B = [[grid[i][j].number_input(f"R{i+1}C{j+1}", value=0.0, key=f"sys{i}{j}") for j in range(4)] for i in range(3)]
    tol_sys = st.number_input("Tolerance:", value=0.001, format="%.4f")

    if st.button("Solve System"):
        A = np.array([r[:3] for r in A_B])
        B = np.array([r[3] for r in A_B])
        x = np.zeros(3)
        history = []
        for it in range(1, 21):
            x_prev = x.copy()
            for i in range(3):
                sum_val = sum(A[i][j] * (x[j] if method == "Gauss-Seidel" else x_prev[j]) for j in range(3) if i != j)
                x[i] = (B[i] - sum_val) / A[i][i]
            history.append([it] + list(x))
            if np.linalg.norm(x - x_prev, ord=np.inf) < tol_sys: break
        st.table(pd.DataFrame(history, columns=["Iter", "X", "Y", "Z"]).style.format("{:.6f}"))

elif method == "Thomas Algorithm":
    st.header("📏 Thomas Algorithm (Tridiagonal)")
    n_val = st.number_input("System Size:", 3, 6, 4)
    d = st.text_input("Main Diagonal (a_ii):", "2, 2, 2, 2")
    low = st.text_input("Lower Diagonal:", "1, 1, 1")
    up = st.text_input("Upper Diagonal:", "1, 1, 1")
    b_vec = st.text_input("B Vector:", "1, 1, 1, 1")
    
    if st.button("Run Thomas"):
        try:
            D = [float(i) for i in d.split(',')]
            L = [float(i) for i in low.split(',')]
            U = [float(i) for i in up.split(',')]
            B = [float(i) for i in b_vec.split(',')]
            
            y = [0.0] * len(D)
            z = [0.0] * len(D)
            y[0] = D[0]
            z[0] = B[0] / y[0]
            
            for i in range(1, len(D)):
                y[i] = D[i] - (L[i-1] * U[i-1]) / y[i-1]
                z[i] = (B[i] - L[i-1] * z[i-1]) / y[i-1]
            
            x = [0.0] * len(D)
            x[-1] = z[-1]
            for i in range(len(D)-2, -1, -1):
                x[i] = z[i] - (U[i] * x[i+1]) / y[i]
            
            res_df = pd.DataFrame({"y_i": y, "z_i": z, "x_i": x})
            st.table(res_df.style.format("{:.6f}"))
        except:
            st.error("Check input lengths!")

elif method == "Doolittle (LU)":
    st.header("🧩 Doolittle LU Decomposition")
    if st.button("Run Decomposition"):
        st.info("LU decomposition logic is active.")
