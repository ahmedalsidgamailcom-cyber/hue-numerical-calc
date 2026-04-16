import streamlit as st
import numpy as np
import pandas as pd
import math

st.set_page_config(page_title="HUE Numerical Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; background-color: #00ffcc; color: black; font-weight: bold; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔢 Numerical Methods Suite - HUE Edition")
st.subheader("Final Project - Student: Ahmed Alsayed Al-Iraqi")

method = st.sidebar.selectbox("Select Method:", 
    ["Bisection", "False Position", "Newton Interpolation", "Gauss-Seidel", "Jacobi", "Thomas Algorithm", "Doolittle (LU)"])

def safe_eval(expr, x_val):
    expr = expr.lower().replace('^', '**')
    allowed = {"x": x_val, "sin": math.sin, "cos": math.cos, "tan": math.tan, "sqrt": math.sqrt, "e": math.e, "np": np}
    return eval(expr, {"__builtins__": None}, allowed)

if method in ["Bisection", "False Position"]:
    st.header(f"Method: {method}")
    eq = st.text_input("f(x):", "x**3 - 4*x - 9")
    col1, col2, col3 = st.columns(3)
    a_in = col1.number_input("x0 (a):", value=2.0)
    b_in = col2.number_input("x1 (b):", value=3.0)
    eps = col3.number_input("ε (Tolerance):", value=0.001, format="%.4f")
    if st.button("Calculate"):
        data, n, c_old = [], 1, a_in
        a, b = a_in, b_in
        while n <= 25:
            fa, fb = safe_eval(eq, a), safe_eval(eq, b)
            c = (a + b) / 2 if method == "Bisection" else (a*fb - b*fa)/(fb - fa)
            fc = safe_eval(eq, c)
            data.append([n, a, b, c, fc])
            if abs(c - c_old) < eps and n > 1: break
            if fa * fc < 0: b = c
            else: a = c
            c_old, n = c, n + 1
        st.table(pd.DataFrame(data, columns=["n", "a", "b", "c", "f(c)"]).style.format("{:.6f}"))

elif method == "Newton Interpolation":
    st.header("Newton Forward Interpolation")
    x_pts = st.text_input("X values:", "0.2, 0.3, 0.4, 0.5, 0.6")
    y_pts = st.text_input("Y values:", "1.49182, 1.82212, 2.22554, 2.71828, 3.32012")
    target = st.number_input("Target X:", value=0.25)
    if st.button("Interpolate"):
        X = [float(x) for x in x_pts.split(',')]
        Y = [float(y) for y in y_input.split(',')] if 'y_input' in locals() else [float(y) for y in y_pts.split(',')]
        n = len(X); diffs = np.zeros((n, n)); diffs[:, 0] = Y
        for j in range(1, n):
            for i in range(n - j): diffs[i, j] = diffs[i+1, j-1] - diffs[i, j-1]
        st.write("Difference Table:")
        st.table(pd.DataFrame(diffs).style.format("{:.6f}"))
        h = X[1] - X[0]; p = (target - X[0]) / h; res = Y[0]; p_term = 1
        for i in range(1, n):
            p_term *= (p - i + 1)
            res += (p_term * diffs[0, i]) / math.factorial(i)
        st.success(f"Result at f({target}) = {res:.6f}")

elif method in ["Gauss-Seidel", "Jacobi"]:
    st.header(f"System Solver: {method}")
    # المثال الافتراضي (Diagonal Dominant)
    defaults = [[10, -1, 2, 6], [-1, 11, -1, 25], [2, -1, 10, -11]]
    grid = [st.columns(4) for _ in range(3)]
    mat = [[grid[i][j].number_input(f"R{i+1}C{j+1}", value=float(defaults[i][j]), key=f"s{i}{j}") for j in range(4)] for i in range(3)]
    tol = st.number_input("ε:", value=0.001, format="%.4f")
    if st.button("Solve"):
        A = np.array([r[:3] for r in mat]); B = np.array([r[3] for r in mat])
        x = np.zeros(3); history = []
        for i in range(1, 21):
            xo = x.copy()
            for j in range(3):
                s = sum(A[j][k] * (x[k] if method == "Gauss-Seidel" else xo[k]) for k in range(3) if j != k)
                x[j] = (B[j] - s) / A[j,j]
            history.append([i] + list(x))
            if np.linalg.norm(x - xo, ord=np.inf) < tol: break
        st.table(pd.DataFrame(history, columns=["Iter", "X", "Y", "Z"]).style.format("{:.6f}"))

elif method == "Thomas Algorithm":
    st.header("Thomas (Tridiagonal Matrix)")
    d = st.text_input("Main Diag (a_ii):", "2.0, 2.0, 2.0, 2.0")
    l = st.text_input("Lower Diag:", "1.0, 1.0, 1.0")
    u = st.text_input("Upper Diag:", "1.0, 1.0, 1.0")
    b = st.text_input("B Vector:", "1.0, 1.0, 1.0, 1.0")
    if st.button("Run Thomas"):
        D, L, U, B = [float(i) for i in d.split(',')], [float(i) for i in l.split(',')], [float(i) for i in u.split(',')], [float(i) for i in b.split(',')]
        y, z = [0.0]*len(D), [0.0]*len(D)
        y[0] = D[0]; z[0] = B[0]/y[0]
        for i in range(1, len(D)):
            y[i] = D[i] - (L[i-1]*U[i-1])/y[i-1]
            z[i] = (B[i] - L[i-1]*z[i-1])/y[i-1]
        x = [0.0]*len(D); x[-1] = z[-1]
        for i in range(len(D)-2, -1, -1): x[i] = z[i] - (U[i]*x[i+1])/y[i]
        st.table(pd.DataFrame({"y_i": y, "z_i": z, "x_i": x}).style.format("{:.6f}"))

elif method == "Doolittle (LU)":
    st.header("Doolittle LU Decomposition (3x3)")
    defaults = [[3, 2, 1, 10], [2, 3, 2, 14], [1, 2, 3, 14]]
    grid = [st.columns(4) for _ in range(3)]
    mat = [[grid[i][j].number_input(f"R{i+1}C{j+1}", value=float(defaults[i][j]), key=f"d{i}{j}") for j in range(4)] for i in range(3)]
    if st.button("Solve"):
        A = np.array([r[:3] for r in mat]); B = np.array([r[3] for r in mat])
        L, U = np.eye(3), np.zeros((3, 3))
        for i in range(3):
            for k in range(i, 3): U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))
            for k in range(i + 1, 3): L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]
        st.write("L Matrix:"); st.table(pd.DataFrame(L).style.format("{:.6f}"))
        st.write("U Matrix:"); st.table(pd.DataFrame(U).style.format("{:.6f}"))
        y = np.zeros(3)
        for i in range(3): y[i] = B[i] - sum(L[i, j] * y[j] for j in range(i))
        x = np.zeros(3)
        for i in range(2, -1, -1): x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i+1, 3))) / U[i, i]
        st.success(f"Final Solution: X={x[0]:.4f}, Y={x[1]:.4f}, Z={x[2]:.4f}")
