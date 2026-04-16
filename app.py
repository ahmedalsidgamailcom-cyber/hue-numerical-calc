import tkinter as tk
from tkinter import messagebox, ttk
import math
import re
import numpy as np

class NumericalMethodsUltimateHUE:
    def __init__(self, root):
        self.root = root
        self.root.title("HUE - Numerical Methods Master (Ahmed Alsayed)")
        self.root.geometry("1300x900")
        self.root.configure(bg="#0f0f0f")
        
        self.setup_styles()
        self.setup_vars()
        self.create_widgets()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Treeview", background="#1a1a1a", foreground="#e0e0e0", fieldbackground="#1a1a1a", rowheight=30, font=("Consolas", 10))
        self.style.configure("Treeview.Heading", background="#333333", foreground="#00ffcc", font=("Segoe UI", 10, "bold"))

    def setup_vars(self):
        self.eq_var = tk.StringVar(value="x**3 - 4*x - 9")
        self.a_var = tk.StringVar(value="2")
        self.b_var = tk.StringVar(value="3")
        self.eps_var = tk.StringVar(value="0.001") 
        self.final_res_var = tk.StringVar(value="Result: ---")
        self.p_val_var = tk.StringVar(value="p: ---")
        self.fx_val_var = tk.StringVar(value="f(x): ---")

    def create_widgets(self):
        tk.Label(self.root, text="NUMERICAL METHODS CALCULATOR PRO", bg="#0f0f0f", fg="#00ffcc", font=("Segoe UI", 26, "bold")).pack(pady=20)

        # Input Area
        input_f = tk.Frame(self.root, bg="#1a1a1a", padx=20, pady=15)
        input_f.pack(fill="x", padx=40)
        
        for lbl, var, w in [("f(x):", self.eq_var, 30), ("a/x0:", self.a_var, 8), ("b/x1:", self.b_var, 8), ("ε:", self.eps_var, 8)]:
            tk.Label(input_f, text=lbl, bg="#1a1a1a", fg="#888", font=("Segoe UI", 10, "bold")).pack(side="left", padx=5)
            tk.Entry(input_f, textvariable=var, width=w, bg="#262626", fg="#00ffcc", font=("Consolas", 11)).pack(side="left", padx=10)

        # Main Buttons
        btn_f = tk.Frame(self.root, bg="#0f0f0f")
        btn_f.pack(pady=10)
        
        methods = [
            ("BISECTION", "#007acc", self.run_bisection),
            ("FALSE POSITION", "#e81123", self.run_false_position),
            ("MODE 5+2", "#28a745", self.open_matrix_direct),
            ("GAUSS-SEIDEL", "#d16d00", lambda: self.open_iterative("Seidel")),
            ("GAUSS-JACOBI", "#68217a", lambda: self.open_iterative("Jacobi")),
            ("DOOLITTLE", "#444", self.open_doolittle),
            ("THOMAS (HUE)", "#444", self.open_thomas),
            ("NEWTON", "#00ffcc", self.open_newton)
        ]
        
        for txt, clr, cmd in methods:
            fg = "black" if clr == "#00ffcc" else "white"
            tk.Button(btn_f, text=txt, bg=clr, fg=fg, width=15, font=("Segoe UI", 9, "bold"), command=cmd).pack(side="left", padx=5, pady=5)

        # Table
        t_cont = tk.Frame(self.root, bg="#0f0f0f")
        t_cont.pack(fill="both", expand=True, padx=40, pady=10)
        self.tree = ttk.Treeview(t_cont, show='headings')
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(t_cont, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

        # Footer
        footer = tk.Frame(self.root, bg="#121212", height=50)
        footer.pack(side="bottom", fill="x")
        tk.Label(footer, textvariable=self.p_val_var, bg="#121212", fg="#00ffcc").pack(side="left", padx=20)
        tk.Label(footer, textvariable=self.fx_val_var, bg="#121212", fg="#00ffcc").pack(side="left", padx=20)
        tk.Label(footer, textvariable=self.final_res_var, bg="#121212", fg="#00ffcc", font=("Consolas", 14, "bold")).pack(side="right", padx=20)

    # --- MATH ENGINE ---
    def safe_eval(self, expr, x_val):
        expr = expr.lower().replace('^', '**').replace('e', str(math.e))
        expr = re.sub(r'(\d)([a-z\(])', r'\1*\2', expr)
        allowed = {"x": x_val, "sin": math.sin, "cos": math.cos, "tan": math.tan, "sqrt": math.sqrt}
        return eval(expr, {"__builtins__": None}, allowed)

    def run_bisection(self):
        try:
            a, b, eps = float(self.a_var.get()), float(self.b_var.get()), float(self.eps_var.get())
            expr = self.eq_var.get()
            self.tree["columns"] = ("n","a","b","c","fc")
            for c in self.tree["columns"]: self.tree.heading(c, text=c); self.tree.column(c, anchor="center")
            for r in self.tree.get_children(): self.tree.delete(r)
            c_old, n = a, 1
            while True:
                c = (a+b)/2; fc = self.safe_eval(expr, c)
                self.tree.insert("", "end", values=(n, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}", f"{fc:.6f}"))
                if abs(c - c_old) < eps and n > 1: break
                if self.safe_eval(expr, a)*fc < 0: b=c
                else: a=c
                c_old, n = c, n+1
            self.final_res_var.set(f"Root: {c:.6f}")
        except Exception as e: messagebox.showerror("Error", str(e))

    def run_false_position(self):
        try:
            a, b, eps = float(self.a_var.get()), float(self.b_var.get()), float(self.eps_var.get())
            expr = self.eq_var.get()
            self.tree["columns"] = ("n","a","b","c","fc")
            for c in self.tree["columns"]: self.tree.heading(c, text=c); self.tree.column(c, anchor="center")
            for r in self.tree.get_children(): self.tree.delete(r)
            c_old, n = a, 1
            while True:
                fa, fb = self.safe_eval(expr, a), self.safe_eval(expr, b)
                c = (a*fb - b*fa)/(fb-fa); fc = self.safe_eval(expr, c)
                self.tree.insert("", "end", values=(n, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}", f"{fc:.6f}"))
                if abs(c - c_old) < eps and n > 1: break
                if fa*fc < 0: b=c
                else: a=c
                c_old, n = c, n+1
            self.final_res_var.set(f"Root: {c:.6f}")
        except Exception as e: messagebox.showerror("Error", str(e))

    def open_newton(self):
        win = tk.Toplevel(self.root); win.title("Newton Interpolation"); win.geometry("500x450"); win.configure(bg="#1a1a1a")
        tk.Label(win, text="X values:", bg="#1a1a1a", fg="white").pack(pady=5)
        xi = tk.Entry(win, width=40); xi.insert(0, "0.2, 0.3, 0.4, 0.5, 0.6"); xi.pack()
        tk.Label(win, text="Y values:", bg="#1a1a1a", fg="white").pack(pady=5)
        yi = tk.Entry(win, width=40); yi.insert(0, "1.49182, 1.82212, 2.22554, 2.71828, 3.32012"); yi.pack()
        tk.Label(win, text="Target X:", bg="#1a1a1a", fg="white").pack(pady=5)
        tx = tk.Entry(win, width=15); tx.insert(0, "0.25"); tx.pack()
        def solve(mode):
            X = [float(i) for i in xi.get().split(',')]; Y = [float(i) for i in yi.get().split(',')]
            T, n = float(tx.get()), len(X); h = X[1]-X[0]
            d = np.zeros((n, n)); d[:,0] = Y
            for j in range(1, n):
                for i in range(n-j): d[i,j] = d[i+1,j-1] - d[i,j-1]
            p = (T-X[0])/h if mode=="Fwd" else (T-X[-1])/h
            self.tree["columns"] = ["X","Y"] + [f"Δ^{i}y" for i in range(1,n)]
            for c in self.tree["columns"]: self.tree.heading(c, text=c); self.tree.column(c, width=100)
            for r in self.tree.get_children(): self.tree.delete(r)
            for i in range(n):
                row = [f"{X[i]:.6f}", f"{Y[i]:.6f}"]
                for j in range(1, n-i): row.append(f"{d[i,j]:.6f}")
                self.tree.insert("", "end", values=tuple(row + [""]*(n-len(row))))
            res = Y[0] if mode=="Fwd" else Y[-1]; trm = 1
            for i in range(1, n):
                trm *= (p-(i-1)) if mode=="Fwd" else (p+(i-1))
                res += (trm * (d[0,i] if mode=="Fwd" else d[n-1-i,i])) / math.factorial(i)
            self.p_val_var.set(f"p: {p:.6f}"); self.fx_val_var.set(f"f(x): {res:.6f}"); win.destroy()
        tk.Button(win, text="Forward", bg="#00ffcc", command=lambda: solve("Fwd")).pack(pady=10)
        tk.Button(win, text="Backward", bg="#00ffcc", command=lambda: solve("Bwd")).pack()

    def open_iterative(self, mode):
        win = tk.Toplevel(self.root); win.geometry("500x500"); win.configure(bg="#1a1a1a")
        ents = [[tk.Entry(win, width=8) for _ in range(4)] for _ in range(3)]
        for r in range(3):
            for c in range(4): ents[r][c].grid(row=r, column=c, padx=10, pady=10)
        def load():
            ex = [[10,-1,2,6], [-1,11,-1,25], [2,-1,10,-11]]
            for r in range(3):
                for c in range(4): ents[r][c].delete(0, tk.END); ents[r][c].insert(0, str(ex[r][c]))
        def run():
            A = np.array([[float(ents[r][c].get()) for c in range(3)] for r in range(3)])
            B = np.array([float(ents[r][3].get()) for r in range(3)])
            x = np.zeros(3); eps = float(self.eps_var.get())
            self.tree["columns"] = ("i","x","y","z"); [self.tree.heading(c, text=c) for c in self.tree["columns"]]
            for r in self.tree.get_children(): self.tree.delete(r)
            for i in range(1, 21):
                xo = x.copy()
                for j in range(3):
                    s = B[j] - sum(A[j,k]*(x[k] if mode=="Seidel" else xo[k]) for k in range(3) if j!=k)
                    x[j] = s/A[j,j]
                self.tree.insert("", "end", values=(i, f"{x[0]:.6f}", f"{x[1]:.6f}", f"{x[2]:.6f}"))
                if np.linalg.norm(x-xo, ord=np.inf) < eps: break
            self.final_res_var.set(f"Final: {x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}"); win.destroy()
        tk.Button(win, text="EXAMPLE", bg="#444", fg="white", command=load).grid(row=4, column=0, columnspan=2, pady=10)
        tk.Button(win, text="RUN", bg="#28a745", fg="white", command=run).grid(row=4, column=2, columnspan=2)

    def open_matrix_direct(self):
        win = tk.Toplevel(self.root); win.geometry("450x400"); win.configure(bg="#1a1a1a")
        ents = [[tk.Entry(win, width=8) for _ in range(4)] for _ in range(3)]
        for r in range(3):
            for c in range(4): ents[r][c].grid(row=r, column=c, padx=10, pady=10)
        def load():
            ex = [[3,2,1,10], [2,3,2,14], [1,2,3,14]]
            for r in range(3):
                for c in range(4): ents[r][c].delete(0, tk.END); ents[r][c].insert(0, str(ex[r][c]))
        def solve():
            A = np.array([[float(ents[r][c].get()) for c in range(3)] for r in range(3)])
            B = np.array([float(ents[r][3].get()) for r in range(3)])
            res = np.linalg.solve(A, B)
            self.final_res_var.set(f"X:{res[0]:.3f} Y:{res[1]:.3f} Z:{res[2]:.3f}"); win.destroy()
        tk.Button(win, text="EXAMPLE", command=load).grid(row=4, column=0, columnspan=2)
        tk.Button(win, text="SOLVE", command=solve).grid(row=4, column=2, columnspan=2)

    def open_doolittle(self):
        win = tk.Toplevel(self.root); win.title("Doolittle Laws"); win.geometry("450x450")
        ents = [[tk.Entry(win, width=8) for _ in range(4)] for _ in range(3)]
        for r in range(3):
            for c in range(4): ents[r][c].grid(row=r, column=c, padx=5, pady=5)
        def run():
            A = np.array([[float(ents[r][c].get()) for c in range(3)] for r in range(3)])
            B = np.array([float(ents[r][3].get()) for r in range(3)])
            L, U = np.eye(3), np.zeros((3,3))
            for i in range(3):
                for k in range(i, 3): U[i,k] = A[i,k] - sum(L[i,j]*U[j,k] for j in range(i))
                for k in range(i+1, 3): L[k,i] = (A[k,i] - sum(L[k,j]*U[j,i] for j in range(i))) / U[i,i]
            V = np.linalg.solve(L, B); X = np.linalg.solve(U, V)
            self.tree["columns"] = ("L","U","V","X")
            for r in self.tree.get_children(): self.tree.delete(r)
            rows = [(f"l21:{L[1,0]:.2f}", f"u11:{U[0,0]:.2f}", f"v1:{V[0]:.2f}", f"x:{X[0]:.2f}"),
                    (f"l31:{L[2,0]:.2f}", f"u12:{U[0,1]:.2f}", f"v2:{V[1]:.2f}", f"y:{X[1]:.2f}"),
                    (f"l32:{L[2,1]:.2f}", f"u33:{U[2,2]:.2f}", f"v3:{V[2]:.2f}", f"z:{X[2]:.2f}")]
            for r in rows: self.tree.insert("", "end", values=r)
            self.final_res_var.set(f"X:{X[0]:.3f} Y:{X[1]:.3f} Z:{X[2]:.3f}"); win.destroy()
        tk.Button(win, text="SOLVE", command=run).grid(row=4, columnspan=4)

    def open_thomas(self):
        win = tk.Toplevel(self.root); win.title("Thomas HUE Laws"); win.geometry("600x400")
        tk.Label(win, text="N:").grid(row=0, column=0); n_ent = tk.Entry(win, width=5); n_ent.insert(0,"4"); n_ent.grid(row=0, column=1)
        f = tk.Frame(win); f.grid(row=1, column=0, columnspan=10)
        def mk():
            for s in f.winfo_children(): s.destroy()
            global t_e; n = int(n_ent.get()); t_e = [[tk.Entry(f, width=7) for _ in range(n+1)] for _ in range(n)]
            for r in range(n):
                for c in range(n+1): t_e[r][c].grid(row=r, column=c)
        tk.Button(win, text="Set", command=mk).grid(row=0, column=2)
        def run():
            n = int(n_ent.get()); a = np.zeros((n,n)); b = np.zeros(n)
            for r in range(n):
                b[r] = float(t_e[r][n].get())
                for c in range(n): a[r,c] = float(t_e[r][c].get())
            y, z, x = np.zeros(n), np.zeros(n), np.zeros(n)
            y[0] = a[0,0]; z[0] = b[0]/y[0]
            for i in range(1, n):
                y[i] = a[i,i] - (a[i,i-1]*a[i-1,i])/y[i-1]
                z[i] = (b[i] - a[i,i-1]*z[i-1])/y[i]
            x[n-1] = z[n-1]
            for i in range(n-2, -1, -1): x[i] = z[i] - (a[i,i+1]*x[i+1])/y[i]
            self.tree["columns"] = ("i","y","z","x")
            for r in self.tree.get_children(): self.tree.delete(r)
            for i in range(n): self.tree.insert("", "end", values=(i+1, f"{y[i]:.4f}", f"{z[i]:.4f}", f"{x[i]:.4f}"))
            win.destroy()
        tk.Button(win, text="RUN", command=run).grid(row=2, column=0)

if __name__ == "__main__":
    root = tk.Tk(); app = NumericalMethodsUltimateHUE(root); root.mainloop()
