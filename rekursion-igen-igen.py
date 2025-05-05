import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, exp, integrate, Eq, simplify, solve, lambdify
import time

# Symbolsk initialisering
t, s = sp.symbols('t s', real=True)
C1, C2 = sp.symbols('C1 C2')

def solve_car_system(v0=20, y1=1, c=1, T=sp.Rational(1,10), b_values=[0.5, 1.5, 2, 3, 5], j_max=3, t_max=10):
    results = {}
    
    for b in b_values:
        print(f"\nLøser for b = {b}")
        # Find rødder til karakteristisk polynomium
        discriminant = b**2 - 4*c
        l1 = (-b + sp.sqrt(discriminant)) / 2
        l2 = (-b - sp.sqrt(discriminant)) / 2
        print(f"Rødder: λ1={l1}, λ2={l2}")
        
        x_list = []
        v_list = []
        
        # ===== Løsning for j=1 =====
        F1 = c*T*v0  # Konstant da v0 er konstant
        
        # Partikulær løsning
        if l1 == l2:  # Dobbeltrod
            x1_par = T*v0
        else:  # To forskellige rødder
            x1_par = T*v0
        
        # Homogen løsning
        if l1 == l2:
            x1_hom = (C1 + C2*t)*exp(l1*t)
        else:
            x1_hom = C1*exp(l1*t) + C2*exp(l2*t)
        
        x1 = x1_hom + x1_par
        x1_deriv = x1.diff(t)
        
        # Initialbetingelser for j=1
        eq1 = Eq(x1.subs(t, 0), T*v0)
        eq2 = Eq(x1_deriv.subs(t, 0), y1)
        sol = solve([eq1, eq2], (C1, C2))
        print(f"Konstanter for j=1: {sol}")
        
        x1 = x1.subs(sol).simplify()
        v1 = (v0 - x1.diff(t)).simplify()
        
        x_list.append(x1)
        v_list.append(v1)
        print(f"x1(t) = {x1}\n")
        print(f"v1(t) = {v1}")
        
        # ===== Rekursiv løsning for j >= 2 =====
        for j in range(2, j_max+1):
            start = time.time()
            print(f"\nLøser for j={j}")
            v_prev = v_list[-1]
            F = v_prev.diff(t) + c*T*v_prev
            
            # Partikulær løsning (nulstartet)
            if l1 == l2:  # Dobbeltrod
                int1 = F * exp(-l1*s)
                int1_val = sp.integrate(int1, (s, 0, t))
                
                int2 = s * F * exp(-l1*s)
                int2_val = sp.integrate(int2, (s, 0, t))
                
                x_par = exp(l1*t) * (t * int1_val - int2_val)
            else:  # To forskellige rødder
                int1 = F * exp(-l1*s)
                int1_val = sp.integrate(int1, (s, 0, t))
                
                int2 = F * exp(-l2*s)
                int2_val = sp.integrate(int2, (s, 0, t))
                
                x_par = (exp(l1*t)*int1_val - exp(l2*t)*int2_val)/(l1 - l2)
            
            # Homogen løsning
            C1_j, C2_j = sp.symbols(f'C1_{j} C2_{j}')
            if l1 == l2:
                x_hom = (C1_j + C2_j*t)*exp(l1*t)
            else:
                x_hom = C1_j*exp(l1*t) + C2_j*exp(l2*t)
            
            x_j = x_hom + x_par
            x_j_deriv = x_j.diff(t)
            
            # Initialbetingelser for j >= 2
            eq1 = Eq(x_j.subs(t, 0), T*v0)
            eq2 = Eq(x_j_deriv.subs(t, 0), 0)
            sol = solve([eq1, eq2], (C1_j, C2_j))
            print(f"Konstanter for j={j}: {sol}")
            
            x_j = x_j.subs(sol).simplify()
            v_j = (v_prev - x_j.diff(t)).simplify()
            
            x_list.append(x_j)
            v_list.append(v_j)
            print(f"x{j}(t) = {x_j}\n")
            print(f"v{j}(t) = {v_j}")

            end = time.time()
            print(f"Tid for j={j}: {end - start:.4f} sekunder")
        
        results[b] = {'x': x_list, 'v': v_list}
    
    return results
jbby = 3
# Kør simulering
results = solve_car_system(j_max=jbby, t_max=20)

# Plot resultater
t_vals = np.linspace(0, 10, 1000)
plt.figure(figsize=(15, 8))

# Plot x_j(t) for alle b
for j in range(jbby):  # x1, x2, x3
    plt.subplot(2, 3, j+1)
    for b in results:
        x_func = lambdify(t, results[b]['x'][j], modules=['numpy'])
        y_vals = x_func(t_vals)
        plt.plot(t_vals, y_vals, label=f'b={b}')
    plt.title(f'Distance $x_{j+1}(t)$ for all b')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.grid(True)
    plt.legend()

# Plot v_j(t) for alle b
for j in range(jbby):  # v1, v2, v3
    plt.subplot(2, 3, j+4)
    for b in results:
        v_func = lambdify(t, results[b]['v'][j], modules=['numpy'])
        y_vals = v_func(t_vals)
        plt.plot(t_vals, y_vals, label=f'b={b}')
    plt.title(f'Speed $v_{j+1}(t)$ for all b')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# Export x_j and v_j to a LaTeX document
def export_to_latex(results, filename="results.tex"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(r"\documentclass[a4paper]{article}" + "\n")
        f.write(r"\usepackage[margin=1in]{geometry}" + "\n")  # Adjust margins
        f.write(r"\usepackage{amsmath}" + "\n")
        f.write(r"\usepackage{breqn}" + "\n")  # For breaking long equations
        f.write(r"\begin{document}" + "\n")
        f.write(r"\section*{Simplified Results}" + "\n")
        
        for b, data in results.items():
            f.write(rf"\subsection*{{Results for $b = {b}$}}" + "\n")
            for j, (xj, vj) in enumerate(zip(data['x'], data['v']), start=1):
                xj_simplified = simplify(xj)
                vj_simplified = simplify(vj)
                f.write(rf"Distance $x_{j}(t)$:" + "\n")
                f.write(r"\begin{dmath}" + "\n")  # Use dmath for line breaking
                f.write(sp.latex(xj_simplified) + "\n")
                f.write(r"\end{dmath}" + "\n")
                f.write(rf"Speed $v_{j}(t)$:" + "\n")
                f.write(r"\begin{dmath}" + "\n")  # Use dmath for line breaking
                f.write(sp.latex(vj_simplified) + "\n")
                f.write(r"\end{dmath}" + "\n")
        
        f.write(r"\end{document}" + "\n")
    print(f"Results exported to {filename}")

# Simplify and export results
export_to_latex(results)




#################Udregninger for eksakte løsninger#######################

# def exact_solutions():
#     # Eksakte løsninger fra rapport
#     solutions = {
#         'x1': t*exp(-t) + 2,
#         'v1': (t + 20*exp(t) - 1)*exp(-t),
#         'x2': (-0.15*t**3 + 0.95*t**2 + 2*exp(t))*exp(-t),
#         'v2': (-0.15*t**3 + 1.45*t**2 - 0.9*t + 20*exp(t) - 1)*exp(-t),
#         'x3': (0.00675*t**5 - 0.1425*t**4 + 0.6017*t**3 + 2*exp(t))*exp(-t),
#         'v3': (0.00675*t**5 - 0.17625*t**4 + 1.0217*t**3 - 0.4051*t - 0.9*t + 20*exp(t) - 1)*exp(-t)
#     }
#     return solutions

# def plot_exact_solutions(t_max=20):
#     sol = exact_solutions()
#     t_vals = np.linspace(0, t_max, 1000)
    
#     plt.figure(figsize=(15, 10))
    
#     # Plot afstande
#     plt.subplot(2, 1, 1)
#     for i in range(1, 4):
#         x_func = lambdify(t, sol[f'x{i}'], 'numpy')
#         plt.plot(t_vals, x_func(t_vals), label=f'$x_{i}(t)$')
#     plt.title('Eksakte løsninger for afstande (b=2)')
#     plt.xlabel('Tid (s)')
#     plt.ylabel('Afstand (m)')
#     plt.grid(True)
#     plt.legend()
    
#     # Plot hastigheder
#     plt.subplot(2, 1, 2)
#     for i in range(1, 4):
#         v_func = lambdify(t, sol[f'v{i}'], 'numpy')
#         plt.plot(t_vals, v_func(t_vals), label=f'$v_{i}(t)$')
#     plt.title('Eksakte løsninger for hastigheder (b=2)')
#     plt.xlabel('Tid (s)')
#     plt.ylabel('Hastighed (m/s)')
#     plt.grid(True)
#     plt.legend()
    
#     plt.tight_layout()
#     plt.show()

# # Kør plotting af eksakte løsninger
# plot_exact_solutions()