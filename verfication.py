from solver import *
from consts import TARGET_VELOCITY
test_genes ={
               "y_root": 0.0,
        "y_break_f": 0.2855583796117732,
        "y_tip_f": 0.95,
        "c_root": 0.33,
        "c_brk": 0.13706343816172928,
        "c_tip": 0.05186252019867052,
        "t_root": -0.5008093517467025,
        "t_brk": 1.7327191527712809,
        "t_tip": -4.501131694433842,
        "x_brk": 0.21525767941235854,
        "x_tip": 0.4053832424404723,
        "z_brk": 0.07508087793042074,
        "z_tip": 0.0,
        "winglet_target_angle": 67.451421754612,
        "h_w": 0.08713811320421185,
        "R_w": 0.1,
        "c_w_end": 0.01,
        "sweep_w": 0.012127454715123505,
        "toe": 3.160613791174645,
        "id_root": 5.841011564138323,
        "id_brk": 1.4272441715853488,
        "id_tip": 0.06223774376347112,
        "id_w": 2.861191521707294
}

airplane, mass = build_airplane_curved(test_genes, 'linear')
print(f"Samolot zbudowany pomyślnie! Szacowana masa: {mass:.3f} kg")
alphas1 = np.linspace(-2, 15, 16)  # Zakres kątów natarcia
data, x_cg_target = GA_aerodynamics(airplane, TARGET_VELOCITY, alphas1)
print("======data======")
print(data)
score, score_details = fitness_function_weighted(data)
print(f"Score: {score:.2f}")
print(f"Details: {score_details}")  
from genetic_algo import fitness_function
score, details = fitness_function(test_genes)

print(f"Score: {score:.2f}")
print(f"Details: {details}")

