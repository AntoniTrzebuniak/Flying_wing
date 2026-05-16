from solver import *
from consts import TARGET_VELOCITY
test_genes ={
        "y_root": 0.07193656695389453,
        "y_break1_f": 0.4300668011961446,
        "y_break2_f": 0.7,
        "y_tip_f": 0.7538318496503014,
        "c_root": 0.34307719870352416,
        "c_brk1": 0.23066847030538362,
        "c_brk2": 0.05,
        "c_tip": 0.03943716886886501,
        "t_root": 0.1595495948922483,
        "t_brk1": -1.59413526279621,
        "t_brk2": 1.9012082884878891,
        "t_tip": -1.2824427999043413,
        "x_root": 0.05524026532437745,
        "x_brk1": 0.7,
        "x_brk2": 0.7553100952585852,
        "x_tip": 0.8239846082637555,
        "z_root": 0.1,
        "z_brk1": 0.12433097276351382,
        "z_brk2": 0.24,
        "z_tip": 0.0,
        "winglet_target_angle": 89.01317957689737,
        "h_w": 0.12292013888079628,
        "R_w": 0.1,
        "c_w_end": 0.01,
        "sweep_w": 0.1312720350901696,
        "toe": 1.520876984230849,
        "id_root": 3.4673083088697063,
        "id_brk1": 0.5156214065410173,
        "id_tip": 5.096560640684331,
        "id_w": 7.897567616620509
}

airplane, mass = build_airplane_curved(test_genes, 'linear')
print(f"Samolot zbudowany pomyślnie! Szacowana masa: {mass:.3f} kg")
alphas1 = np.linspace(-10, 20, 60)  # Zakres kątów natarcia
data, x_cg_target = GA_aerodynamics(airplane, TARGET_VELOCITY, alphas1, sm_target = 0.1)
print("======data======")
print(data)
score, score_details = fitness_function_weighted(data)
print(f"Score: {score:.2f}")
print(f"Details: {score_details}")  
from genetic_algo import fitness_function
score, details = fitness_function(test_genes)

print(f"Score: {score:.2f}")
print(f"Details: {details}")
print(f"Docelowy X CG: {x_cg_target:.4f} m")

