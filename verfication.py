from solver import *
from consts import TARGET_VELOCITY
test_genes ={
        "y_root": 0.0,
        "y_break1_f": 0.20069991554571642,
        "y_break2_f": 0.42049408313736963,
        "y_tip_f": 0.9481092743323929,
        "c_root": 0.26675353516595696,
        "c_brk1": 0.3312119907158733,
        "c_brk2": 0.14730549264905018,
        "c_tip": 0.2190443882475981,
        "t_root": -0.3820175065052861,
        "t_brk1": -0.49640221440431603,
        "t_brk2": -0.6765949882401157,
        "t_tip": -3.904009188886078,
        "x_root": 0.10423250174931582,
        "x_brk1": 0.11943769809000064,
        "x_brk2": 0.17539797029013116,
        "x_tip": 0.652332138519087,
        "z_root": 0.07967843110358873,
        "z_brk1": 0.058949954204139884,
        "z_brk2": 0.03599804388575134,
        "z_tip": 0.028947449290238236,
        "winglet_target_angle": 77.75484941347592,
        "h_w": 0.03190761249774811,
        "R_w": 0.06474899453029252,
        "c_w_end": 0.01,
        "sweep_w": 0.006361086000250319,
        "toe": -0.12482574525796908,
        "id_root": 3.531438389647819,
        "id_brk1": 1.8401570668693754,
        "id_tip": 6.536767710376913,
        "id_w": 6.840308362096924
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

