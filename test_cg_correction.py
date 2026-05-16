import aerosandbox as asb
import aerosandbox.numpy as np
from solver import *

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

def test_cg_correction():
    airplane, _ = build_airplane_curved(test_genes, 'linear')
    velocity = 15.0
    alphas = np.linspace(-10, 20, 60)
    sm_target = 0.1
    
    print("==== TEST KOREKTY ŚRODKA CIĘŻKOŚCI (CG) ====")
    
    # --- BIEG 1: Arbitralny xyz_ref_orig = [0, 0, 0] ---
    ref_1 = [0.0, 0.0, 0.0]
    data1, x_cg_target1 = GA_aerodynamics(airplane, velocity, alphas, xyz_ref_orig=ref_1, sm_target=sm_target)

    print(f"--- BIEG 1: Ręczna korekta na dystansie od xyz_ref = {ref_1} ---")
    print(f"Wyliczony docelowy X CG: {x_cg_target1:.4f} m")
    print(f"Przetransferowane Cma: {data1['cma']:.6f} | Cmq: {data1['cmq']:.6f}")
    print(f"Przetransferowane Cnb: {data1['cnb']:.6f} | Cnr: {data1['cnr']:.6f}")

    
    # --- BIEG 2: xyz_ref w docelowym CG (korekta matematyczna dx będzie = 0) ---
    ref_2 = [x_cg_target1, 0.0, 0.0]
    data2, x_cg_target2 = GA_aerodynamics(airplane, velocity, alphas, xyz_ref_orig=ref_2, sm_target=sm_target)
    
    print(f"\n--- BIEG 2: Natywne zachowanie AeroBuildup w x_cg = {ref_2} ---")
    print(f"środek ciężkości: {x_cg_target2}")
    print(f"Natywne Cma: {data2['cma']:.6f} | Cmq: {data2['cmq']:.6f}")
    print(f"Natywne Cnb: {data2['cnb']:.6f} | Cnr: {data2['cnr']:.6f}")
 
    
    print("\n--- WERYFIKACJA WYNIKÓW ---")
    assert np.isclose(data1['cma'], data2['cma'], rtol=1e-2, atol=1e-4), f"Błąd Cma: {data1['cma']} vs {data2['cma']}"
    assert np.isclose(data1['cnb'], data2['cnb'], rtol=1e-2, atol=1e-4), f"Błąd Cnb: {data1['cnb']} vs {data2['cnb']}"
    assert np.isclose(data1['cmq'], data2['cmq'], rtol=1e-2, atol=1e-4), f"Błąd Cmq: {data1['cmq']} vs {data2['cmq']}"
    assert np.isclose(data1['cnr'], data2['cnr'], rtol=1e-2, atol=1e-4), f"Błąd Cnr: {data1['cnr']} vs {data2['cnr']}"
    
    print("✅ TEST ZALICZONY POMYŚLNIE! Translacja matematyczna idealnie zgadza się z natywnym symulatorem.")

if __name__ == "__main__":
    test_cg_correction()