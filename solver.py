import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt

from consts import *
import aerosandbox as asb
import aerosandbox.numpy as np
from scipy.interpolate import interp1d
from consts import *

def build_airplane_curved(genes, kind='quadratic'):
    """
    Buduje samolot z nieliniowymi (łukowatymi) przejściami między sekcjami
    dla optymalnego rozkładu siły nośnej.
    Struktura: base (y=0) -> root -> brk1 -> brk2 -> tip
    """
    # Stałe dla profilu bazowego (na środku w y=0)
    y_base = 0
    c_base = 0.4
    t_base = 0
    x_base = 0
    z_base = 0
    
    # Zmienne punkty przerwania
    y_root = genes['y_root']
    y_brk1 = y_root + genes['y_break1_f'] * (B/2 - y_root)
    y_brk2 = y_brk1 + genes['y_break2_f'] * (B/2 - y_brk1)
    y_tip = y_brk2 + genes['y_tip_f'] * (B/2 - y_brk2)
    
    # Tablice interpolacyjne (5 punktów: base, root, brk1, brk2, tip)
    y_pts = np.array([y_base, y_root, y_brk1, y_brk2, y_tip])
    c_pts = np.array([c_base, genes['c_root'], genes['c_brk1'], genes['c_brk2'], genes['c_tip']])
    x_pts = np.array([x_base, genes['x_root'], genes['x_brk1'], genes['x_brk2'], genes['x_tip']])
    z_pts = np.array([z_base, genes['z_root'], genes['z_brk1'], genes['z_brk2'], genes['z_tip']])
    t_pts = np.array([t_base, genes['t_root'], genes['t_brk1'], genes['t_brk2'], genes['t_tip']])

    f_chord = interp1d(y_pts, c_pts, kind=kind)
    f_x     = interp1d(y_pts, x_pts, kind=kind)
    f_z     = interp1d(y_pts, z_pts, kind=kind)
    f_t     = interp1d(y_pts, t_pts, kind=kind)

    num_main_sections = 10
    y_dense = np.linspace(y_base, y_tip, num_main_sections)
    
    sections = []
    
    # Pobieramy profile obiektowe (4 profile przejściowe)
    af_base = AIRFOIL_DATABASE[int(genes['id_root'])]  # base używa tego samego profilu co root
    af_root = AIRFOIL_DATABASE[int(genes['id_root'])]
    af_brk1 = AIRFOIL_DATABASE[int(genes['id_brk1'])]
    af_brk2 = AIRFOIL_DATABASE[int(genes['id_brk1'])]  # brk2 używa tego samego profilu co brk1
    af_tip  = AIRFOIL_DATABASE[int(genes['id_tip'])]

    for i in range(len(y_dense)):
        y_val = y_dense[i]
        
        if y_val <= y_root:
            # Interwał [y_base, y_root]
            fraction = (y_val - y_base) / (y_root - y_base) if y_root != y_base else 0
            af = af_base.blend_with_another_airfoil(af_root, fraction)
        elif y_val <= y_brk1:
            # Interwał [y_root, y_brk1]
            fraction = (y_val - y_root) / (y_brk1 - y_root) if y_brk1 != y_root else 0
            af = af_root.blend_with_another_airfoil(af_brk1, fraction)
        elif y_val <= y_brk2:
            # Interwał [y_brk1, y_brk2]
            fraction = (y_val - y_brk1) / (y_brk2 - y_brk1) if y_brk2 != y_brk1 else 0
            af = af_brk1.blend_with_another_airfoil(af_brk2, fraction)
        else:
            # Interwał [y_brk2, y_tip]
            fraction = (y_val - y_brk2) / (y_tip - y_brk2) if y_tip != y_brk2 else 0
            af = af_brk2.blend_with_another_airfoil(af_tip, fraction)

        sections.append(
            asb.WingXSec(
                xyz_le=[f_x(y_val), y_val, f_z(y_val)],
                chord=f_chord(y_val),
                twist=f_t(y_val),
                airfoil=af
            )
        )

    # 4. Logika Wingletu (zachowujemy Twoją sprawdzoną gładką geometrię)
    dy = y_tip - y_brk2
    dz = genes['z_tip'] - genes['z_brk2']
    wing_dihedral_rad = np.arctan2(dz, dy)
    
    target_angle_rad = np.radians(genes['winglet_target_angle'])
    target_angle_rad = np.maximum(target_angle_rad, wing_dihedral_rad + np.radians(2))
    
    R_w = genes['R_w']
    h_w = genes['h_w']
    extension_len = np.maximum(0, h_w - R_w)
    
    num_arc_steps = 8
    phi_arc = np.linspace(wing_dihedral_rad - np.pi/2, target_angle_rad - np.pi/2, num_arc_steps)
    
    cy = y_tip - R_w * np.cos(wing_dihedral_rad - np.pi/2)
    cz = genes['z_tip'] - R_w * np.sin(wing_dihedral_rad - np.pi/2)
    
    y_arc = cy + R_w * np.cos(phi_arc)
    z_arc = cz + R_w * np.sin(phi_arc)
    
    if extension_len > 0:
        num_ext_steps = 1
        dist_ext = np.linspace(0, extension_len, num_ext_steps)[1:]
        y_w_full = np.concatenate([y_arc, y_ext := y_arc[-1] + dist_ext * np.cos(target_angle_rad)])
        z_w_full = np.concatenate([z_arc, z_ext := z_arc[-1] + dist_ext * np.sin(target_angle_rad)])
    else:
        y_w_full, z_w_full = y_arc, z_arc

    # Parametry wingletu (skos i cięciwa)
    actual_w_steps = len(y_w_full)
    x_w_full = genes['x_tip'] + np.linspace(0, genes['sweep_w'], actual_w_steps)
    c_w_full = np.linspace(genes['c_tip'], genes['c_w_end'], actual_w_steps)

    # Dodanie stacji wingletu do listy (pomijamy pierwszą, bo to y_tip)
    af_w = AIRFOIL_DATABASE[int(genes['id_w'])]
    for i in range(1, actual_w_steps):
        fraction = i / (actual_w_steps - 1)
        af = af_tip.blend_with_another_airfoil(af_w, fraction)
        sections.append(
            asb.WingXSec(
                xyz_le=[x_w_full[i], y_w_full[i], z_w_full[i]],
                chord=c_w_full[i],
                twist=np.interp(fraction, [0, 1], [genes['t_tip'], genes['t_tip'] - genes['toe']]),
                airfoil=af
            )
        )

    # 5. Finalizacja modelu
    wing = asb.Wing(name="Curved Wing", xsecs=sections, symmetric=True,
                    spanwise_panels=30,      # Stała liczba paneli na stronę
                    spanwise_spacing="cosine", # Zagęszczenie na końcach i w środku
                    chordwise_panels=8,      # Standard dla VLM
                    chordwise_spacing="cosine")
    airplane = asb.Airplane(wings=[wing])
    total_mass = FIXED_MASS + (wing.area() * WING_DENSITY)
    
    return airplane, total_mass



def build_airplane(genes):
    # 1. Dekodowanie geometrii podstawowej
    y_root = genes['y_root']
    y_break = y_root + genes['y_break_f'] * (B/2 - y_root)
    y_tip = y_break + genes['y_tip_f'] * (B/2 - y_break)
    
    # 2. Kąty i Dihedral
    dy = y_tip - y_break
    dz = genes['z_tip'] - genes['z_brk']
    wing_dihedral_rad = np.arctan2(dz, dy)
    
    target_angle_rad = np.radians(genes['winglet_target_angle'])
    target_angle_rad = np.maximum(target_angle_rad, wing_dihedral_rad + np.radians(2))
    
    # 3. Parametryzacja łuku i wydłużenia
    R_w = genes['R_w']
    h_w = genes['h_w']
    extension_len = np.maximum(0, h_w - R_w)
    
    # Definiujemy kroki dla łuku
    num_arc_steps = 8
    phi_arc = np.linspace(wing_dihedral_rad - np.pi/2, target_angle_rad - np.pi/2, num_arc_steps)
    
    # Środek łuku
    cy = y_tip - R_w * np.cos(wing_dihedral_rad - np.pi/2)
    cz = genes['z_tip'] - R_w * np.sin(wing_dihedral_rad - np.pi/2)
    
    # Współrzędne Y i Z dla łuku
    y_arc = cy + R_w * np.cos(phi_arc)
    z_arc = cz + R_w * np.sin(phi_arc)
    
    # 4. Dodanie wydłużenia jako integralnej części wektora współrzędnych
    if extension_len > 0:
        # Kolejne stacje na prostej (np. 3 dodatkowe punkty dla gładkości wizualnej)
        num_ext_steps = 4
        dist_ext = np.linspace(0, extension_len, num_ext_steps)[1:] # [1:] bo 0 to koniec łuku
        
        y_ext = y_arc[-1] + dist_ext * np.cos(target_angle_rad)
        z_ext = z_arc[-1] + dist_ext * np.sin(target_angle_rad)
        
        y_w_full = np.concatenate([y_arc, y_ext])
        z_w_full = np.concatenate([z_arc, z_ext])
    else:
        y_w_full = y_arc
        z_w_full = z_arc

    # 5. Skos (X) i Cięciwa (Chord) rozłożone na CAŁEJ długości wingleta
    # Obliczamy "długość" wingleta po krzywej, aby sweep był stały/liniowy
    actual_steps = len(y_w_full)
    
    # Skos (X) narasta od x_tip do x_tip + sweep_w
    x_w_full = genes['x_tip'] + np.linspace(0, genes['sweep_w'], actual_steps)
    
    # Cięciwa (Chord) maleje do c_w_end
    c_w_full = np.linspace(genes['c_tip'], genes['c_w_end'], actual_steps)

    # 6. Definicja stacji
    sections = [
        asb.WingXSec(xyz_le=[0, y_root, 0], chord=genes['c_root'], twist=genes['t_root'], 
                     airfoil=AIRFOIL_DATABASE[int(genes['id_root'])]),
        asb.WingXSec(xyz_le=[genes['x_brk'], y_break, genes['z_brk']], chord=genes['c_brk'], twist=genes['t_brk'], 
                     airfoil=AIRFOIL_DATABASE[int(genes['id_brk'])]),
        asb.WingXSec(xyz_le=[genes['x_tip'], y_tip, genes['z_tip']], chord=genes['c_tip'], twist=genes['t_tip'], 
                     airfoil=AIRFOIL_DATABASE[int(genes['id_tip'])])
    ]

    # Dodanie wszystkich stacji wingleta (pomijamy indeks 0, bo to y_tip)
    for i in range(1, actual_steps):
        sections.append(
            asb.WingXSec(
                xyz_le=[x_w_full[i], y_w_full[i], z_w_full[i]], 
                chord=c_w_full[i], 
                twist=genes['t_tip'] - genes['toe'], 
                airfoil=AIRFOIL_DATABASE[int(genes['id_w'])]
            )
        )

    wing = asb.Wing(name="Main Wing", xsecs=sections, symmetric=True)
    airplane = asb.Airplane(wings=[wing])
    total_mass = FIXED_MASS + (wing.area() * WING_DENSITY)
    
    return airplane, total_mass


import numpy as numpy

def GA_aerodynamics(airplane, velocity, alphas, xyz_ref_orig=[0.19, 0, 0], beta=0, sm_target=0.08):
    """
    Wykonuje wektoryzowaną analizę AeroBuildup i przygotowuje skorygowane dane 
    dla funkcji fitness w punkcie najlepszego L/D.
    """
    op = asb.OperatingPoint(
        velocity=velocity,
        alpha=alphas,
        beta=beta
    )

    ab = asb.AeroBuildup(airplane, op, xyz_ref=xyz_ref_orig)
    res = ab.run_with_stability_derivatives(alpha=True, beta=True, p=True, q=True, r=True)

    CL = res["CL"]
    CD = res["CD"]
    LD = numpy.where(CD != 0, CL / CD, 0)
    # Znalezienie punktu pracy (Best L/D)
    idx_best = numpy.nanargmax(LD)
    
    # 3. KOREKTA NA DOCELOWY ŚRODEK CIĘŻKOŚCI (CG)
    # Wyciągamy dane dla punktu Best L/D
    main_wing = airplane.wings[0]
    mac = main_wing.mean_aerodynamic_chord()
    x_np = res["x_np"][idx_best]  # Położenie punktu neutralnego w tym punkcie pracy
    
    x_cg_target = x_np - (sm_target * mac)
    
    # Ramię siły (różnica między punktem analizy 0.19 a docelowym CG)
    dx_bar = (x_cg_target - xyz_ref_orig[0]) / mac
    
    # 4. Przeliczanie momentów i pochodnych na x_cg_target
    # Korekta Cm (moment trymujący)
    cm_trimmed = res["Cm"][idx_best] + CL[idx_best] * dx_bar
    
    # Korekta Cma (stateczność podłużna)
    cma_trimmed = res["Cma"][idx_best] + res["CLa"][idx_best] * dx_bar
    
    # Korekta Cnb (stateczność kierunkowa)
    # Cnb = Cnb_ref - CYb * dx_bar (CYb jest ujemne, więc przesunięcie CG w przód zwiększa stabilność)
    cnb_trimmed = res["Cnb"][idx_best] - res["CYb"][idx_best] * dx_bar

    # Korekta Cnr (tłumienie odchylania) - również zależy od ramienia
    cnr_trimmed = res["Cnr"][idx_best] + res["CYr"][idx_best] * dx_bar

    
    mask = (CL > -0.2) & (CL < 0.8)  
    k, _ = np.polyfit(CL[mask]**2, CD[mask], 1)

    span = main_wing.span()
    AR = main_wing.aspect_ratio()
    z_coords = [xsec.xyz_le[2] for xsec in main_wing.xsecs]
    h_winglet = np.max(z_coords) - np.min(z_coords)
    ar_eff = AR * (1 + 1.9 * (h_winglet / span))
    e = 1 / (np.pi * ar_eff * k)

    data_for_fitness = {
        'ld':      LD[idx_best],
        'cm_cg':   cm_trimmed,
        'cma':     cma_trimmed,
        'cnb':     cnb_trimmed,
        'oswald':  e,
        'cmq':     res['Cmq'][idx_best], # Tłumienie pitch (mała korekta pomijana dla uproszczenia)
        'clp':     res['Clp'][idx_best], # Tłumienie roll (nie zależy od X_cg)
        'cnr':     cnr_trimmed,          # Tłumienie yaw (skorygowane)
        'alpha':   alphas[idx_best]
    }

    return data_for_fitness, x_cg_target


def fitness_function_weighted(data):
    """
    Oblicza ocenę samolotu (0-100 pkt).
    
    Argument 'data' musi być słownikiem zawierającym:
    - ld (float): Doskonałość
    - cm_cg (float): Moment trymujący przy założonym CG
    - cma, cnb (float): Pochodne stateczności statycznej
    - cmq, clp, cnr (float): Pochodne tłumienia dynamicznego
    - oswald (float): Współczynnik efektywności obrysu
    """
    if data is None: return 0.0, 

    # --- KONFIGURACJA WAG (Suma = 1.0) ---
    weights = {
        'w_ld':      0.40,  # Główny priorytet: osiągi
        'w_trim':    0.15,  # Jakość trymowania (Cm_cg blisko 0)
        'w_stab':    0.10,  # Stateczność statyczna (Cma, Cnb)
        'w_d_pitch': 0.1,  # Tłumienie pochylania (Cmq)
        'w_d_roll':  0.05,  # Tłumienie przechylania (Clp)
        'w_d_yaw':   0.20,  # Tłumienie odchylania (Cnr - ważne dla wingletów)
        'w_oswald':  0.10   # Eliptyczność (Oswald) - promuje smukłe skrzydła
    }

    # 1. DOSKONAŁOŚĆ (0-120 pkt) - liniowe skalowanie względem celu 25
    score_ld = np.clip((np.maximum(data['ld']-10,0) / 25) * 100, 0, 120)

    # 2. TRYM (0-100 pkt) - Gauss wokół Cm = 0
    # Sigma 0.02 pozwala na nieco większą tolerancję przy mniejszej wadze
    sigma_trim = 0.01
    score_trim = np.exp(-(data['cm_cg']**2) / (2 * sigma_trim**2)) * 100

    # 3. STATECZNOŚĆ STATYCZNA (0-100 pkt)
    f_cma = 1 / (1 + np.exp(10 * (data['cma'] + 0.4)))
    f_cnb = 1 / (1 + np.exp(-50 * (data['cnb'] - 0.04)))
    score_stab = (f_cma * 0.5 + f_cnb * 0.5) * 100

    # 4. TŁUMIENIE DYNAMICZNE (Rozdzielone)
    # Każde tłumienie musi być ujemne (ujemna wartość oznacza stabilność)
    def calc_damp_score(val, target):
        if val > 0: return 0.0  # Kara za niestabilność dynamiczną
        return np.clip((abs(val) / target) * 100, 0, 100)

    # Wartości targetowe są przybliżone dla typowych małych UAV:
    score_damp_pitch = calc_damp_score(data['cmq'], 4.0)
    score_damp_roll  = calc_damp_score(data['clp'], 1.0)
    score_damp_yaw   = calc_damp_score(data['cnr'], 0.5)

    # 5. OSWALD (Bonusowy mnożnik końcowy lub waga)
    # Traktujemy to jako modyfikator jakości osiągów
    score_oswald = np.clip((data['oswald'] - 0.8) / 0.25 * 100, 0, 100)
    # --- OBLICZENIE ŚREDNIEJ WAŻONEJ ---
    final_score = (
        score_ld         * weights['w_ld'] +
        score_trim       * weights['w_trim'] +
        score_stab       * weights['w_stab'] +
        score_damp_pitch * weights['w_d_pitch'] +
        score_damp_roll  * weights['w_d_roll'] +
        score_damp_yaw   * weights['w_d_yaw'] +
        score_oswald     * weights['w_oswald']
    ) / sum(weights.values()) 
    
    score_details = {
        'ld':     score_ld,
        'trim': score_trim,
        'stab': score_stab,
        'd_pitch':score_damp_pitch,
        'd_roll':score_damp_roll,
        'd_yaw':score_damp_yaw,
        'oswald': score_oswald
    }
    
    # "Bezpiecznik" - jeśli samolot jest statycznie niestabilny, wynik drastycznie spada
    if data['cma'] > 0 or data['cnb'] < 0:
        final_score *= 0.15

    return float(final_score), score_details


def analyze_aerodynamics(airplane, total_mass, velocity, method='buildup'):
    """
    Zoptymalizowana analiza: oblicza NP, trymuje i pobiera Cnbeta w jednym procesie.

    Args:
        airplane: Obiekt Airplane zbudowany z genów.
        total_mass: Całkowita masa samolotu (używana do obliczeń statycznego marginesu).
        velocity: Prędkość przelotowa [m/s].
        method: Metoda analizy {'buildup', 'vortex'}
    """
    # 1. Obliczamy Punkt Neutralny (NP) metodą różniczkową
    # Używamy małego kroku, by zachować liniowość
    alpha_base = 2.0
    alpha_step = 0.5
    
    op_base = asb.OperatingPoint(velocity=velocity, alpha=alpha_base)
    op_pert = asb.OperatingPoint(velocity=velocity, alpha=alpha_base + alpha_step)
    op_beta = asb.OperatingPoint(velocity=velocity, alpha=alpha_base, beta=1.0)
    
    try:
        # Wykonujemy dwa biegi VLM
        if(method=='buildup'):
            ab1 = asb.AeroBuildup(airplane, op_base)
            res1 = ab1.run()
            
            ab1.op_point = op_pert 
            res2 = ab1.run()

            ab1.op_point = op_beta
            res_beta = ab1.run()

        elif method=='votex':
            vlm1 = asb.VortexLatticeMethod(airplane, op_base)
            res1 = vlm1.run()
            vlm1.op_point = op_pert
            res2 = vlm1.run()
            vlm1.op_point = op_beta
            res_beta = vlm1.run()
        
        # 2. Wyznaczamy NP (Neutral Point) - niezależny od aktualnego CG
        main_wing = airplane.wings[0]
        mac = main_wing.mean_aerodynamic_chord()
        dCL = res2['CL'] - res1['CL']
        dCm = res2['Cm'] - res1['Cm']
        
        # x_np względem aktualnego xyz_ref (zazwyczaj 0,0,0)
        x_np = airplane.xyz_ref[0] - (dCm / dCL) * mac if dCL != 0 else 0
        
        # 3. NARZUCAMY Środek Ciężkości (CG) dla zadanego marginesu (np. 8%)
        # Zamiast go zgadywać, przesuwamy go tam, gdzie być powinien
        desired_sm = 0.08 
        x_cg = x_np - (desired_sm * mac)
        
        # 4. ROZWIĄZANIE BŁĘDU Cnbeta: 
        # Pochodne stateczności (jak Cnb) wymagają vlm.substitute_solution() 
        # lub wywołania vlm.run_stability_analysis()
        # Najszybsza metoda na Cnbeta w VLM:
        
        
        # Cnbeta = dCn / dbeta (beta w radianach)
        cnb = (res_beta['Cn'] - res1['Cn']) / np.radians(1.0)

        # 5. Obliczamy moment trymujący Cm_cg
        # Musimy przeliczyć Cm z punktu ref (0,0,0) na nasz nowy x_cg
        # Cm_cg = Cm_ref + CL * ( (x_cg - x_ref) / mac )
        cm_at_cg = res1['Cm'] + res1['CL'] * ((x_cg - airplane.xyz_ref[0]) / mac)
        cm_at_cg = cm_at_cg[0] if isinstance(cm_at_cg, np.ndarray) else cm_at_cg
        ld = res1['CL'] / res1['CD'] if res1['CD'] != 0 else 0
        ld = ld[0] if isinstance(ld, np.ndarray) else ld
        return {
            'ld': ld,
            'sm': desired_sm,    # W zwracanym wyniku SM jest stały, bo tak wyważyliśmy
            'cnb': cnb[0] if isinstance(cnb, np.ndarray) else cnb,         # Stateczność boczna
            'cl': res1['CL'][0] if isinstance(res1['CL'], np.ndarray) else res1['CL'],
            'cd': res1['CD'][0] if isinstance(res1['CD'], np.ndarray) else res1['CD'],
            'cm_cg': cm_at_cg,   # To mówi nam, jak bardzo samolot chce "nurkować"
            'mass': total_mass,
            'x_np': x_np,
            'x_cg': x_cg,
            'obj': vlm1 if method=='votex' else ab1,
            'res1': res1
        }
        
    except Exception as e:
        print(f"Błąd analizy: {e}")
        return None
    

def calculate_extra_metrics(airplane, total_mass, velocity):
    """
    Oblicza zaawansowane parametry geometryczne i fizyczne modelu.
    """
    wing = airplane.wings[0]
    
    # 1. Powierzchnia i Rozpiętość
    s_area = wing.area()
    span = wing.span()
    
    # 2. Aspect Ratio (AR) - klasyczne
    ar = (span**2) / s_area
    #ar2 = wing.aspect_ratio()
    
    # 3. Effective Aspect Ratio (AR_eff)
    # Winglety zwiększają efektywną rozpiętość bez fizycznego jej zwiększania.
    # Uproszczony model: AR_eff = AR * (1 + 1.9 * (h_winglet / span))
    # Wyciągamy wysokość wingletu z różnicy Z stacji
    z_coords = [xsec.xyz_le[2] for xsec in wing.xsecs]
    h_winglet = np.max(z_coords) - np.min(z_coords)
    ar_eff = ar * (1 + 1.9 * (h_winglet / span))
    
    # 4. Średnia Cięciwa Aerodynamiczna (MAC)
    mac = wing.mean_aerodynamic_chord()
    
    # 5. Liczba Reynoldsa (Re)
    # Re = (rho * v * L) / mu
    rho = 1.225 # gęstość powietrza na poziomie morza
    mu = 1.81e-5 # lepkość dynamiczna
    re = (rho * velocity * mac) / mu
    

    wing_loading = total_mass / s_area # kg/m^2

    taper = wing.taper_ratio()
    e_raymer = 1.78 * (1 - 0.045 * ar_eff**0.68) - 0.64
    e = e_raymer * (1 - 0.1 * (1 - taper)**2)
    MtA = wing.mean_twist_angle()
    MsA = wing.mean_sweep_angle()
    MdA = wing.mean_dihedral_angle()
    aero_center = wing.aerodynamic_center()
    return {
        'S [m^2]': s_area,
        'Span': span,
        'Aerodynamic center': aero_center,
        'MAC [m]': mac,
        'Re': re,
        'AR': ar,
        'AR_eff': ar_eff,
        'Wing Loading [kg/m^2]': wing_loading,
        'Taper': taper,
        'e': e,
        'MtA': MtA,
        'MsA': MsA,
        'MdA': MdA
    }

def stability_analysis(airplane, x_cg):
    """
    Wykonuje pełną analizę stateczności statycznej i dynamicznej.
    """
    op = asb.OperatingPoint(velocity=TARGET_VELOCITY, alpha=2.0)
    ab = asb.AeroBuildup(airplane, op, xyz_ref=[x_cg, 0, 0])
    
    # Obliczamy pochodne względem kątów (alpha, beta) i prędkości kątowych (p, q, r)
    full_res = ab.run_with_stability_derivatives(
        alpha=True, beta=True, p=True, q=True, r=True
    )
    
    print("\n--- RAPORT STATECZNOŚCI ---")
    print(f"Punkt Neutralny (x_np): {full_res['x_np']:.4f} m")
    print(f"Pochodna momentu pochylającego (Cma): {full_res['Cma']:.4f} (Ujemna = stabilny)")
    print(f"Pochodna momentu kierunkowego (Cnb): {full_res['Cnb']:.4f} (Dodatnia = stabilny)")
    print(f"Pochodna momentu przechylającego (Clb): {full_res['Clb']:.4f} (Ujemna = efekt dwuścienny)")
    
    print("\n--- TŁUMIENIE (DYNAMICZNE) ---")
    print(f"Tłumienie pochylania (Cmq): {full_res['Cmq']:.4f} (Ujemne = wygasza oscylacje)")
    print(f"Tłumienie przechylania (Clp): {full_res['Clp']:.4f} (Ujemne = stabilne)")
    print(f"Tłumienie odchylania (Cnr): {full_res['Cnr']:.4f} (Ujemne = wygasza 'rybkowanie')")
    
    return full_res



def analyze_alpha_sweep_and_plot(
    airplane,
    velocity,
    alphas,
    xyz_ref=[0, 0, 0],
    beta=0,
    plot=True
):
    """
    Szybka analiza aerodynamiczna vs AoA (wektoryzowana, AeroBuildup).

    Zwraca:
        dict z wynikami + metryki (max L/D, trim, Oswald e)
    """

    # --- wektoryzowany OperatingPoint ---
    op = asb.OperatingPoint(
        velocity=velocity,
        alpha=alphas,
        beta=beta
    )

    ab = asb.AeroBuildup(airplane, op, xyz_ref=xyz_ref)
    res = ab.run_with_stability_derivatives(alpha=True, beta=True, p=True, q=True, r=True)

    CL = res["CL"]
    CD = res["CD"]
    Cm = res["Cm"]
    L = res["L"]
    D = res["D"]

    LD = np.where(CD != 0, CL / CD, np.nan)

    # =========================
    # 🔍 ANALIZA WYNIKÓW
    # =========================

    # --- max L/D ---
    idx_best = np.nanargmax(LD)
    alpha_best = alphas[idx_best]
    LD_best = LD[idx_best]


    # --- Oswald efficiency z polary ---
    try:
        mask = (CL > -0.2) & (CL < 0.8)  # zakres liniowy
        k, _ = np.polyfit(CL[mask]**2, CD[mask], 1)

        AR = airplane.wings[0].aspect_ratio()
        e = 1 / (np.pi * AR * k)
    except:
        print("Error occurred while calculating Oswald efficiency.")
        e = np.nan

    # =========================
    # 📊 WYKRESY
    # =========================
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))

        # --- L/D vs alpha ---
        axs[0, 0].plot(alphas, LD, label="L/D")
        axs[0, 0].scatter(alpha_best, LD_best, label=f"Max L/D = {LD_best:.2f}")
        axs[0, 0].set_title("Efficiency (L/D) vs Angle of Attack")
        axs[0, 0].set_xlabel("Angle of Attack α [deg]")
        axs[0, 0].set_ylabel("Lift-to-Drag Ratio (L/D)")
        axs[0, 0].legend()
        axs[0, 0].grid()

        # --- CL vs alpha ---
        axs[0, 1].plot(alphas, CL, label="CL")
        axs[0, 1].set_title("Lift Coefficient vs Angle of Attack")
        axs[0, 1].set_xlabel("Angle of Attack α [deg]")
        axs[0, 1].set_ylabel("Lift Coefficient CL [-]")
        axs[0, 1].grid()

        # --- CL vs CD ---
        axs[1, 0].plot(CD, CL, label="Polar")
        axs[1, 0].set_title("Drag Polar (CL vs CD)")
        axs[1, 0].set_xlabel("Drag Coefficient CD [-]")
        axs[1, 0].set_ylabel("Lift Coefficient CL [-]")
        axs[1, 0].grid()

        # --- Cm vs alpha ---
        axs[1, 1].plot(alphas, Cm, label="Cm")
        axs[1, 1].set_title("Pitching Moment vs Angle of Attack")
        axs[1, 1].set_xlabel("Angle of Attack α [deg]")
        axs[1, 1].set_ylabel("Moment Coefficient Cm [-]")
        axs[1, 1].legend()
        axs[1, 1].grid()

        plt.tight_layout()
        plt.show()

        print("\n=== SUMMARY ===")
        print(f"Max L/D       : {LD_best:.2f} at α = {alpha_best:.2f} deg")
        print(f"Oswald e      : {e:.3f}")

    # =========================
    # 📦 ZWRACANE DANE
    # =========================
    return {
        "alpha": alphas,
        "CL": CL,
        "CD": CD,
        "Cm": Cm,
        "L": L,
        "D": D,
        "L_over_D": LD,

        "alpha_best_LD": alpha_best,
        "LD_max": LD_best,
        "oswald_e": e,
        'idx_best': idx_best, 
        "raw_result": res,
        "airplane": airplane,
        "velocity": velocity
    }



def analyze_alpha_sweep(airplane, velocity, alphas, xyz_ref=[0, 0, 0], beta=0):
    """
    Wykonuje wektoryzowaną analizę aerodynamiczną AeroBuildup.
    Zwraca słownik z wynikami gotowy do analizy lub wizualizacji.
    """
    # --- Wektoryzowany OperatingPoint ---
    op = asb.OperatingPoint(
        velocity=velocity,
        alpha=alphas,
        beta=beta
    )

    ab = asb.AeroBuildup(airplane, op, xyz_ref=xyz_ref)
    res = ab.run_with_stability_derivatives(alpha=True, beta=True, p=True, q=True, r=True)

    CL = res["CL"]
    CD = res["CD"]
    Cm = res["Cm"]
    
    # Obliczanie L/D (obsługa dzielenia przez zero)
    LD = np.where(CD != 0, CL / CD, np.nan)

    # --- max L/D ---
    idx_best = np.nanargmax(LD)
    alpha_best = alphas[idx_best]
    LD_best = LD[idx_best]

    # --- Oswald efficiency (metryka pomocnicza z polary) ---
    e = res['wing_aero_components'][0].oswalds_efficiency

    return {
        "alpha": alphas,
        "CL": CL,
        "CD": CD,
        "Cm": Cm,
        "L": res["L"],
        "D": res["D"],
        "L_over_D": LD,
        "alpha_best_LD": alpha_best,
        "LD_max": LD_best,
        "oswald_e": e,
        "raw_result": res,
        "airplane": airplane,
        "velocity": velocity,
        "idx_best": idx_best  # przydatne do wyciągania pochodnych w punkcie pracy
    }


def plot_alpha_sweep(results):
    """
    Tworzy wykresy na podstawie wyników z analyze_alpha_sweep.
    """
    alphas = results["alpha"]
    LD = results["L_over_D"]
    CL = results["CL"]
    CD = results["CD"]
    Cm = results["Cm"]
    LD_best = results["LD_max"]
    alpha_best = results["alpha_best_LD"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # --- L/D vs alpha ---
    axs[0, 0].plot(alphas, LD, label="L/D", color='green')
    axs[0, 0].scatter(alpha_best, LD_best, color='red', label=f"Max L/D = {LD_best:.2f}")
    axs[0, 0].set_title("Efficiency (L/D) vs Angle of Attack")
    axs[0, 0].set_xlabel("Angle of Attack α [deg]")
    axs[0, 0].set_ylabel("Lift-to-Drag Ratio (L/D)")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # --- CL vs alpha ---
    axs[0, 1].plot(alphas, CL, label="CL", color='blue')
    axs[0, 1].set_title("Lift Coefficient vs Angle of Attack")
    axs[0, 1].set_xlabel("Angle of Attack α [deg]")
    axs[0, 1].set_ylabel("Lift Coefficient CL [-]")
    axs[0, 1].grid(True, alpha=0.3)

    # --- CL vs CD (Biegunowa) ---
    axs[1, 0].plot(CD, CL, label="Polar", color='black')
    axs[1, 0].set_title("Drag Polar (CL vs CD)")
    axs[1, 0].set_xlabel("Drag Coefficient CD [-]")
    axs[1, 0].set_ylabel("Lift Coefficient CL [-]")
    axs[1, 0].grid(True, alpha=0.3)

    # --- Cm vs alpha ---
    axs[1, 1].plot(alphas, Cm, label="Cm", color='orange')
    axs[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5) # Linia trymu
    axs[1, 1].set_title("Pitching Moment vs Angle of Attack")
    axs[1, 1].set_xlabel("Angle of Attack α [deg]")
    axs[1, 1].set_ylabel("Moment Coefficient Cm [-]")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n=== ANALYZE SUMMARY ===")
    print(f"Max L/D        : {LD_best:.2f} at α = {alpha_best:.2f} deg")
    print(f"Oswald e       : {results['oswald_e']:.3f}")
    print(f"CL at Max L/D  : {CL[results['idx_best']]:.3f}")
    print(f"Cm at Max L/D  : {Cm[results['idx_best']]:.4f}")

def _find_trim_alpha(alpha_range, lift, weight):
    exact_match = np.nonzero(np.isclose(lift, weight, atol=1e-6))[0]
    if exact_match.size > 0:
        return alpha_range[exact_match[0]]

    sign_changes = np.nonzero(np.diff(np.sign(lift - weight)) != 0)[0]
    if sign_changes.size == 0:
        return np.nan

    idx = sign_changes[0]
    return np.interp(
        weight,
        lift[idx : idx + 2],
        alpha_range[idx : idx + 2],
    )


def analyze_velocity_polar(
    airplane,
    mass,
    velocities,
    alpha_range=np.linspace(-5, 12, 20),
    plot=True,
):
    """
    Biegunowa prędkości (sink rate vs velocity).

    Returns:
        dict z wynikami + charakterystyczne prędkości.
    """
    g = 9.81
    weight = mass * g

    velocities = np.asarray(velocities, dtype=float)
    alpha_range = np.asarray(alpha_range, dtype=float)

    sink_rates = np.full_like(velocities, np.nan, dtype=float)
    lift_to_drag = np.full_like(velocities, np.nan, dtype=float)
    alpha_trimmed = np.full_like(velocities, np.nan, dtype=float)

    for i, V in enumerate(velocities):
        op = asb.OperatingPoint(velocity=V, alpha=alpha_range)
        res = asb.AeroBuildup(airplane, op).run()

        L = res["L"]
        D = res["D"]

        alpha_eq = _find_trim_alpha(alpha_range, L, weight)
        if np.isfinite(alpha_eq):
            D_eq = np.interp(alpha_eq, alpha_range, D)
            if D_eq > 0.0:
                sink_rates[i] = V * (D_eq / weight)
                lift_to_drag[i] = weight / D_eq
            alpha_trimmed[i] = alpha_eq

    if np.any(np.isfinite(sink_rates)):
        idx_min_sink = np.nanargmin(sink_rates)
        V_min_sink = velocities[idx_min_sink]
        min_sink = sink_rates[idx_min_sink]
    else:
        V_min_sink = np.nan
        min_sink = np.nan

    if np.any(np.isfinite(lift_to_drag)):
        idx_best_LD = np.nanargmax(lift_to_drag)
        V_best_LD = velocities[idx_best_LD]
        best_LD = lift_to_drag[idx_best_LD]
    else:
        V_best_LD = np.nan
        best_LD = np.nan



    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))

        axs[0, 0].plot(velocities, -sink_rates, label="Sink rate")
        if np.isfinite(min_sink):
            axs[0, 0].scatter(
                V_min_sink,
                -min_sink,
                color="red",
                label=f"Min sink: {min_sink:.2f} m/s",
            )
            axs[0, 0].scatter(
                0,
                0,
                color="black"            )
        axs[0, 0].set_title("Sink Rate vs Velocity")
        axs[0, 0].set_xlabel("Velocity [m/s]")
        axs[0, 0].set_ylabel("Sink Rate [m/s]")
        axs[0, 0].legend()
        axs[0, 0].grid()

        axs[0, 1].plot(velocities, lift_to_drag, label="L/D")
        if np.isfinite(best_LD):
            axs[0, 1].scatter(
                V_best_LD,
                best_LD,
                color="red",
                label=f"Best L/D: {best_LD:.1f}",
            )
        axs[0, 1].set_title("L/D vs Velocity")
        axs[0, 1].set_xlabel("Velocity [m/s]")
        axs[0, 1].set_ylabel("Lift-to-Drag Ratio")
        axs[0, 1].legend()
        axs[0, 1].grid()

        axs[1, 0].plot(velocities, alpha_trimmed)
        axs[1, 0].set_title("Trim Angle of Attack vs Velocity")
        axs[1, 0].set_xlabel("Velocity [m/s]")
        axs[1, 0].set_ylabel("Alpha [deg]")
        axs[1, 0].grid()

        axs[1, 1].plot(velocities, -sink_rates)
        axs[1, 1].set_title("Velocity Polar (Glider-style)")
        axs[1, 1].set_xlabel("Velocity [m/s]")
        axs[1, 1].set_ylabel("Sink Rate [m/s]")
        axs[1, 1].grid()
        
        plt.tight_layout()
        plt.show()

        print("\n=== VELOCITY POLAR SUMMARY ===")
        print(f"Min sink speed   : {V_min_sink:.2f} m/s")
        print(f"Min sink rate    : {min_sink:.2f} m/s")
        print(f"Best glide speed : {V_best_LD:.2f} m/s")
        print(f"Max L/D          : {best_LD:.2f}")

    return {
        "velocity": velocities,
        "sink_rate": sink_rates,
        "L_over_D": lift_to_drag,
        "alpha_trim": alpha_trimmed,
        "V_min_sink": V_min_sink,
        "min_sink": min_sink,
        "V_best_LD": V_best_LD,
        "LD_max": best_LD,
    }


def plot_wing_performance(analysis_results, airplane):
    """
    Wizualizuje wydajność skrzydła na podstawie wyników VLM.
    """
    vlm = analysis_results['vlm_obj']
    op_point = vlm.op_point
    
    # Tworzymy layout wykresów
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # 1. Lift Distribution (Rozkład siły nośnej wzdłuż rozpiętości)
    # Wyciągamy Y-współrzędne z punktów kolokacji paneli
    collocation_pts = vlm.collocation_points
    y_centers = collocation_pts[:, 1]  # Y-coordinate
    
    # Obliczamy lokalny Cl z intensywności wirów
    # Cl_local = 2 * Gamma / (V * chord_eq), gdzie chord_eq ≈ Area/span_local
    V_inf = op_point.velocity
    gamma = vlm.vortex_strengths
    areas = vlm.areas
    cl_local = 2 * gamma / (V_inf * areas)
    
    # Grupowanie paneli po Y do wykresu (średniowanie dla czytelności)
    unique_y = np.unique(y_centers)
    y_plot = []
    cl_plot = []
    
    for y_val in unique_y:
        mask = np.isclose(y_centers, y_val)
        y_plot.append(y_val)
        cl_plot.append(np.mean(cl_local[mask]))
    
    axs[0, 0].plot(y_plot, cl_plot, 'b-o', linewidth=2, markersize=4)
    axs[0, 0].set_title("Rozkład lokalnego współczynnika siły nośnej", fontsize=12)
    axs[0, 0].set_xlabel("Pozycja na rozpiętości [m]")
    axs[0, 0].set_ylabel("C_l (lokalny)")
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Wykresy biegunowe (Sweep Alpha)
    alphas = np.linspace(-2, 10, 12)
    cl_polar = []
    cd_polar = []
    ld_polar = []
    
    for a in alphas:
        op = asb.OperatingPoint(velocity=TARGET_VELOCITY, alpha=a)
        try:
            res = asb.VortexLatticeMethod(airplane, op).run()
            cl_polar.append(res['CL'])
            cd_polar.append(res['CD'])
            ld_polar.append(res['CL'] / res['CD'] if res['CD'] > 0.001 else 0)
        except:
            pass

    # Wykres C_L od Alpha
    axs[0, 1].plot(alphas[:len(cl_polar)], cl_polar, 'r-o', linewidth=2, markersize=5)
    axs[0, 1].set_title("C_L vs Alpha", fontsize=12)
    axs[0, 1].set_xlabel("Alpha [deg]")
    axs[0, 1].set_ylabel("C_L")
    axs[0, 1].grid(True, alpha=0.3)

    # Wykres L/D od Alpha
    axs[1, 0].plot(alphas[:len(ld_polar)], ld_polar, 'g-o', linewidth=2, markersize=5)
    axs[1, 0].set_title("Doskonałość L/D vs Alpha", fontsize=12)
    axs[1, 0].set_xlabel("Alpha [deg]")
    axs[1, 0].set_ylabel("L/D")
    axs[1, 0].grid(True, alpha=0.3)

    # Wykres C_L od C_D (Biegunowa)
    axs[1, 1].plot(cd_polar, cl_polar, 'k-o', linewidth=2, markersize=5)
    axs[1, 1].set_title("Biegunowa (C_L vs C_D)", fontsize=12)
    axs[1, 1].set_xlabel("C_D")
    axs[1, 1].set_ylabel("C_L")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].invert_xaxis()  # Konwencja: C_D od prawej do lewej

    plt.tight_layout()
    plt.show()
    


# --- SOLVER AERODYNAMICZNY ---
def solve_aerodynamics(genes):
    """
    Buduje samolot z genów i uruchamia analizę VLM.
    """
    # Dekodowanie genów pozycji y (procenty na wartości bezwzględne)
    y_root = genes['y_root']
    y_break = y_root + genes['y_break_f'] * (B/2 - y_root)
    y_tip = y_break + genes['y_tip_f'] * (B/2 - y_break)
    
    # Budowa sekcji wingletu (zaokrąglenie) z kątem wzniosu
    # Wylicz wznios skrzydła na czubku (z Break do Tip)
    delta_y = y_tip - y_break
    delta_z = genes['z_tip'] - genes['z_brk']
    wing_dihedral_tip = np.arctan2(delta_z, delta_y)
    
    # Dihedral wingletu (w stopniach → radiany)
    dihedral_w = np.radians(genes['dihedral_w'])
    # Warunek: kąt wzniosu wingletu nie może być mniejszy niż wznios skrzydła na czubku
    dihedral_w = np.maximum(dihedral_w, wing_dihedral_tip)
    
    # Kąt czubka wingletu (w stopniach → radiany)
    winglet_target_angle_rad = np.radians(genes['winglet_target_angle'])
    
    num_w_steps = 6
    # Łuk od kąta wzniosu wingletu do kąta czubka
    phi = np.linspace(dihedral_w, winglet_target_angle_rad, num_w_steps)
    
    # Środek łuku dopasowany do warunku startowego
    cy = y_tip - genes['R_w'] * np.sin(dihedral_w)
    cz = genes['z_tip'] + genes['R_w'] * np.cos(dihedral_w)
    
    y_winglet = cy + genes['R_w'] * np.sin(phi)
    z_winglet = cz - genes['R_w'] * np.cos(phi)
    # Dodatkowe wydłużenie wingletu na czubku
    z_winglet = np.where(np.isclose(phi, winglet_target_angle_rad), z_winglet + (genes['h_w'] - genes['R_w']), z_winglet)

    # Definicja stacji skrzydła
    sections = [
        asb.WingXSec(xyz_le=[0, y_root, 0], chord=genes['c_root'], twist=genes['t_root'], 
                        airfoil=AIRFOIL_DATABASE[int(genes['id_root'])]),
        asb.WingXSec(xyz_le=[genes['x_brk'], y_break, genes['z_brk']], chord=genes['c_brk'], twist=genes['t_brk'], 
                        airfoil=AIRFOIL_DATABASE[int(genes['id_brk'])]),
        asb.WingXSec(xyz_le=[genes['x_tip'], y_tip, genes['z_tip']], chord=genes['c_tip'], twist=genes['t_tip'], 
                        airfoil=AIRFOIL_DATABASE[int(genes['id_tip'])])
    ]

    # Dodanie stacji wingletu (wektoryzacja cięciwy i skosu)
    c_w_steps = np.linspace(genes['c_tip'], genes['c_w_end'], num_w_steps)
    x_w_steps = genes['x_tip'] + np.linspace(0, genes['sweep_w'], num_w_steps)
    
    for i in range(1, num_w_steps):
        sections.append(
            asb.WingXSec(xyz_le=[x_w_steps[i], y_winglet[i], z_winglet[i]], 
                            chord=c_w_steps[i], twist=genes['t_tip'] - genes['toe'], 
                            airfoil=AIRFOIL_DATABASE[int(genes['id_w'])])
        )

    wing = asb.Wing(name="Main Wing", xsecs=sections, symmetric=True)
    airplane = asb.Airplane(wings=[wing])

    # Model masowy
    s_area = wing.area()
    total_mass = FIXED_MASS + (s_area * WING_DENSITY)
    lift_req = total_mass * 9.81

    # Analiza operacyjna
    op_point = asb.OperatingPoint(velocity=TARGET_VELOCITY, alpha=0) # Alpha startowe
    vlm = asb.VortexLatticeMethod(airplane, op_point)
    
    try:
        res = vlm.run()
        # Wyliczenie parametrów do funkcji celu
        cl = res['CL']
        cd = res['CD']
        cm = res['Cm']
        cnb = res['Cnbeta']
        sm = res['static_margin']
        
        # Obliczenie wymaganej doskonałości przy trymowaniu (uproszczone)
        ld = cl / cd if cd != 0 else 0
        return {'ld': ld, 'sm': sm, 'cnb': cnb, 'cl': cl, 'mass': total_mass, 'cm': cm}
    except:
        return None

