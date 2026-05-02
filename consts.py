from pathlib import Path
import aerosandbox as asb
# --- PARAMETRY GLOBALNE ---
B = 1.2                 # Rozpiętość całkowita [m]
TARGET_VELOCITY = 15.0  # Prędkość przelotowa [m/s]
FIXED_MASS = 0.4        # Masa elektroniki i baterii [kg]
TOTAL_MASS = 1.0        # Docelowa masa całkowita [kg]
WING_DENSITY = 1.2      # Masa struktury skrzydła [kg/m2]
AIRFOIL_FOLDER = Path("./airfoils/") # Folder z plikami .dat

# --- FOLDERY DO ZAPISYWANIA PLIKÓW ---
OUTPUT_FOLDER = Path("./output/")  # Główny folder wyjściowy
BEST_SOLUTIONS_FOLDER = OUTPUT_FOLDER / "best_solutions"  # Dla plików best_solution_*.json
HISTORIES_FOLDER = OUTPUT_FOLDER / "histories"  # Dla plików elites_history_*.json, scores_history_*.json, ga_history_*.json
MEDIA_FOLDER = Path("./media/")  # Dla wykresów i innych mediów (jeśli potrzebne)

# Funkcja do tworzenia folderów, jeśli nie istnieją
def ensure_folders_exist():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    BEST_SOLUTIONS_FOLDER.mkdir(parents=True, exist_ok=True)
    HISTORIES_FOLDER.mkdir(parents=True, exist_ok=True)
    MEDIA_FOLDER.mkdir(parents=True, exist_ok=True)

# Wywołanie funkcji przy imporcie modułu
ensure_folders_exist()

# Lista dostępnych profilów (załadowana raz na starcie)
def load_airfoils(folder):
    files = list(folder.glob("*.dat"))
    return {i: asb.Airfoil(name=f.stem, coordinates=f) for i, f in enumerate(files)}

AIRFOIL_DATABASE = load_airfoils(AIRFOIL_FOLDER)