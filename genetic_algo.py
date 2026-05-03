import numpy as np
from solver import build_airplane_curved, GA_aerodynamics, fitness_function_weighted
from consts import TARGET_VELOCITY, AIRFOIL_DATABASE, HISTORIES_FOLDER
import multiprocessing
import json
import datetime
import threading
import time
import sys

# Zmienne globalne dla trybu interaktywnego
_interactive_stop = False
_interactive_extend = 0

def _interactive_input_thread():
    """Wątek obsługujący interaktywny input bez blokowania głównego procesu."""
    global _interactive_stop, _interactive_extend
    
    print("\n=== INTERAKTYWNY TRYB AKTYWNY ===")
    print("Naciśnij 'q' aby przerwać po obecnej generacji")
    print("Wpisz liczbę aby wydłużyć trening o tyle generacji")
    print("Wpisz 'h' aby wyświetlić pomoc")
    print("================================")
    
    while True:
        try:
            user_input = input().strip().lower()
            
            if user_input == 'q':
                _interactive_stop = True
                print("Przerwanie po obecnej generacji...")
                break
            elif user_input == 'h':
                print("\n=== POMOC ===")
                print("'q' - przerwanie po obecnej generacji")
                print("liczba - wydłużenie treningu o tyle generacji")
                print("'h' - ta pomoc")
                print("============")
            elif user_input.isdigit():
                extend_by = int(user_input)
                _interactive_extend += extend_by
                print(f"Wydłużenie treningu o {extend_by} generacji (łącznie +{_interactive_extend})")
            else:
                print("Nieznana komenda. Wpisz 'h' dla pomocy.")
                
        except (EOFError, KeyboardInterrupt):
            _interactive_stop = True
            break
        except Exception as e:
            print(f"Błąd input: {e}")
            continue


def fitness_function(genes):
    """
    Funkcja celu dla algorytmu genetycznego.
    Buduje samolot, oblicza aerodynamikę i zwraca ocenę fitness oraz szczegóły składkowe.
    """
    
    airplane, total_mass = build_airplane_curved(genes, kind='linear')
    alphas = np.linspace(-5, 10, 30)
    data, x_cg_target = GA_aerodynamics(airplane, TARGET_VELOCITY, alphas)
    score, score_details = fitness_function_weighted(data, genes)
    return score, score_details


class GeneticAlgorithm:
    def __init__(self, pop_size=50, elite_size=5):
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.bounds = {
            # Zmienne punkty Y
            'y_root': (0, 0.2),
            'y_break1_f': (0.2, 0.7), 'y_break2_f': (0.3, 0.7), 'y_tip_f': (0.71, 0.95),
            # Cięciwa
            'c_root': (0.2, 0.4), 'c_brk1': (0.09, 0.35), 'c_brk2': (0.05, 0.3), 'c_tip': (0.03, 0.25),
            # Skręcenie
            't_root': (-1.0, 1.0), 't_brk1': (-2, 2), 't_brk2': (-2, 3), 't_tip': (-5, 5),
            # Pozycja X (sweep)
            'x_root': (0, 0.24), 'x_brk1': (0, 0.7), 'x_brk2': (0.1, 0.99), 'x_tip': (0.2, 1), 
            # Pozycja Z (dihedral)
            'z_root': (-0.03, 0.1), 'z_brk1': (-0.06, 0.15), 'z_brk2': (-0.03, 0.24), 'z_tip': (0, 0.25), 
            # Winglet
            'winglet_target_angle': (45, 90),
            'h_w': (0.01, 0.15), 'R_w': (0.02, 0.1), 'c_w_end': (0.01, 0.09),
            'sweep_w': (0, 0.3), 'toe': (-5, 5),
            # Profile (indeksy do AIRFOIL_DATABASE)
            'id_root': (0, len(AIRFOIL_DATABASE)-1), 'id_brk1': (0, len(AIRFOIL_DATABASE)-1),
            'id_tip': (0, len(AIRFOIL_DATABASE)-1), 'id_w': (0, len(AIRFOIL_DATABASE)-1)
        }

    def create_individual(self):
        return {k: np.random.uniform(v[0], v[1]) for k, v in self.bounds.items()}

    def run_evolution(self, generations=100, interactive=False):
        global _interactive_stop, _interactive_extend
        
        # Resetuj flagi interaktywne
        _interactive_stop = False
        _interactive_extend = 0
        
        # Uruchom wątek interaktywny jeśli włączony
        input_thread = None
        if interactive:
            input_thread = threading.Thread(target=_interactive_input_thread, daemon=True)
            input_thread.start()
        
        print("Creating population...")
        population = [self.create_individual() for _ in range(self.pop_size)]
        
        best_individual = None
        best_score = -np.inf
        elites_history = []
        best_scores_history = []
        mean_scores_history = []
        score_details_history = []
        best_individuals_history = []
        
        target_generations = generations
        
        # Użyj multiprocessing do zrównoleglenia ewaluacji populacji
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            gen = 0
            while gen < target_generations:
                gen += 1
                print(f"Generation {gen}/{target_generations}")
                
                # Sprawdź czy użytkownik chce przerwać
                if _interactive_stop:
                    print(f"Przerwano przez użytkownika po generacji {gen}")
                    break
                
                # Sprawdź czy użytkownik chce wydłużyć
                if _interactive_extend > 0:
                    target_generations += _interactive_extend
                    print(f"Wydłużono trening do {target_generations} generacji")
                    _interactive_extend = 0
                
                # 1. Ewaluacja z zrównolegleniem
                results = pool.map(fitness_function, population)
                scores = np.array([r[0] for r in results])
                score_details_list = [r[1] for r in results]
                
                # Oblicz średni score populacji
                mean_score = float(np.mean(scores))
                mean_scores_history.append(mean_score)
                
                # Znajdź najlepszego w tej generacji
                max_idx = np.argmax(scores)
                current_best_score = float(scores[max_idx])
                if current_best_score > best_score:
                    best_score = current_best_score
                    best_individual = population[max_idx].copy()
                
                # Zapisz historię najlepszego
                best_scores_history.append(best_score)
                score_details_history.append(score_details_list[max_idx])
                best_individuals_history.append(population[max_idx].copy())
                
                # 2. Selekcja elity
                idx_sorted = np.argsort(scores)[::-1]
                elite = [population[i] for i in idx_sorted[:self.elite_size]]
                elites_history.append(elite)
                
                # 3. Reprodukcja (krzyżowanie i mutacja)
                next_gen = elite.copy()
                while len(next_gen) < self.pop_size:
                    parent = population[np.random.choice(idx_sorted[:10])]  # Turniej
                    child = parent.copy()
                    # Mutacja losowego genu
                    gene_to_mutate = np.random.choice(list(self.bounds.keys()))
                    child[gene_to_mutate] += np.random.normal(0, 0.05)
                    # Clip do granic
                    child[gene_to_mutate] = np.clip(child[gene_to_mutate], 
                                                    self.bounds[gene_to_mutate][0], 
                                                    self.bounds[gene_to_mutate][1])
                    next_gen.append(child)
                    
                population = next_gen
                print(f"Gen {gen}: Best Score = {best_score:.4f}, Mean Score = {mean_score:.4f}")
        
        # Zatrzymaj wątek interaktywny jeśli działa
        if input_thread and input_thread.is_alive():
            print("Zatrzymywanie wątku interaktywnego...")
            # Wątek daemon sam się zatrzyma
        
        # Zapisz historię elity do JSON
        with open(HISTORIES_FOLDER / f'elites_history_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json', 'w') as f:
            json.dump(elites_history, f, indent=4)
        
        # Zapisz dodatkowe historie do JSON
        with open(HISTORIES_FOLDER / f'ga_history_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json', 'w') as f:
            json.dump({
                'best_scores_history': best_scores_history,
                'mean_scores_history': mean_scores_history,
                'score_details_history': score_details_history,
                'best_individuals_history': best_individuals_history,
                'generations_completed': gen,
                'target_generations': target_generations,
                'pop_size': self.pop_size,
                'elite_size': self.elite_size,
                'bounds': self.bounds,
                'interactive_mode': interactive
            }, f, indent=4)
        
        print(f"\nOptymalizacja zakończona. Najlepszy wynik: {best_score:.4f}")
        print("Najlepsze geny:", best_individual)
        return best_individual, best_score