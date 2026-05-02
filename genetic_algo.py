import numpy as np
from solver import build_airplane_curved, GA_aerodynamics, fitness_function_weighted
from consts import TARGET_VELOCITY, AIRFOIL_DATABASE, HISTORIES_FOLDER
import multiprocessing
import json
import datetime


def fitness_function(genes):
    """
    Funkcja celu dla algorytmu genetycznego.
    Buduje samolot, oblicza aerodynamikę i zwraca ocenę fitness oraz szczegóły składkowe.
    """
    try:
        airplane, total_mass = build_airplane_curved(genes, kind='linear')
        alphas = np.linspace(-2, 15, 16)
        data, x_cg_target = GA_aerodynamics(airplane, TARGET_VELOCITY, alphas)
        score, score_details = fitness_function_weighted(data)
        return score, score_details
    except Exception as e:
        print(f"Błąd w fitness_function: {e}")
        return 0.0, {}  # Kara za błędy


class GeneticAlgorithm:
    def __init__(self, pop_size=50, elite_size=5):
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.bounds = {
            'y_root': (0, 0),  # Zadane z góry jako 0
            'y_break_f': (0.2, 0.7), 'y_tip_f': (0.71, 0.95),
            'c_root': (0.33, 0.33),  # Zadane z góry jako 0.33
            'c_brk': (0.10, 0.25), 'c_tip': (0.05, 0.15),
            't_root': (0.0, 0.0), 't_brk': (-2, 2), 't_tip': (-5, 5),
            'x_brk': (0, 0.25), 'x_tip': (0.2, 0.5), 
            'z_brk': (-0.03, 0.1), 'z_tip': (0, 0.15), 'winglet_target_angle': (45, 90),
            'h_w': (0.05, 0.15), 'R_w': (0.02, 0.1), 'c_w_end': (0.01, 0.07),
            'sweep_w': (0, 0.1), 'toe': (-5, 5),
            'id_root': (0, len(AIRFOIL_DATABASE)-1), 'id_brk': (0, len(AIRFOIL_DATABASE)-1),
            'id_tip': (0, len(AIRFOIL_DATABASE)-1), 'id_w': (0, len(AIRFOIL_DATABASE)-1)
        }

    def create_individual(self):
        return {k: np.random.uniform(v[0], v[1]) for k, v in self.bounds.items()}

    def run_evolution(self, generations=100):
        print("Creating population...")
        population = [self.create_individual() for _ in range(self.pop_size)]
        
        best_individual = None
        best_score = -np.inf
        elites_history = []
        best_scores_history = []
        mean_scores_history = []
        score_details_history = []
        best_individuals_history = []
        
        # Użyj multiprocessing do zrównoleglenia ewaluacji populacji
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for gen in range(generations):
                print(f"Generation {gen+1}/{generations}")
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
                print(f"Gen {gen+1}: Best Score = {best_score:.4f}, Mean Score = {mean_score:.4f}")
        
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
                'generations': generations,
                'pop_size': self.pop_size,
                'elite_size': self.elite_size,
                'bounds': self.bounds
            }, f, indent=4)
        
        print(f"\nOptymalizacja zakończona. Najlepszy wynik: {best_score:.4f}")
        print("Najlepsze geny:", best_individual)
        return best_individual, best_score