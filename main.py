from genetic_algo import GeneticAlgorithm
import json
import datetime
import os
from consts import BEST_SOLUTIONS_FOLDER

# Natychmiastowe wyłączenie

def main():
    # Inicjalizacja algorytmu genetycznego
    ga = GeneticAlgorithm(pop_size=80, elite_size=6)
    
    # Uruchomienie ewolucji
    best_individual, best_score = ga.run_evolution(generations=30)
    
    # Zapisanie najlepszych genów do pliku
    with open(BEST_SOLUTIONS_FOLDER / f'best_solution_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json', 'w') as f:
        json.dump({
            'best_score': best_score,
            'best_genes': best_individual
        }, f, indent=4)
    
    print("Najlepsze rozwiązanie zapisane do 'best_solution.json'")

if __name__ == "__main__":
    main()

