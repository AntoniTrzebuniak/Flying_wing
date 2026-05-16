from genetic_algo import GeneticAlgorithm
import json
import datetime
import os
import sys
from consts import BEST_SOLUTIONS_FOLDER

def main(interactive=False):
    # Inicjalizacja algorytmu genetycznego
    ga = GeneticAlgorithm(pop_size=80, elite_size=5)
    
    # Uruchomienie ewolucji
    best_individual, best_score = ga.run_evolution(generations=40, interactive=interactive)
    
    # Zapisanie najlepszych genów do pliku
    with open(BEST_SOLUTIONS_FOLDER / f'best_solution_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json', 'w') as f:
        json.dump({
            'best_score': best_score,
            'best_genes': best_individual
        }, f, indent=4)
    
    print("Najlepsze rozwiązanie zapisane do 'best_solution.json'")

if __name__ == "__main__":
    # Sprawdź argumenty wiersza poleceń
    interactive = "--interactive" in sys.argv or "-i" in sys.argv
    
    if interactive:
        print("Uruchamianie w trybie interaktywnym...")
        print("Użyj 'q' aby przerwać, liczby aby wydłużyć trening")
    
    main(interactive=interactive)

