import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from parallel_BFOA import run_bfoa
from evaluadorBlosum import evaluadorBlosum


def run_multiple_executions(n_runs=30, output_file='resultados.csv'):
    results = []
    total_start_time = time.time()
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_filename = now.strftime("%Y-%m-%d_%H%M")

    blosum_evaluator = evaluadorBlosum()

    for run in tqdm(range(1, n_runs + 1), desc="Ejecutando corridas"):
        start_time = time.time()

        best_fitness, interacciones, blosum = run_bfoa(
            numeroDeBacterias=30,
            iteraciones=50,
            tumbo=2,
            nado=3,
            max_processes=4
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        results.append({
            'Corrida': run,
            'Fitness': best_fitness,
            'Tiempo (s)': elapsed_time,
            'Interacciones': interacciones,
            'BLOSUM': blosum
        })

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Resultados guardados en {output_file}")

    plt.figure(figsize=(10, 6))
    plt.plot(df['Corrida'], df['Fitness'], marker='o')
    plt.title('Fitness por Corrida')
    plt.xlabel('Corrida')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.savefig('grafica_fitness.png')
    print("Gráfica guardada como grafica_fitness.png")

    fitness_mean = df['Fitness'].mean()
    fitness_max = df['Fitness'].max()
    fitness_min = df['Fitness'].min()

    report_filename = f"tiempo_total_{timestamp_filename}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"Fecha y hora de ejecución: {timestamp}\n")
        f.write(f"Tiempo total de ejecución: {total_elapsed:.2f} segundos\n")
        f.write(f"Fitness promedio: {fitness_mean:.4f}\n")
        f.write(f"Mejor fitness: {fitness_max:.4f}\n")
        f.write(f"Peor fitness: {fitness_min:.4f}\n")
    print(f"Tiempo total, resumen de fitness y fecha guardados en {report_filename}")


if __name__ == "__main__":
    run_multiple_executions()
