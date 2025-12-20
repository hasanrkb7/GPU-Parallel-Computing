import sys
import csv
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_task3.py scalability_task3.csv")
        return 1

    path = sys.argv[1]
    threads = []
    speedup = []
    efficiency = []

    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            threads.append(int(row["threads"]))
            speedup.append(float(row["speedup"]))
            efficiency.append(float(row["efficiency"]))

    # Speedup plot
    plt.figure()
    plt.plot(threads, speedup, marker="o")
    plt.plot(threads, threads, marker=None)  # ideal
    plt.xlabel("Threads")
    plt.ylabel("Speedup (T1 / Tn)")
    plt.grid(True)
    plt.title("Exercise 2 Task 3 — Strong scalability (OpenMP)")
    plt.savefig("scalability_task3.png", dpi=200)
    print("Wrote scalability_task3.png")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
