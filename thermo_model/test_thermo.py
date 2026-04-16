import numpy as np
import cantera as ct
import time
from numba import njit

from thermo import ThermoTableBuilder


# ==========================================
# JIT-Compiled Benchmark Loop
# ==========================================
@njit
def run_numba_loop(engine, T_arr, P_arr):
    """
    By compiling the loop itself, we ensure no Python overhead.
    The execution never leaves C/LLVM memory space.
    """
    h_arr = np.empty(len(T_arr), dtype=np.float64)
    for i in range(len(T_arr)):
        h_arr[i] = engine.state_from_TP(T_arr[i], P_arr[i])[0]
    return h_arr


def run_benchmark(N_points=100_000):
    print(f"--- Setting up benchmark for {N_points:,} state evaluations ---")

    thermo_engine = ThermoTableBuilder.build('air.yaml',
                                             T_range=(200, 2000),
                                             P_range=(10000, 5e6),
                                             grid_size=(500, 500))

    np.random.seed(42)
    T_test = np.random.uniform(250, 1900, N_points)
    P_test = np.random.uniform(50000, 4.5e6, N_points)

    h_cantera = np.zeros(N_points)

    # --- CANTERA BENCHMARK ---
    print("\n--- Running Cantera (C++ Backend) ---")
    fluid = ct.Solution('air.yaml')


    # 1. Generate test data that makes thermodynamic sense
    # (Using the forward engine to get valid entropy values first)
    s_test = np.zeros(N_points)
    for i in range(N_points):
        s_test[i] = thermo_engine.state_from_TP(T_test[i], P_test[i])[1]

    # --- CANTERA BENCHMARK (Reverse Lookup) ---
    print("\n--- Running Cantera (Reverse: s, P -> T) ---")
    start_ct = time.perf_counter()
    for i in range(N_points):
        # Setting SP forces Cantera's internal non-linear solver to iterate
        fluid.SP = s_test[i], P_test[i]
        h_cantera[i] = fluid.T
    time_ct = time.perf_counter() - start_ct
    print(f"Cantera took: {time_ct:.4f} seconds")

    # --- NUMBA BENCHMARK (Reverse Lookup) ---
    print("\n--- Running Numba (Reverse: s, P -> T) ---")

    @njit
    def run_numba_reverse(engine, s_arr, P_arr):
        T_out = np.empty(len(s_arr), dtype=np.float64)
        for i in range(len(s_arr)):
            T_out[i] = engine.T_from_sP(s_arr[i], P_arr[i])
        return T_out

    # Warmup
    _ = run_numba_reverse(thermo_engine, np.array([s_test[0]]), np.array([P_test[0]]))

    start_fast = time.perf_counter()
    T_fast = run_numba_reverse(thermo_engine, s_test, P_test)
    time_fast = time.perf_counter() - start_fast
    print(f"Numba took  : {time_fast:.6f} seconds")

    print(f"\nSPEEDUP: {time_ct / time_fast:,.1f}x faster")


if __name__ == "__main__":
    run_benchmark(N_points=100_000)