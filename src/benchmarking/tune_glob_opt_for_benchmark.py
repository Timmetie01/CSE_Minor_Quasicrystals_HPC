import time as t
import numpy as np
from tqdm import tqdm
from src.main_code.classes import LennardJones, Optimizer
from src.main_code.local_optimizers import bfgs
from data.data_handling import read_LJwales_data


def evaluate_hit_rate(run_fn, N_atoms: int, opt_pot: float, n_eval: int = 20, tol: float = 1e-4):
    """Runs `run_fn()` `n_eval` times and returns (hit_rate, mean_time, std_time)."""
    Potential = LennardJones(1,1)
    times = []
    hits = 0
    for _ in range(n_eval):
        print(_)
        t0 = t.time()
        coords = run_fn(N_atoms, bfgs.lbfgs_njit)
        times.append(t.time() - t0)
        if abs(Potential.potential(coords) - opt_pot) <= tol:
            hits += 1
    return hits / n_eval, float(np.mean(times)), float(np.std(times))

def evaluate_hit_rate(
    run_fn,
    Potential,
    opt_pot: float,
    target_rate: float = 0.95,
    n_eval: int = 20,
    tol: float = 1e-4,
    desc: str = "",
    early_stop: bool = True,
):
    """
    Runs `run_fn()` up to `n_eval` times with a live tqdm progress bar.
    If `early_stop=True`, aborts early as soon as reaching `target_rate` is
    mathematically impossible (saving huge amounts of time on failing guesses).
    """
    times = []
    hits = 0
    min_hits_needed = int(np.ceil(target_rate * n_eval))
    max_allowed_misses = n_eval - min_hits_needed

    pbar = tqdm(range(n_eval), desc=desc, unit="run", leave=False)
    for idx in pbar:
        t0 = t.time()
        coords = run_fn()
        dt = t.time() - t0
        times.append(dt)

        if abs(Potential.potential(coords) - opt_pot) <= tol:
            hits += 1

        completed = idx + 1
        misses = completed - hits
        curr_rate = 100.0 * hits / completed
        pbar.set_postfix(
            hits=f"{hits}/{completed} ({curr_rate:.0f}%)",
            last=f"{dt:.1f}s",
            avg=f"{np.mean(times):.1f}s",
        )

        # Early bailout if we already have too many misses to hit target_rate
        if early_stop and misses > max_allowed_misses:
            pbar.close()
            return hits / completed, float(np.mean(times)), float(np.std(times)), True

    pbar.close()
    return hits / n_eval, float(np.mean(times)), float(np.std(times)), False


def find_95_percent_iters(
    name: str,
    make_runner,
    Potential,
    N_atoms: int,
    opt_pot: float,
    init_iters: int,
    min_iters: int,
    max_iters: int,
    target_rate: float = 0.95,
    n_eval: int = 20,
    bisect_steps: int = 4,
    early_stop: bool = True,
):
    """
    Automatically adjusts iteration count via bracketing + binary search
    until success rate >= target_rate (default 95%), printing live progress.
    """
    t_method_start = t.time()
    print(f"\n{'='*72}")
    print(f"Calibrating {name} | N={N_atoms} | Target >= {target_rate*100:.0f}% ({int(np.ceil(target_rate*n_eval))}/{n_eval} hits)")
    print(f"{'='*72}")

    low_fail = None
    high_pass = None
    best_stats = None

    curr = int(np.clip(init_iters, min_iters, max_iters))
    attempt = 0

    # 1. Bracketing phase
    while True:
        attempt += 1
        rate, mean_t, std_t, aborted = evaluate_hit_rate(
            make_runner(curr),
            Potential=Potential,
            opt_pot=opt_pot,
            target_rate=target_rate,
            n_eval=n_eval,
            desc=f"[Bracket #{attempt}] iters={curr}",
            early_stop=early_stop,
        )
        status = "EARLY-BAIL (FAIL)" if aborted else ("PASS" if rate >= target_rate else "FAIL")
        elapsed = t.time() - t_method_start
        print(
            f"  [Bracket #{attempt}] iters={curr:<8d} | {status:<17s} | "
            f"rate={rate*100:5.1f}% | avg={mean_t:6.2f}s ± {std_t:4.2f}s | elapsed={elapsed/60:5.1f}m"
        )

        if rate >= target_rate:
            high_pass = curr
            best_stats = (curr, rate, mean_t, std_t)
            if curr <= min_iters or (low_fail is not None):
                break
            curr = max(min_iters, curr // 2)
        else:
            low_fail = curr
            if curr >= max_iters or (high_pass is not None):
                break
            curr = min(max_iters, int(curr * 1.6))

    if high_pass is None:
        total_min = (t.time() - t_method_start) / 60
        print(f"  => {name} did not reach {target_rate*100:.0f}% before max_iters={max_iters} (took {total_min:.1f} min).")
        return curr, rate, mean_t, std_t

    # 2. Binary search between low_fail and high_pass to trim excess runtime
    if low_fail is not None:
        for b_step in range(1, bisect_steps + 1):
            if high_pass - low_fail <= max(10, int(0.05 * high_pass)):
                break
            mid = (low_fail + high_pass) // 2
            rate, mean_t, std_t, aborted = evaluate_hit_rate(
                make_runner(mid),
                Potential=Potential,
                opt_pot=opt_pot,
                target_rate=target_rate,
                n_eval=n_eval,
                desc=f"[Bisect {b_step}/{bisect_steps}] iters={mid}",
                early_stop=early_stop,
            )
            status = "EARLY-BAIL (FAIL)" if aborted else ("PASS" if rate >= target_rate else "FAIL")
            elapsed = t.time() - t_method_start
            print(
                f"  [Bisect {b_step}/{bisect_steps}] iters={mid:<8d} | {status:<17s} | "
                f"rate={rate*100:5.1f}% | avg={mean_t:6.2f}s ± {std_t:4.2f}s | elapsed={elapsed/60:5.1f}m"
            )

            if rate >= target_rate:
                high_pass = mid
                best_stats = (mid, rate, mean_t, std_t)
            else:
                low_fail = mid

    opt_iters, final_rate, final_t, final_std = best_stats
    total_min = (t.time() - t_method_start) / 60
    print(
        f"  => DONE {name}: iters={opt_iters} | hit_rate={final_rate*100:.1f}% | "
        f"avg_time={final_t:.2f}s ± {final_std:.2f}s | total_elapsed={total_min:.1f} min"
    )
    return best_stats


if __name__ == "__main__":
    N_atoms = 50
    Potential = LennardJones(1, 1)
    Opt = Optimizer(Potential)
    opt_pot = Potential.potential(read_LJwales_data(N_atoms))

    print("Compiling Numba JIT functions...")
    t_jit = t.time()
    # NOTE: Replace 'n_iter', 'n_steps', 'generations' with your exact Optimizer kwargs!
    Opt.basin_hopping(N_atoms, bfgs.lbfgs_njit, max_iter=5)
    Opt.genetic_algorithm(N_atoms, number_of_iterations=5)
    Opt.parallel_tempering_numba(N_atoms, N_sweeps=10)
    print(f"JIT compilation finished in {t.time() - t_jit:.1f}s.")

    t_all = t.time()
    results = {}

    # results["Basin Hopping"] = find_95_percent_iters(
    #     name="Basin Hopping",
    #     make_runner=lambda it: lambda: Opt.basin_hopping(N_atoms, bfgs.lbfgs_njit, max_iter=it),
    #     Potential=Potential,
    #     N_atoms=N_atoms,
    #     opt_pot=opt_pot,
    #     init_iters=5_000,
    #     min_iters=500,
    #     max_iters=25_000,
    #     n_eval=20,
    # )

    # results["Parallel Tempering (Numba)"] = find_95_percent_iters(
    #     name="Parallel Tempering (Numba)",
    #     make_runner=lambda it: lambda: Opt.parallel_tempering_numba(N_atoms, N_sweeps=it),
    #     Potential=Potential,
    #     N_atoms=N_atoms,
    #     opt_pot=opt_pot,
    #     init_iters=100_000,
    #     min_iters=10_000,
    #     max_iters=400_000,
    #     n_eval=20,
    # )

    results["Genetic Algorithm"] = find_95_percent_iters(
        name="Genetic Algorithm",
        make_runner=lambda it: lambda: Opt.genetic_algorithm(N_atoms, max_attempts=it, Pop_size=50),
        Potential=Potential,
        N_atoms=N_atoms,
        opt_pot=opt_pot,
        init_iters=600,
        min_iters=100,
        max_iters=10_000,
        n_eval=20,
    )

    print(f"\n{'='*72}")
    print(f"FINAL CALIBRATION SUMMARY (N={N_atoms}, total time: {(t.time() - t_all)/60:.1f} min)")
    print(f"{'='*72}")
    for method_name, (iters, rate, mean_t, std_t) in results.items():
        print(f"  {method_name:<28s} | iters={iters:<8d} | rate={rate*100:5.1f}% | runtime={mean_t:6.2f}s ± {std_t:4.2f}s")


#iters for 50 atoms: 
# Basin-hopping: 4687
# Parallel Tempering (JIT): 256000