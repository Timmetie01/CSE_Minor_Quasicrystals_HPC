import time
from unittest.mock import patch
import numpy as np
import pyvista as pv
import vtk
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from src.main_code.visuals.visualization import plot_cluster
from src.main_code.local_optimizers.bfgs import LBFGS
from data.data_handling import read_LJwales_data
from src.main_code.potentials_calculations import cluster_potential


# =====================================================================
# 1. GENERAL BASE FUNCTIONS
# =====================================================================

def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Computes the optimal 3D rotation matrix R (det = +1) that aligns coordinate
    set Q onto reference set P with minimal RMSD, preventing frame-to-frame
    spinning and mirror-image reflections.

    Assumes P and Q share the same origin (e.g., center of mass or a pinned
    anchor atom at (0, 0, 0)) and that row i in P corresponds to row i in Q.

    :param P: ndarray, shape (N, 3)
        Reference 3D coordinates from the previous frame.
    :param Q: ndarray, shape (N, 3)
        Target 3D coordinates from the next frame to be rotated onto P.
    :returns: ndarray, shape (3, 3)
        Proper orthogonal rotation matrix R satisfying R.T @ R = I and det(R) = +1,
        applied to row-vector coordinates via Q @ R.T.
    """
    H = Q.T @ P
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def match_and_align(
    coords_prev: np.ndarray,
    coords_next: np.ndarray,
    prev_anchor: int,
    match_indices: bool = True,
    pin_atom: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Aligns `coords_next` onto `coords_prev` in 3D space while tracking the same
    physical anchor atom across frames, even when atom indices are reordered
    or new atoms are added.

    :param coords_prev: ndarray, shape (N_prev, 3)
        Already-aligned 3D coordinates from the previous animation frame.
    :param coords_next: ndarray, shape (N_next, 3)
        Raw 3D coordinates for the upcoming frame (N_next >= N_prev).
    :param prev_anchor: int
        Row index of the tracked anchor atom inside `coords_prev`.
    :param match_indices: bool, default True
        If True, solves a 1-to-1 atom assignment (via candidate rotations and
        the Hungarian algorithm, using `linear_sum_assignment`) before rotating, 
        handling re-indexed or growing clusters. If False, assumes row i in 
        `coords_next` is already the same physical atom as row i in `coords_prev`.
    :param pin_atom: bool, default True
        If True, shifts coordinates so the tracked anchor atom stays fixed at
        (0, 0, 0). If False, centers the cluster's center of mass at (0, 0, 0).
    :returns: tuple[ndarray, int]
        aligned_coords_next : ndarray, shape (N_next, 3)
            `coords_next` shifted to the chosen origin and rotated to match
            `coords_prev`.
        next_anchor : int
            Row index of the tracked physical anchor atom inside `coords_next`.
    """
    
    # Fixed-N evolution: atom indices are already in 1-to-1 correspondence
    if not match_indices:
        next_anchor = prev_anchor
        if pin_atom:
            p_ref = coords_prev - coords_prev[prev_anchor]
            q_ref = coords_next - coords_next[next_anchor]
            R = kabsch_rotation(p_ref, q_ref)
            return q_ref @ R.T, next_anchor
        else:
            p_com = coords_prev - coords_prev.mean(axis=0)
            q_com = coords_next - coords_next.mean(axis=0)
            R = kabsch_rotation(p_com, q_com)
            return q_com @ R.T, next_anchor


    # Scrambled-N growth: atom indices are not in 1-to-1 correspondence yet

    # 1: Center both clusters at their centre of mass to find the rotation-invariant atom matching
    p_com = coords_prev - coords_prev.mean(axis=0)
    q_com = coords_next - coords_next.mean(axis=0)

    # 2: Generate candidate initial rotations (4 axis alignments + 24 evenly spread 3D angles)
    _, _, Vt_p = np.linalg.svd(p_com, full_matrices=False)
    _, _, Vt_q = np.linalg.svd(q_com, full_matrices=False)
    if Vt_p.shape[0] < 3:
        Vt_p = np.eye(3)
    if Vt_q.shape[0] < 3:
        Vt_q = np.eye(3)

    sign_flips = [
        np.diag([1, 1, 1]),
        np.diag([1, -1, -1]),
        np.diag([-1, 1, -1]),
        np.diag([-1, -1, 1]),
    ]
    candidates = [Vt_p.T @ S @ Vt_q for S in sign_flips]
    candidates = [R if np.linalg.det(R) > 0 else R @ np.diag([1, 1, -1]) for R in candidates]

    rng = np.random.default_rng(42)
    candidates.extend(Rotation.random(24, random_state=rng).as_matrix())

    # 3: Test candidate initial rotations and keep the best
    best_msd = np.inf
    best_row, best_col = None, None

    for R_init in candidates:
        dist_matrix = cdist(p_com, q_com @ R_init.T)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        R_ref = kabsch_rotation(p_com[row_ind], q_com[col_ind])
        diff = p_com[row_ind] - (q_com[col_ind] @ R_ref.T)
        msd = np.mean(np.sum(diff ** 2, axis=1))
        if msd < best_msd:
            best_msd = msd
            best_row, best_col = row_ind, col_ind

    # 4: Return aligned coordinates and anchor
    anchor_pos = np.where(best_row == prev_anchor)[0][0]
    next_anchor = int(best_col[anchor_pos])

    if pin_atom:
        p_pinned = coords_prev - coords_prev[prev_anchor]
        q_pinned = coords_next - coords_next[next_anchor]
        R_final = kabsch_rotation(p_pinned[best_row], q_pinned[best_col])
        return q_pinned @ R_final.T, next_anchor
    else:
        R_final = kabsch_rotation(p_com[best_row], q_com[best_col])
        return q_com @ R_final.T, next_anchor


def compute_trajectory_bounds(
    aligned: list[np.ndarray],
    sphere_padding: float = 0.75,
) -> tuple[float, float, float, float, float, float]:
    """
    Computes padded global limits covering every atom across all 
    animation frames so the 3D frame stays fixed without clipping.

    :param aligned: list[ndarray]
        Sequence of coordinate arrays, each of shape (N, 3), representing
        aligned atom positions per frame (N may vary across frames).
    :param sphere_padding: float, default 0.75
        Margin added beyond the extreme atom centers along each axis (typically
        at least the rendered atom sphere radius so spheres do not clip at the
        frame edges).
    :returns: tuple[float, float, float, float, float, float]
        Padded bounding limits formatted as (xmin, xmax, ymin, ymax, zmin, zmax).
    """
    all_points = np.vstack(aligned)
    mins = all_points.min(axis=0) - sphere_padding
    maxs = all_points.max(axis=0) + sphere_padding
    return (
        float(mins[0]), float(maxs[0]),
        float(mins[1]), float(maxs[1]),
        float(mins[2]), float(maxs[2]),
    )


def subsample_indices(n_total: int, step: int) -> list[int]:
    """
    Generates evenly spaced frame indices `[0, step, 2*step, ...]` for
    downsampling a trajectory, always preserving the first and final frames.

    :param n_total: int
        Total number of frames in the trajectory.
    :param step: int
        Sampling stride between kept frames (must be >= 1; 1 keeps every frame).
    :returns: list[int]
        Sorted frame indices starting at the index of the first frame and ending 
        at the index of the final frame.
    """
    if step < 1:
        raise ValueError("step must be >= 1.")
    indices = list(range(0, n_total, step))
    if indices[-1] != n_total - 1:
        indices.append(n_total - 1)
    return indices


def animate_cluster(
    coords_list: list[np.ndarray],
    labels: list[str] | None = None,
    frame_delay: float = 0.7,
    gif_path: str | None = None,
    match_indices: bool = True,
    pin_atom: bool = True
):
    """
    Animates a sequence of atomic cluster configurations either in an interactive
    PyVista window or by exporting directly to an animated GIF file.

    Wraps the single-frame `plot_cluster` function by intercepting its `pv.Plotter`
    instantiation and `.show()` calls, reusing a single shared plotter across all
    aligned frames to maintain consistent camera bounds, HUD overlays, and responsive
    playback controls.

    :param coords_list: list[np.ndarray]
        Ordered sequence of atomic coordinate arrays, each of shape (N_i, 3).
    :param labels: list[str] | None, optional
        Display title for each frame shown in the upper-left HUD overlay. Defaults to
        `["N = <atom_count>", ...]` when None.
    :param frame_delay: float, optional
        Target delay in seconds between consecutive frames during playback (and inverse
        frame rate for GIF export). Defaults to 0.7.
    :param gif_path: str | None, optional
        Destination file path for GIF export. When None, opens an interactive GUI window 
        instead. Defaults to None.
    :param match_indices: bool, optional
        If True, matches atoms across frames with changing counts (cluster growth).
        If False, aligns frames assuming fixed atom indexing (cluster evolution).
        Defaults to True.
    :param pin_atom: bool, optional
        If True, anchors one physical reference atom at the origin (0, 0, 0) across
        all frames. If False, centers each frame's center of mass at (0, 0, 0).
        Defaults to True.

    Controls (Interactive Mode):
        - Space                 : [ > ] / [ I I ]       (Play / Pause, or replay from frame 0 when finished)
        - Left / Right arrows   : < / >                 (Pause and step backward / forward one frame)
        - B                     : I<                    (Pause and jump back to the first frame)
        - Q / Esc / Window Close: X                     (Exit cleanly at any time)
    """
    
    # 0: Validation
    if len(coords_list) == 0:
        raise ValueError("coords_list cannot be empty.")

    if labels is None:
        labels = [f"N = {len(c)}" for c in coords_list]

    # 1: Picking initial anchor atom (central atom of the final frame for evolution, atom 0 for growth)
    if not match_indices:
        final_c = coords_list[-1]
        anchor_idx = int(np.argmin(np.linalg.norm(final_c - final_c.mean(axis=0), axis=1)))
    else:
        anchor_idx = 0

    # 2: Aligning frames and computing global plotting bounds
    first_frame = (coords_list[0] - coords_list[0][anchor_idx] if pin_atom else coords_list[0] - coords_list[0].mean(axis=0)    )
    aligned = [first_frame]

    for nxt in coords_list[1:]:
        aligned_nxt, anchor_idx = match_and_align(
            aligned[-1],
            nxt,
            prev_anchor=anchor_idx,
            match_indices=match_indices,
            pin_atom=pin_atom,
        )
        aligned.append(aligned_nxt)

    n_frames = len(aligned)
    trajectory_bounds = compute_trajectory_bounds(aligned)

    # 3: Initialising PyVista and plotting state
    vtk.vtkObject.GlobalWarningDisplayOff()
    shared_plotter = pv.Plotter()
    real_show = shared_plotter.show

    state = {
        "idx": 0,
        "paused": False,
        "finished": False,
        "needs_redraw": True,
        "closed": False,
    }

    # 4: Defining plotter-helper and user-control functions
    def is_alive() -> bool:
        """
        Checks whether the PyVista render window is still open and safe to update.

        :returns: bool
            True if the plotter window is active and valid, False if closed or destroyed.
        """
        return (
            not state["closed"]
            and not getattr(shared_plotter, "_closed", False)
            and getattr(shared_plotter, "render_window", None) is not None
        )

    def on_close():
        """
        Marks the animation window as closed when triggered by a keyboard exit shortcut.
        """
        state["closed"] = True

    def on_exit_event(_caller, _event):
        """
        Marks the animation window as closed when VTK fires an ExitEvent.

        :param _caller: vtkObject
            VTK interactor object that emitted the event.
        :param _event: str
            Name of the VTK event.
        """
        state["closed"] = True

    def clear_3d_actors():
        """
        Removes all 3D actors from the plotter renderer while preserving 2D HUD overlays.

        """
        if not is_alive():
            return
        for actor in list(shared_plotter.renderer.GetActors()):
            shared_plotter.remove_actor(actor, render=False)

    def update_hud():
        """
        Updates the top-left frame label and top-right playback status indicator on the HUD.

        """
        if not is_alive():
            return
        label = labels[state["idx"]]
        if state["finished"] and state["paused"]:
            state_icon = " >I "
        elif state["paused"]:
            state_icon = " I I "
        else:
            state_icon = " > "

        shared_plotter.add_text(
            label,
            position="upper_left",
            name="hud_title",
            font_size=13,
        )
        shared_plotter.add_text(
            f"[{state_icon}]",
            position="upper_right",
            name="hud_status",
            font_size=13,
        )

    def toggle_pause():
        """
        Toggles playback between playing and paused, or restarts from frame 0 if finished.

        """
        if state["finished"] and state["idx"] == n_frames - 1:
            # Restart from the beginning if user presses Space after finishing
            state["idx"] = 0
            state["finished"] = False
            state["paused"] = False
            state["needs_redraw"] = True
        else:
            state["paused"] = not state["paused"]
            state["finished"] = False
            update_hud()

    def step_prev():
        """
        Pauses playback and steps backward by one frame if not already at frame 0.

        """
        state["paused"] = True
        state["finished"] = False
        if state["idx"] > 0:
            state["idx"] -= 1
            state["needs_redraw"] = True
        else:
            update_hud()

    def step_next():
        """
        Pauses playback and advances forward by one frame if not already at the final frame.

        """
        state["paused"] = True
        if state["idx"] < n_frames - 1:
            state["idx"] += 1
            state["needs_redraw"] = True
        else:
            state["finished"] = True
            update_hud()

    def reset_first():
        """
        Resets the animation to frame 0 in a paused state.

        """
        state["idx"] = 0
        state["paused"] = True
        state["finished"] = False
        state["needs_redraw"] = True

    # 5: Main animation
    with patch("pyvista.Plotter", return_value=shared_plotter), patch.object(shared_plotter, "show"):
        # 5.1: Saving animation as GIF
        if gif_path:
            shared_plotter.open_gif(gif_path, fps=max(1, int(round(1.0 / frame_delay))))
            for idx, coords in enumerate(aligned):
                state["idx"] = idx
                with patch.object(shared_plotter, "render"):
                    clear_3d_actors()
                    shared_plotter.add_text(
                        labels[idx],
                        position="upper_left",
                        name="hud_title",
                        font_size=13,
                    )
                    plot_cluster(coords)
                    if idx == 0:
                        shared_plotter.reset_camera(bounds=trajectory_bounds)
                    shared_plotter.renderer.ResetCameraClippingRange()
                shared_plotter.write_frame()
        
        # 5.2: Showing animation in interactive window
        else:
            # 5.2.1: Pre-rendering frame 0 and static bottom controls before opening the window, and adding key-events
            with patch.object(shared_plotter, "render"):
                shared_plotter.add_text(
                    "I< [B]    < [Left]    I I / > [Space]    > [Right]    X [Q/Esc]",
                    position="lower_left",
                    name="hud_controls",
                    font_size=10,
                    color="gray",
                )
                update_hud()
                plot_cluster(aligned[0])
                shared_plotter.reset_camera(bounds=trajectory_bounds)
                shared_plotter.renderer.ResetCameraClippingRange()
            state["needs_redraw"] = False

            real_show(auto_close=False, interactive_update=True)

            if is_alive():                                                  #Re-applying trajectory bounds once the OS 
                shared_plotter.reset_camera(bounds=trajectory_bounds)       #window has its final viewport size (finetuning)
                shared_plotter.renderer.ResetCameraClippingRange()
                shared_plotter.render()

            if getattr(shared_plotter, "iren", None) is not None:
                shared_plotter.iren.add_observer("ExitEvent", on_exit_event)

            shared_plotter.add_key_event("space", toggle_pause)
            shared_plotter.add_key_event("Left", step_prev)
            shared_plotter.add_key_event("Right", step_next)
            shared_plotter.add_key_event("b", reset_first)
            shared_plotter.add_key_event("q", on_close)
            shared_plotter.add_key_event("Escape", on_close)

            # 5.2.2: Updating frames
            elapsed = 0.0
            step = 0.02
            while is_alive():
                if state["needs_redraw"]:
                    with patch.object(shared_plotter, "render"):
                        clear_3d_actors()
                        update_hud()
                        plot_cluster(aligned[state["idx"]])
                        if is_alive():
                            shared_plotter.renderer.ResetCameraClippingRange()
                    state["needs_redraw"] = False
                    elapsed = 0.0

                try:
                    if is_alive():
                        shared_plotter.update()
                except Exception:
                    state["closed"] = True
                    break

                if not is_alive():
                    break

                time.sleep(step)

                if not state["paused"]:
                    elapsed += step
                    if elapsed >= frame_delay:
                        elapsed = 0.0
                        if state["idx"] < n_frames - 1:
                            state["idx"] += 1
                            state["needs_redraw"] = True
                        else:
                            state["paused"] = True
                            state["finished"] = True
                            update_hud()

    # 6: Cleaning up and closing the plotter if necessary
    try:
        if not getattr(shared_plotter, "_closed", False):
            shared_plotter.close()
    except Exception:
        pass


# =====================================================================
# 2. SPECIALIZED ANIMATION FUNCTIONS (INPUT: PRE-COMPUTED COORDINATES)
# =====================================================================

def animate_cluster_growth(
    coords_list: list[np.ndarray],
    frame_delay: float = 0.7,
    gif_path: str | None = None,
    pin_atom: bool = True,
    step: int = 1
):
    """
    Animates the growth of an atomic cluster across configurations of increasing
    atom count (matching atoms across frames even when atom indexing changes) in 
    either an interactive PyVista window or by exporting directly to an animated 
    GIF file.

    :param coords_list: list[np.ndarray]
        Ordered sequence of coordinate arrays, each of shape (N_i, 3) with
        non-decreasing atom counts.
    :param frame_delay: float, optional
        Target delay in seconds between consecutive frames during playback (and inverse
        frame rate for GIF export). Defaults to 0.7.
    :param gif_path: str | None, optional
        Destination file path for GIF export. When None, opens an interactive GUI
        window instead. Defaults to None.
    :param pin_atom: bool, optional
        If True, anchors one physical reference atom at the origin (0, 0, 0) across
        all frames. If False, centers each frame's center of mass at (0, 0, 0).
        Defaults to True.
    :param step: int, optional
        Sampling stride between kept cluster sizes (must be >= 1; 1 shows every
        cluster size, k shows every k'th cluster size, always preserving the first 
        and final frames). Defaults to 1.

    """
    all_coords = list(coords_list)
    if len(all_coords) == 0:
        raise ValueError("coords_list cannot be empty.")
    indices = subsample_indices(len(all_coords), step)
    sampled_coords = [all_coords[i] for i in indices]
    labels = [f"N = {len(coords)}" for coords in sampled_coords]

    animate_cluster(
        sampled_coords,
        labels=labels,
        frame_delay=frame_delay,
        gif_path=gif_path,
        match_indices=True,
        pin_atom=pin_atom
    )


def animate_cluster_evolution(
    coords_list: list[np.ndarray] | np.ndarray,
    frame_delay: float = 0.4,
    gif_path: str | None = None,
    pin_atom: bool = True,
    step: int = 1
):
    """
    Animates the structural evolution (e.g., intermediate optimisation steps) of a
    fixed-size atomic cluster across coordinate snapshots in either an interactive 
    PyVista window or by exporting directly to an animated GIF file.

    :param coords_list: list[np.ndarray] | np.ndarray
        Ordered sequence of coordinate arrays, each of shape (N, 3), or a single
        3D array of shape (S, N, 3) for S optimization steps and N atoms.
    :param frame_delay: float, optional
        Target delay in seconds between consecutive frames during playback (and inverse
        frame rate for GIF export). Defaults to 0.4.
    :param gif_path: str | None, optional
        Destination file path for GIF export. When None, opens an interactive GUI
        window instead. Defaults to None.
    :param pin_atom: bool, optional
        If True, anchors the central reference atom at the origin (0, 0, 0) across
        all frames. If False, centers each frame's center of mass at (0, 0, 0).
        Defaults to True.
    :param step: int, optional
        Sampling stride between kept (optimisation) steps (must be >= 1; 1 shows every 
        step, k shows every k'th step, always preserving the first and final frames). 
        Defaults to 1.

    """
    all_steps = list(coords_list)
    if len(all_steps) == 0:
        raise ValueError("coords_list cannot be empty.")
    total_steps = len(all_steps)
    indices = subsample_indices(total_steps, step)
    sampled_steps = [all_steps[i] for i in indices]

    n_atoms = len(sampled_steps[0])
    if any(len(s) != n_atoms for s in sampled_steps):
        raise ValueError("All snapshots in coords_list must have the same number of atoms.")
    labels = [f"N = {n_atoms}  (Step {i + 1}/{total_steps})" for i in indices]

    animate_cluster(
        sampled_steps,
        labels=labels,
        frame_delay=frame_delay,
        gif_path=gif_path,
        match_indices=False,
        pin_atom=pin_atom
    )


# =====================================================================
# 3. EXAMPLE USAGE
# =====================================================================

if __name__ == "__main__":
    # --- 1. Growth animation ---
    growth_trajectory = []
    for i in range(1, 14):
        x = read_LJwales_data(i)
        final_coords = LBFGS(x)
        E = cluster_potential(final_coords)
        print(f'Total potential for {i} atoms: {E}')
        growth_trajectory.append(final_coords)

    animate_cluster_growth(growth_trajectory, frame_delay=0.7, step=1)


    # --- 2. Evolution animation (intermediate optimization steps) ---
    x_init = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 1.0, 1.0],
        [4.0, 1.0, 0.0],
    ])
    final_coords, intermediate_coords = LBFGS(x_init, keep_steps=True)

    animate_cluster_evolution(intermediate_coords, frame_delay=0.4, step=1)

