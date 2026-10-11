"""CPU-only reference: fixed-K pedestrian selection by action-space risk coverage.

Research prototype, NOT a validated social-navigation method. Does not modify
or require the user's navigation project. Candidate human motion: CV for up to 1s;
robot candidate action: constant velocity for selection only. Selection is one
shared set of identities for the current observation and all 80 action queries.
"""
from __future__ import annotations
import numpy as np


def ttc5_indices(robot_xy, robot_vxy, humans, k=5):
    """Reference selector from reported TTC5 rule; returns original-order IDs."""
    H = np.asarray(humans, dtype=np.float64).reshape(-1, 5)
    if len(H) <= k:
        return list(range(len(H)))
    p = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    v = np.asarray(robot_vxy, dtype=np.float64).reshape(2)
    rel = H[:, :2] - p
    relv = H[:, 2:4] - v
    distance = np.sqrt(np.sum(rel**2, axis=1) + 1e-6)
    closing = -np.sum(rel*relv, axis=1)/(distance + 1e-6)
    score = np.where(closing > .1, distance/(closing + 1e-6), distance*10.)
    return sorted(np.argsort(score, kind='stable')[:k].tolist())


def select_action_coverage(robot_xy, robot_vxy, robot_radius, humans, commands,
                           k=5, dt=0.25, horizon=1.0,
                           filter_margin=0.2, social_margin=0.2):
    """Return indices of up to k humans, in original observation order.

    humans: N x 5 array [px,py,vx,vy,radius]
    commands: A x 2 candidate executed Cartesian velocities (A == 80)
    No privileged future, IDs, or pedestrian goals are used.
    """
    p = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    H = np.asarray(humans, dtype=np.float64).reshape(-1, 5)
    U = np.asarray(commands, dtype=np.float64).reshape(-1, 2)
    if U.shape != (80, 2):
        raise ValueError(f"Expected exactly 80 commands; got {U.shape}")
    n = len(H)
    if n <= k:
        return list(range(n))  # bitwise-equivalent path must bypass selector
    fallback = ttc5_indices(p, robot_vxy, H, k)
    if k < 1 or not (dt > 0 and horizon >= dt) or social_margin <= 0:
        raise ValueError("invalid selector configuration")
    if not (np.isfinite(H).all() and np.isfinite(U).all()
            and np.isfinite(p).all()):
        raise ValueError("nonfinite state")

    # Relative motion: pedestrians are CV; robot command U is held fixed in
    # this analytic selector, but is NOT committed to the real controller.
    r0 = H[None, :, :2] - p[None, None, :]
    vrel = H[None, :, 2:4] - U[:, None, :]
    r2 = np.broadcast_to(r0, vrel.shape)
    vv = np.sum(vrel * vrel, axis=2)
    rv = np.sum(r2 * vrel, axis=2)
    tstar = np.clip(-rv / np.maximum(vv, 1e-12), 0.0, horizon)
    dmin = np.linalg.norm(r2 + vrel * tstar[:, :, None], axis=2)
    clearance_1s = dmin - robot_radius - H[None, :, 4]

    # Use full-crowd 0.25 s end-point safety solely to exclude clearly masked
    # actions. Mirrors parent endpoint geometry, not a certified swept shield.
    endpoint = r2 + vrel * dt
    clearance_dt = np.linalg.norm(endpoint, axis=2) - robot_radius - H[None, :, 4]
    feasible = np.min(clearance_dt, axis=1) >= filter_margin
    active_actions = np.flatnonzero(feasible)
    if len(active_actions) == 0:
        # all-unsafe is outside this method; preserve original TTC5 policy.
        return fallback

    # Threat of individual human i to a candidate action a during 1 s:
    # 0 beyond social_margin; increases smoothly up to 1 at overlap.
    risk = np.clip((social_margin - clearance_1s) / social_margin, 0., 1.)
    risk = risk[active_actions]
    # Exclude pedestrian danger that is common to every feasible action.
    # The network needs information that can distinguish candidate actions;
    # full-crowd collision screening separately retains absolute safety.
    risk = np.maximum(0., risk - np.min(risk, axis=0, keepdims=True))
    cover = np.zeros(len(active_actions), dtype=np.float64)
    chosen = []
    for _ in range(min(k, n)):
        # Diminishing-returns coverage across distinct candidate actions.
        best_id, best_gain = None, -np.inf
        base = np.minimum(1., cover)
        for i in range(n):
            if i in chosen:
                continue
            gain = float((np.minimum(1., cover + risk[:, i]) - base).sum())
            if gain > best_gain + 1e-12:
                best_id, best_gain = i, gain
        if best_id is None or best_gain <= 1e-12:
            break
        chosen.append(best_id)
        cover += risk[:, best_id]
    chosen += [i for i in fallback if i not in chosen][:k-len(chosen)]
    return sorted(chosen)


def test_reference():
    theta = np.linspace(0., 2. * np.pi, 16, endpoint=False)
    speed = [0.1289, 0.2862, 0.4785, 0.7132, 1.0]
    commands = np.array([[v*np.cos(t), v*np.sin(t)] for t in theta for v in speed])
    assert commands.shape == (80, 2)
    H = np.array([
        [.8, .2, 0., 0., .25], [1.1, -.4, 0., .1, .25],
        [1.2, .7, -.1, 0., .25], [1.4, -.7, .1, 0., .25],
        [1.7, 0., -.3, 0., .25], [2.2, .2, -.9, 0., .25],
        [-.8, 1.2, .2, -.3, .25], [-.7, -.9, 0., .2, .25]])
    assert select_action_coverage([0, 0], [0, 0], .3, H[:5], commands, 5) == list(range(5))
    ids = select_action_coverage([0, 0], [0, 0], .3, H, commands, 5)
    assert len(ids) == 5 and len(set(ids)) == 5 and ids == sorted(ids)
    for _ in range(2):
        assert ids == select_action_coverage([0, 0], [0, 0], .3, H, commands, 5)
    print('CPU_REFERENCE_PASS', 'n_humans=', len(H), 'selected=', ids)


if __name__ == '__main__':
    test_reference()
