"""
SIDARTHE Model for COVID-19 Epidemic Simulation
================================================
Extended compartmental model based on:
  Giordano et al., "Modelling the COVID-19 epidemic and implementation of
  population-wide interventions in Italy", Nature Medicine, 2020.

Numerically solved using the Runge-Kutta 4th Order (RK4) method.

Author : Arka Ghosh
Course : P346 Computational Physics, NISER Bhubaneswar
Guide  : Dr. Colin Benjamin
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# ODE System
# ---------------------------------------------------------------------------

def sidarthe(t, y, params):
    """
    Returns the time derivatives of all 8 SIDARTHE compartments.

    Compartments
    ------------
    S : Susceptible
    I : Infected (asymptomatic, undetected)
    D : Diagnosed (asymptomatic, detected)
    A : Ailing (symptomatic, undetected)
    R : Recognized (symptomatic, detected)
    T : Threatened (acutely symptomatic, detected)
    H : Healed
    E : Extinct
    """
    S, I, D, A, R, T, H, E = y

    alpha   = params['alpha']
    beta    = params['beta']
    gamma   = params['gamma']
    delta   = params['delta']
    epsilon = params['epsilon']
    zeta    = params['zeta']
    lambda_ = params['lambda_']
    theta   = params['theta']
    eta     = params['eta']
    rho     = params['rho']
    kappa   = params['kappa']
    mu      = params['mu']
    nu      = params['nu']
    xi      = params['xi']
    sigma   = params['sigma']
    tau     = params['tau']

    dSdt = -S * (alpha*I + beta*D + gamma*A + delta*R)
    dIdt =  S * (alpha*I + beta*D + gamma*A + delta*R) - (epsilon + zeta + lambda_)*I
    dDdt =  epsilon*I - (eta + rho)*D
    dAdt =  zeta*I   - (theta + mu + kappa)*A
    dRdt =  eta*D + theta*A - (nu + xi)*R
    dTdt =  mu*A  + nu*R   - (sigma + tau)*T
    dHdt =  lambda_*I + rho*D + kappa*A + xi*R + sigma*T
    dEdt =  tau*T

    return np.array([dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt])


# ---------------------------------------------------------------------------
# RK4 Solver
# ---------------------------------------------------------------------------

def rk4_step(f, t, y, dt, params):
    k1 = f(t,            y,              params)
    k2 = f(t + 0.5*dt,   y + 0.5*dt*k1, params)
    k3 = f(t + 0.5*dt,   y + 0.5*dt*k2, params)
    k4 = f(t + dt,       y + dt*k3,     params)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


def rk4(f, y0, t, params):
    y  = np.zeros((len(t), len(y0)))
    y[0] = y0
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        y[i] = rk4_step(f, t[i-1], y[i-1], dt, params)
    return y


# ---------------------------------------------------------------------------
# Basic Reproduction Number
# ---------------------------------------------------------------------------

def compute_R0(params):
    """
    R0 for the SIDARTHE model (Giordano et al., 2020).

        R0 = (1/r1) * [alpha + beta*eps/r2 + gamma*zeta/r3
                        + delta*eta*eps/(r2*r4) + delta*zeta*theta/(r3*r4)]

    where r1 = eps+zeta+lambda, r2 = eta+rho, r3 = theta+mu+kappa, r4 = nu+xi
    """
    p = params
    r1 = p['epsilon'] + p['zeta']  + p['lambda_']
    r2 = p['eta']     + p['rho']
    r3 = p['theta']   + p['mu']    + p['kappa']
    r4 = p['nu']      + p['xi']

    R0 = (1/r1) * (
        p['alpha']
        + p['beta']  * p['epsilon'] / r2
        + p['gamma'] * p['zeta']    / r3
        + p['delta'] * p['eta'] * p['epsilon'] / (r2 * r4)
        + p['delta'] * p['zeta'] * p['theta']  / (r3 * r4)
    )
    return R0


# ---------------------------------------------------------------------------
# Parameter Sets  (one per intervention phase)
# ---------------------------------------------------------------------------

SCENARIOS = {
    "Initial (no intervention)": {
        'alpha': 0.570, 'beta': 0.011, 'gamma': 0.456, 'delta': 0.011,
        'epsilon': 0.171, 'zeta': 0.125, 'lambda_': 0.034, 'theta': 0.371,
        'eta': 0.125, 'rho': 0.034, 'kappa': 0.017, 'mu': 0.017,
        'nu': 0.027, 'tau': 0.01, 'xi': 0.034, 'sigma': 0.017,
    },
    "Day 4 – Social Distancing": {
        'alpha': 0.422, 'beta': 0.0057, 'gamma': 0.285, 'delta': 0.0057,
        'epsilon': 0.171, 'zeta': 0.125, 'lambda_': 0.034, 'theta': 0.371,
        'eta': 0.125, 'rho': 0.034, 'kappa': 0.017, 'mu': 0.017,
        'nu': 0.027, 'tau': 0.01, 'xi': 0.034, 'sigma': 0.017,
    },
    "Day 12 – Complete Lockdown": {
        'alpha': 0.422, 'beta': 0.0057, 'gamma': 0.285, 'delta': 0.0057,
        'epsilon': 0.143, 'zeta': 0.125, 'lambda_': 0.034, 'theta': 0.371,
        'eta': 0.125, 'rho': 0.034, 'kappa': 0.017, 'mu': 0.017,
        'nu': 0.027, 'tau': 0.01, 'xi': 0.034, 'sigma': 0.017,
    },
    "Day 22 – Partial Lockdown": {
        'alpha': 0.360, 'beta': 0.005, 'gamma': 0.200, 'delta': 0.005,
        'epsilon': 0.143, 'zeta': 0.125, 'lambda_': 0.034, 'theta': 0.371,
        'eta': 0.034, 'rho': 0.034, 'kappa': 0.017, 'mu': 0.008,
        'nu': 0.015, 'tau': 0.01, 'xi': 0.034, 'sigma': 0.017,
    },
    "Day 28 – Full Lockdown": {
        'alpha': 0.210, 'beta': 0.005, 'gamma': 0.110, 'delta': 0.005,
        'epsilon': 0.143, 'zeta': 0.125, 'lambda_': 0.034, 'theta': 0.371,
        'eta': 0.034, 'rho': 0.034, 'kappa': 0.017, 'mu': 0.008,
        'nu': 0.015, 'tau': 0.01, 'xi': 0.034, 'sigma': 0.017,
    },
    "Day 38 – Wider Testing Campaign": {
        'alpha': 0.210, 'beta': 0.005, 'gamma': 0.110, 'delta': 0.005,
        'epsilon': 0.200, 'zeta': 0.025, 'lambda_': 0.034, 'theta': 0.371,
        'eta': 0.025, 'rho': 0.020, 'kappa': 0.020, 'mu': 0.008,
        'nu': 0.015, 'tau': 0.01, 'xi': 0.020, 'sigma': 0.010,
    },
}

LABELS   = ['Susceptible (S)', 'Infected (I)', 'Diagnosed (D)', 'Ailing (A)',
            'Recognized (R)', 'Threatened (T)', 'Healed (H)', 'Extinct (E)']
COLORS   = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
            'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run_simulation(save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)

    # Initial conditions  (Italy, ~60 M population, fractional)
    I0, D0, A0, R0_0, T0, H0, E0 = (200/60e6, 20/60e6, 1/60e6,
                                      2/60e6,   0,       0,      0)
    S0 = 1 - I0 - D0 - A0 - R0_0 - T0 - H0 - E0
    y0 = [S0, I0, D0, A0, R0_0, T0, H0, E0]

    t = np.linspace(0, 365, 2000)

    print(f"{'Scenario':<35}  R0")
    print("-" * 45)

    for title, params in SCENARIOS.items():
        sol = rk4(sidarthe, y0, t, params)
        R0  = compute_R0(params)
        print(f"{title:<35}  {R0:.3f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, (label, color) in enumerate(zip(LABELS, COLORS)):
            ax.plot(t, sol[:, idx], label=label, color=color)
        ax.set_title(f"SIDARTHE Model — {title}  (R₀ = {R0:.2f})")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Population Fraction")
        ax.legend(loc="center right", fontsize=8)
        ax.set_xlim(0, 200)
        fig.tight_layout()

        fname = title.replace(" ", "_").replace("–", "-").replace("(", "").replace(")", "") + ".png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150)
        plt.close(fig)

    print(f"\nPlots saved to '{save_dir}/'")


if __name__ == "__main__":
    run_simulation()
