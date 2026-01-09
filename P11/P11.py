import random

import numpy as np


# ===================   Utils     ==============

def average_degree(G):
    """
    calculate <k> and <k^2>
    """
    degrees = np.array([deg for _, deg in G.degree()])
    k_mean = degrees.mean()
    k2_mean = (degrees ** 2).mean()
    return k_mean, k2_mean


def initial_infection(G, rho0):
    """
    initial fraction rho0 of infected nodes
    Returns a dict: node -> state ('S', 'I' etc.)
    """
    nodes = list(G.nodes())
    n_init = max(1, int(rho0 * len(nodes)))
    infected = set(random.sample(nodes, n_init))
    state = {}
    for v in nodes:
        state[v] = 'I' if v in infected else 'S'
    return state


# ===================   SIS     =================

def sis_step(G, state, beta, gamma):
    """
    state[v] in {'S', 'I'}.
    - I -> S with prob gamma
    - S -> I if  with infected neighbor with prob beta
    """
    new_state = state.copy()

    for v in G.nodes():
        if state[v] == 'I':
            # recovery
            if random.random() < gamma:
                new_state[v] = 'S'

    # infection
    for v in G.nodes():
        if state[v] == 'S':
            infected_neighbors = [u for u in G.neighbors(v) if state[u] == 'I']
            if infected_neighbors:
                # >=1 infected neighbor
                # prob that v stays S given k infected neighbors:
                # (1 - beta)^k, so prob(infection) = 1 - (1 - beta)^k
                k_inf = len(infected_neighbors)
                p_inf = 1.0 - (1.0 - beta) ** k_inf
                if random.random() < p_inf:
                    new_state[v] = 'I'
    return new_state


def simulate_sis(G, beta, gamma, rho0=0.05, T=200, burn_in=100, n_runs=10, seed=None):
    """
    simulate SIS on graph G for n_runs independent runs.

    Returns:
      - ts: time points (0..T)
      - prevalence_mean: mean I(t)/N across runs
      - prevalence_std: std I(t)/N across runs
      - rho_stationary_mean: mean stationary prevalence after burn-in
      - rho_stationary_std: std stationary prevalence after burn-in
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    N = G.number_of_nodes()
    all_prevalence = []

    for run in range(n_runs):
        state = initial_infection(G, rho0)
        prevalence = []
        for t in range(T + 1):
            I_t = sum(1 for v in G.nodes() if state[v] == 'I')
            prevalence.append(I_t / N)
            state = sis_step(G, state, beta, gamma)
        all_prevalence.append(prevalence)

    all_prevalence = np.array(all_prevalence)
    ts = np.arange(T + 1)
    prevalence_mean = all_prevalence.mean(axis=0)
    prevalence_std = all_prevalence.std(axis=0)

    # stationary estimate = average over [burn_in, T]
    stationary_vals = all_prevalence[:, burn_in:]
    rho_stationary_mean = stationary_vals.mean()
    rho_stationary_std = stationary_vals.std()

    return ts, prevalence_mean, prevalence_std, rho_stationary_mean, rho_stationary_std


# ===================   SIR     =================

def sir_step(G, state, beta, gamma):
    """
    state[v] in {'S', 'I', 'R'}
    - I -> R with prob gamma
    - S -> I by infected neighbor with prob beta
    """
    new_state = state.copy()

    # recovery
    for v in G.nodes():
        if state[v] == 'I':
            if random.random() < gamma:
                new_state[v] = 'R'

    # infection
    for v in G.nodes():
        if state[v] == 'S':
            infected_neighbors = [u for u in G.neighbors(v) if state[u] == 'I']
            if infected_neighbors:
                k_inf = len(infected_neighbors)
                p_inf = 1.0 - (1.0 - beta) ** k_inf
                if random.random() < p_inf:
                    new_state[v] = 'I'

    return new_state


def simulate_sir(G, beta, gamma, rho0=0.05, T=200, n_runs=20, seed=None):
    """
    simulate SIR on graph G.

    Returns:
      - ts
      - I_mean(t)
      - R_mean(t): final outbreak size = R_mean at large t
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    N = G.number_of_nodes()
    I_runs = []
    R_runs = []

    for run in range(n_runs):
        state = initial_infection(G, rho0)
        I_t = []
        R_t = []
        for t in range(T + 1):
            I_curr = sum(1 for v in G.nodes() if state[v] == 'I')
            R_curr = sum(1 for v in G.nodes() if state[v] == 'R')
            I_t.append(I_curr / N)
            R_t.append(R_curr / N)
            state = sir_step(G, state, beta, gamma)
        I_runs.append(I_t)
        R_runs.append(R_t)

    I_runs = np.array(I_runs)
    R_runs = np.array(R_runs)
    ts = np.arange(T + 1)

    return ts, I_runs.mean(axis=0), R_runs.mean(axis=0)


# ===================   voter     =================

def initialize_biased_voter(G, gamma=0.1, seed=None):
    """
    init opinions and voter types

    - opinion[v] in {-1, +1}
    - voter_type[v] in {'unbiased', 'biased'}
    - gamma: fraction of biased voters
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nodes = list(G.nodes())
    N = len(nodes)

    # random init
    opinion = {v: random.choice([-1, 1]) for v in nodes}

    # biased voters
    n_biased = int(gamma * N)
    biased_nodes = set(random.sample(nodes, n_biased))

    voter_type = {}
    for v in nodes:
        voter_type[v] = 'biased' if v in biased_nodes else 'unbiased'

    return opinion, voter_type


def voter_step_biased(G, opinion, voter_type, v=0.2):
    """
    one pair interaction update

      - G:  Graph
      - opinion[v] in {-1, +1}
      - voter_type[v] in {'unbiased', 'biased'}
      - v: bias strength in [0,1)

    Returns:
      - new_opinion
    """
    new_opinion = opinion.copy()
    nodes = list(G.nodes())

    # pick random node i
    i = random.choice(nodes)
    neighbors = list(G.neighbors(i))
    if not neighbors:
        # Isolated node: no change
        return new_opinion

    # pick random neighbor j
    j = random.choice(neighbors)

    si = opinion[i]
    sj = opinion[j]

    if si == sj:
        # pass
        return new_opinion

    node_type = voter_type[i]

    if node_type == 'unbiased' and random.random() < 0.5:
        # classic, copy neighbor with prob 0.5
        new_opinion[i] = sj

    elif node_type == 'biased':
        # neighbor has preferred opinion {1}-> more likely to adopt
        if random.random() < (1.0 + sj*v) / 2.0:
            new_opinion[i] = sj

    return new_opinion


def simulate_voter_biased(G, T=10000, gamma=0.1, v=0.2, n_runs=20, seed=None):
    """
    biased voter model for T steps (each step = N single pair updates).

    Returns:
      - ts: time array of len T+1
      - m_mean: mean magnetization m(t) across runs
      - m_std: std of magnetization
      - consensus_fraction: fraction of runs reaching consensus
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    preferred_state = 1

    N = G.number_of_nodes()
    ts = np.arange(T + 1)

    all_magnetizations = []
    consensus_states = []

    for run in range(n_runs):
        opinion, voter_type = initialize_biased_voter(G, gamma=gamma, seed=None)

        mags = []

        for t in range(T + 1):
            # mag at the start of step t
            m_t = sum(opinion[v] for v in G.nodes()) / N
            mags.append(m_t)

            # N single pair updates
            for _ in range(N):
                opinion = voter_step_biased(G, opinion, voter_type, v=v)

            # early stop if consensus
            if abs(m_t) == 1.0:
                break

        # if stopped early -> pad mags with final val
        if len(mags) < T + 1:
            mags += [mags[-1]] * (T + 1 - len(mags))

        all_magnetizations.append(mags)

        # save final consensus state
        final_m = mags[-1]
        if abs(final_m) == 1.0:
            consensus_states.append(int(final_m))  # +1 or -1
        else:
            consensus_states.append(0)  # no full consensus

    all_magnetizations = np.array(all_magnetizations)
    m_mean = all_magnetizations.mean(axis=0)
    m_std = all_magnetizations.std(axis=0)

    consensus_fraction = {"preferred": consensus_states.count(preferred_state) / n_runs,
                          "non_preferred": consensus_states.count(-preferred_state) / n_runs,
                          "no_consensus": consensus_states.count(0) / n_runs, }

    return ts, m_mean, m_std, consensus_fraction
