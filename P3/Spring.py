import os

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from P3.utils import read_network


def compute_Columbus_force(pos_u, pos_v, k=3.0, q_v=1, q_u=1):
    delta = np.array(pos_u) - np.array(pos_v)
    distance = np.linalg.norm(delta) + 1e-6
    force =-k * q_v * q_u / distance ** 2
    angle = np.arctan2(delta[1], delta[0])

    f_x = force * np.cos(angle)
    f_y = force * np.sin(angle)
    f = np.array([f_x, f_y])
    for i in range(2):  # to return forces correctly directed for the node passed first (the other is opposite)
        sign = np.sign(delta[i]) * np.sign(force)
        f[i] = -abs(f[i]) * sign
    return f


def compute_spring_force(pos_u, pos_v, k=0.3, x0=2.0):
    delta = np.array(pos_u) - np.array(pos_v)
    distance = np.linalg.norm(delta) + 1e-6
    force = k * (distance - x0)
    angle = np.arctan2(delta[1], delta[0])

    f_x = force * np.cos(angle)
    f_y = force * np.sin(angle)
    f = np.array([f_x, f_y])
    for i in range(2):  # to return forces correctly directed for the node passed first (the other is opposite)
        sign = np.sign(delta[i])*np.sign(force)
        f[i]= -abs(f[i])*sign
    return f


def compute_movement(force, velocity, m=1, t=1.0):
    return velocity * t + force / m * t ** 2


def compute_velocity(force, m=1, t=1.0):
    return force / m * t


def update(positions, G, velocities=None, t=1.0, k_c=1.0, k_s=1.0, x0=1.0):
    if velocities is None:
        velocities = {node: np.zeros(2) for node in G.nodes()}
    if positions is None:
        positions = {node: [np.random.uniform(0, 10), np.random.uniform(0, 10)] for node in G.nodes()}

    forces = {node: np.zeros(2) for node in G.nodes()}
    displacements = {node: np.zeros(2) for node in G.nodes()}
    v_change = {node: np.zeros(2) for node in G.nodes()}

    for u, v in G.edges():
        force = compute_spring_force(positions[u], positions[v], k=k_s, x0=x0)
        forces[u] += force
        forces[v] -= force

    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                forces[u] += compute_Columbus_force(positions[u], positions[v], k=k_c)
        displacements[u] += compute_movement(forces[u], velocities[u], t=t)
        v_change[u] += compute_velocity(forces[u], t=t)

    # Update
    for node in G.nodes():
       positions[node] += displacements[node]
       velocities[node] += v_change[node]

    return positions, velocities


def draw(G, positions=None, i=0, folder='anim'):
    if positions is None:
        positions = {node: [np.random.uniform(0, 5), np.random.uniform(0, 5)] for node in G.nodes()}

    fig, ax = plt.subplots(figsize=(12, 12))

    # draw edges
    for u, v in G.edges():
        xs = [positions[u][0], positions[v][0]]
        ys = [positions[u][1], positions[v][1]]
        plt.plot(xs, ys,linewidth=1)

    # draw nodes
    plt.scatter(*zip(*positions.values()), color='blue', s=150)
    plt.axis('equal')
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}\\spring_{i}.png"
    fig.savefig(filename, pad_inches=0)
    plt.close()
    return filename


def make_animation(G, k_c=1.0, k_s=1.0, x0=1.0, j='0'):
    velocities = {node: np.zeros(2) for node in G.nodes()}
    positions = {node: [np.random.uniform(0, 10), np.random.uniform(0, 10)] for node in G.nodes()}
    filenames = []
    folder = 'anim'
    for i in range(100):
        filenames.append(draw(G, positions, i, folder=folder))
        positions, velocities = update(positions, G, velocities, t=0.1, k_c=k_c, k_s=k_s, x0=x0)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        os.remove(filename)
    imageio.mimsave(f'spring_animation_{j}.gif', images, fps=10)


def main():
    # G = read_network('..\\P2\\aves-weaver-social-01.edges')
    # make_animation(G, k_c=5.0, k_s=0.3, x0=2.0, j=0)
    file_map = {
        'firm': 'soc-firm-hi-tech.txt',
        'trib': 'soc-tribes.edges',
        'dolp': 'soc-dolphins.mtx',
        'weav': 'aves-weaver-social-01.edges'
    }

    for key in file_map:
        G = read_network(f'..\\P2\\{file_map[key]}')
        make_animation(G, k_c=5.0, k_s=0.3, x0=2.0, j=key)

if __name__ == '__main__':
    main()
