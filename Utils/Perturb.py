import numpy as np
from SGRLvCLT import *

# --- Rotation and transformation functions ---

def transformx(p, theta):
    """Rotate around x-axis by theta."""
    rotation = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])
    return np.matmul(rotation, p)

def transformy(p, theta):
    """Rotate around y-axis by theta."""
    rotation = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return np.matmul(rotation, p)

def transformz(p, theta):
    """Rotate around z-axis by theta."""
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    return np.matmul(rotation, p)

def flip(p, fx, fy, fz):
    """Flip signs across axes."""
    return np.array([[p[0] * fx], [p[1] * fy], [p[2] * fz]])

def randomRotation(points):
    """Apply a random sequence of flips, rotations, and translations to a set of 3D points."""
    fx = -1 if np.random.rand() > 0.5 else 1
    fy = -1 if np.random.rand() > 0.5 else 1
    fz = -1 if np.random.rand() > 0.5 else 1

    tx = np.random.rand(3, 1) * 20 - 10
    ty = np.random.rand(3, 1) * 20 - 10
    tz = np.random.rand(3, 1) * 20 - 10
    tf = np.random.rand(3, 1) * 20 - 10

    ax = np.random.rand() * 2 * np.pi
    ay = np.random.rand() * 2 * np.pi
    az = np.random.rand() * 2 * np.pi

    transformed = []
    for p in points:
        p1 = flip(p, fx, fy, fz)
        p1 = transformx(p1 + tx, ax)
        p1 = transformy(p1 + ty, ay)
        p1 = transformz(p1 + tz, az)
        transformed.append((p1 + tf).T)

    return np.vstack(transformed)

# --- Perturbation Evaluation ---

max_iter = 1000
gamma_vals = [
    0, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025,
    0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5
]

avglist = []

for gamma in gamma_vals:
    dist_list = []

    for _ in range(max_iter):
        n_points = np.random.randint(3, 11)
        matrix = np.random.uniform(-1, 1, (n_points, 3))

        perturbation = np.random.uniform(-1, 1, (n_points, 3))
        perturbed_matrix = (perturbation * gamma) + matrix
        transformed_matrix = randomRotation(perturbed_matrix)

        canonical_matrix = standardTotal(matrix)
        canonical_perturbed = standardTotal(perturbed_matrix)

        flat_canonical = np.vstack(canonical_matrix)
        flat_perturbed = np.vstack(canonical_perturbed)

        euclidean = np.linalg.norm(flat_canonical - flat_perturbed)
        dist_list.append(euclidean)

    mean_distance = np.mean(dist_list)
    avglist.append(mean_distance)
