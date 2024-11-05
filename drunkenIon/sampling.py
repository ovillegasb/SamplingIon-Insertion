"""Functions and tools for sampling."""

import time
import pickle
import numpy as np
import pandas as pd
from scipy.constants import epsilon_0, e
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pymatgen.core import Element, Molecule
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from multiprocessing import Pool, Process, Manager, Lock
import psutil
import os
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.transform import Rotation
from ase import Atoms

angstroms_per_meter = 1e10


def task_function():
    # Obtener el ID del proceso actual
    pid = os.getpid()
    # Usar psutil para encontrar el CPU en el que se ejecuta
    process = psutil.Process(pid)
    # cpu_affinity = process.cpu_affinity()  # Afinidad del proceso con los CPUs

    # Obtener el CPU actual asignado
    cpu_num = process.cpu_num()
    return cpu_num


def calcular_silhouette(k, data):
    # , scores_dict, completed=0, lock=None
    """Función auxiliar para calcular el score de silhouette para un valor de k"""
    print("Studying cluster number:", k, end=" - ")
    t_i_c = time.time()
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(data)
    score = silhouette_score(data, labels)
    t_f_c = time.time()
    # return score
    # time.sleep(5.0)
    # score = np.random.rand()
    #####$###with lock:
    #####$###    scores_dict[k] = score
    #####$###    completed.value += 1

    # print("k={} finished!".format(k))
    # return
    execution_time_c = t_f_c - t_i_c
    print("done %.3f" % execution_time_c)
    return score


def rotate_molecule(coords):
    # Asegúrate de que coords esté en forma (N, 3)
    np.random.seed(None)
    coords = np.array(coords)
    if coords.shape[1] != 3:
        coords = coords.T

    # Genera un vector de rotación aleatorio y un ángulo aleatorio
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # Normaliza para que sea un vector unitario
    angle = np.random.uniform(0, 2 * np.pi)  # Ángulo aleatorio entre 0 y 2pi

    # Crea la matriz de rotación
    rotation_matrix = Rotation.from_rotvec(angle * axis).as_matrix()

    # Aplica la rotación
    rotated_coords = coords @ rotation_matrix.T  # Multiplicación matricial

    return rotated_coords


def rotate_molecule_2(coords):
    """
    Rota un conjunto de coordenadas 3D usando un eje aleatorio y un ángulo aleatorio.

    Parameters:
        coords (np.ndarray): Array 3xN que representa las coordenadas de los átomos en la molécula.

    Returns:
        np.ndarray: Array 3xN con las coordenadas rotadas.
    """
    # Genera un vector aleatorio y normalízalo para obtener un eje de rotación unitario
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    # Genera un ángulo de rotación aleatorio en radianes entre 0 y 2π
    angle = np.random.uniform(0, 2 * np.pi)

    # Crea la matriz de rotación usando la fórmula de Rodrigues
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    ux, uy, uz = axis

    rotation_matrix = np.array([
        [cos_angle + ux**2 * (1 - cos_angle),       ux * uy * (1 - cos_angle) - uz * sin_angle, ux * uz * (1 - cos_angle) + uy * sin_angle],
        [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy**2 * (1 - cos_angle),       uy * uz * (1 - cos_angle) - ux * sin_angle],
        [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle, cos_angle + uz**2 * (1 - cos_angle)]
    ])

    # Aplica la rotación a cada punto en `coords`
    rotated_coords = rotation_matrix @ coords

    return rotated_coords


def calculate_rmsd(energies, reference=None):
    """
    Calculate the RMSD of a list of energies.

    Parameters:
    -----------
    energies : array-like
        List of energies throughout the simulation.
    reference : float, optional
        Reference value for RMSD calculation. If not provided, the average of the energies
        will be used.

    Return:
    -------
    float
        RMDS values.
    """
    energies = np.array([val for val in energies if not np.isnan(val)])

    # If a reference value is not provided, use the average of the energies
    if reference is None:
        reference = np.mean(energies)

    # Calculate the root mean square deviation
    rmsd = np.sqrt(np.mean((energies - reference) ** 2))
    return rmsd


def show_axis_3D(lattice):
    print(lattice)
    # Origen y vectores de la celda
    origin = np.zeros(3)
    v1, v2, v3 = lattice
    plt.close()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Graficar cada vector desde el origen
    ax.quiver(*origin, *v1, color='r', arrow_length_ratio=0.1, label="v1")
    ax.quiver(*origin, *v2, color='g', arrow_length_ratio=0.1, label="v2")
    ax.quiver(*origin, *v3, color='b', arrow_length_ratio=0.1, label="v3")

    # Configurar límites de los ejes para una mejor visualización
    max_limit = np.max(np.abs(lattice)) * 1.2
    ax.set_xlim([-max_limit, max_limit])
    ax.set_ylim([-max_limit, max_limit])
    ax.set_zlim([-max_limit, max_limit])

    # Etiquetas y leyenda
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()
    plt.show()


def plot_lattice_and_atoms5(ax, atoms, samples, lattice, plane='xy'):
    """Plot atoms and lattice boundaries based on the specified plane."""
    # Definir los índices de coordenadas según el plano deseado
    planes = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    if plane not in planes:
        raise ValueError(f"Invalid plane '{plane}'. Choose from 'xy', 'xz', or 'yz'.")
    indices = planes[plane]

    # Obtener las combinaciones de vectores que forman los límites de la celda unitaria en el plano
    origin = np.zeros(3)
    v1 = lattice[0]  # Primer vector de la celda
    v2 = lattice[1]  # Segundo vector de la celda
    v3 = lattice[2]  # Tercer vector de la celda

    # Generar los puntos de la celda en el plano deseado
    vertices = [origin, v1, v1 + v2, v2, v3, v1 + v3, v2 + v3, v1 + v2 + v3, origin]

    # Proyectar los puntos en el plano especificado
    x_vals = [vertex[indices[0]] for vertex in vertices]
    y_vals = [vertex[indices[1]] for vertex in vertices]

    # Trazar las líneas que representan la celda en el plano
    ax.plot(x_vals[:5], y_vals[:5], 'k-')  # Contorno de la celda en el plano original
    ax.plot(x_vals[5:], y_vals[5:], 'k--')  # Proyecciones desplazadas en z

    # Graficar los átomos y las muestras en el plano seleccionado
    ax.scatter(atoms[:, indices[0]], atoms[:, indices[1]], c="violet", s=200, edgecolor="k", label="Átomos")
    ax.scatter(samples[:, indices[0]], samples[:, indices[1]], s=200, edgecolor="k", marker=".", color="blue", label="Muestras")

    ax.set_xlabel(f'{plane[0]}-axis')
    ax.set_ylabel(f'{plane[1]}-axis')
    ax.grid(False)


def plot_lattice_and_atoms(ax, atoms, samples, lattice, plane='xy'):
    """Plot atoms and lattice lines based on the specified plane."""
    # Define the indices for the desired plane
    planes = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    if plane not in planes:
        raise ValueError(f"Invalid plane '{plane}'. Choose from 'xy', 'xz', or 'yz'.")
    indices = planes[plane]

    origin = np.zeros(3)
    # show_axis_3D(lattice)

    # Define the start and end points for lattice vectors in the chosen plane
    # v1, v2 = lattice[indices[0]], lattice[indices[1]
    # v1, v2, v3 = lattice[:, indices[0]], lattice[:, indices[1]], [0, 0]
    v1, v2, v3 = lattice
    # vertices = [origin, v1, v1 + v2, v2, origin]
    vertices = [origin, v1, v1 + v2, v2, origin, v1, v1 + v3, v3, v2 + v3, v2, origin]

    # Plot the lattice by connecting vertices in the chosen plane
    ax.plot([vertices[i][indices[0]] for i in range(len(vertices))],
            [vertices[i][indices[1]] for i in range(len(vertices))], 'k-')

    # Plot the atoms and samples in the selected plane
    ax.scatter(atoms[:, indices[0]], atoms[:, indices[1]], c="violet", s=200, edgecolor="k")
    ax.scatter(samples[:, indices[0]], samples[:, indices[1]], s=200, edgecolor="k", marker=".", color="blue")

    ax.set_xlabel(f'{plane[0]}-axis')
    ax.set_ylabel(f'{plane[1]}-axis')
    ax.grid(False)


def plot_lattice_and_atoms3(ax, atoms, samples, lattice, plane='xy'):
    """
    Plot atoms and lattice vectors projected in a specified plane.

    Parameters:
    - ax: Matplotlib axis object for plotting.
    - atoms: np.ndarray of shape (N, 3), positions of atoms.
    - samples: np.ndarray of shape (N, 3), additional points to plot.
    - lattice: np.ndarray of shape (3, 3), the lattice matrix.
    - plane: str, plane to project ('xy', 'xz', or 'yz').
    """
    # Determine plane indices based on the selected plane
    if plane == 'xy':
        indices = (0, 1)
    elif plane == 'xz':
        indices = (0, 2)
    elif plane == 'yz':
        indices = (1, 2)
    else:
        raise ValueError("Invalid plane. Choose from 'xy', 'xz', 'yz'.")

    # Extract vectors in the selected plane
    v1, v2, v3 = lattice[:, indices[0]], lattice[:, indices[1]], [0, 0]

    # Plot lattice vectors in the specified plane
    for start, end in [(0, v1), (0, v2), (v1, v1 + v2), (v2, v1 + v2)]:
        ax.plot([start[indices[0]], end[indices[0]]],
                [start[indices[1]], end[indices[1]]], 'k-')

    # Plot atoms and samples in the selected plane
    ax.scatter(atoms[:, indices[0]], atoms[:, indices[1]], c="violet", s=200, edgecolor="k")
    ax.scatter(samples[:, indices[0]], samples[:, indices[1]], s=200, edgecolor="k", marker=".", color="blue")

    # Axis labels and grid
    ax.set_xlabel(f'{plane[0]}-axis')
    ax.set_ylabel(f'{plane[1]}-axis')
    ax.grid(False)


def plot_lattice_and_atoms2(ax, atoms, samples, lattice, plane='xy'):
    """Plot atoms and lattice like lines."""
    if plane == 'xy':
        # BOX
        ax.plot([0, lattice[0, 0]], [0, lattice[1, 0]], 'k-')
        ax.plot([0, lattice[0, 0]], [lattice[1, 1], lattice[1, 1]], 'k-')
        ax.plot([0, lattice[0, 1]], [0, lattice[1, 1]], 'k-')
        ax.plot([lattice[0, 0], lattice[0, 0]], [0, lattice[1, 1]], 'k-')
        # Atoms
        ax.scatter(atoms[:, 0], atoms[:, 1], c="violet", s=200, edgecolor="k")
        ax.scatter(samples[:, 0], samples[:, 1], s=200, edgecolor="k", marker=".", color="blue")
    elif plane == 'xz':
        ax.plot([0, lattice[0, 0]], [0, lattice[2, 0]], 'k-')
        ax.plot([0, lattice[0, 0]], [lattice[2, 2], lattice[2, 2]], 'k-')
        ax.plot([0, lattice[0, 2]], [0, lattice[2, 2]], 'k-')
        ax.plot([lattice[0, 0], lattice[0, 0]], [0, lattice[2, 2]], 'k-')
        ax.scatter(atoms[:, 0], atoms[:, 2], c="violet", s=100, edgecolor="k")
        ax.scatter(samples[:, 0], samples[:, 2], s=200, edgecolor="k", marker=".", color="blue")
    elif plane == 'yz':
        ax.plot([0, lattice[1, 1]], [0, lattice[2, 1]], 'k-')
        ax.plot([0, lattice[1, 1]], [lattice[2, 2], lattice[2, 2]], 'k-')
        ax.plot([0, lattice[1, 2]], [0, lattice[2, 2]], 'k-')
        ax.plot([lattice[1, 1], lattice[1, 1]], [0, lattice[2, 2]], 'k-')
        ax.scatter(atoms[:, 1], atoms[:, 2], c="violet", s=100, edgecolor="k")
        ax.scatter(samples[:, 1], samples[:, 2], s=200, edgecolor="k", marker=".", color="blue")

    ax.set_xlabel(f'{plane[0]}-axis')
    ax.set_ylabel(f'{plane[1]}-axis')
    # ax.legend()
    ax.grid(False)


def plot_clusters(ax, atoms, samples, centroids, cluster_labels, lattice, plane='xy'):
    """Plot atoms and lattice like lines."""
    planes = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    if plane not in planes:
        raise ValueError(f"Invalid plane '{plane}'. Choose from 'xy', 'xz', or 'yz'.")
    indices = planes[plane]

    origin = np.zeros(3)
    # show_axis_3D(lattice)

    # Define the start and end points for lattice vectors in the chosen plane
    # v1, v2 = lattice[indices[0]], lattice[indices[1]
    # v1, v2, v3 = lattice[:, indices[0]], lattice[:, indices[1]], [0, 0]
    v1, v2, v3 = lattice
    # vertices = [origin, v1, v1 + v2, v2, origin]
    vertices = [origin, v1, v1 + v2, v2, origin, v1, v1 + v3, v3, v2 + v3, v2, origin]

    # Plot the lattice by connecting vertices in the chosen plane
    ax.plot([vertices[i][indices[0]] for i in range(len(vertices))],
            [vertices[i][indices[1]] for i in range(len(vertices))], 'k-')

    if plane == 'xy':
        # BOX
        # Atoms
        ax.scatter(atoms[:, 0], atoms[:, 1], c="violet", s=200, edgecolor="k")
        ax.scatter(samples[:, 0], samples[:, 1], s=50, c=cluster_labels, cmap='viridis')
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75)
    elif plane == 'xz':
        ax.scatter(atoms[:, 0], atoms[:, 2], c="violet", s=100, edgecolor="k")
        ax.scatter(samples[:, 0], samples[:, 2], s=50, c=cluster_labels, cmap='viridis')
        ax.scatter(centroids[:, 0], centroids[:, 2], c='red', s=200, alpha=0.75)
    elif plane == 'yz':
        ax.scatter(atoms[:, 1], atoms[:, 2], c="violet", s=100, edgecolor="k")
        ax.scatter(samples[:, 1], samples[:, 2], s=50, c=cluster_labels, cmap='viridis')
        ax.scatter(centroids[:, 1], centroids[:, 2], c='red', s=200, alpha=0.75)

    ax.set_xlabel(f'{plane[0]}-axis')
    ax.set_ylabel(f'{plane[1]}-axis')
    # ax.legend()
    ax.grid(False)


def plot_densities(ax, atoms, samples, lattice, plane='xy'):
    """Plot atoms and lattice like lines."""

    planes = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    if plane not in planes:
        raise ValueError(f"Invalid plane '{plane}'. Choose from 'xy', 'xz', or 'yz'.")
    indices = planes[plane]

    origin = np.zeros(3)
    # show_axis_3D(lattice)

    # Define the start and end points for lattice vectors in the chosen plane
    # v1, v2 = lattice[indices[0]], lattice[indices[1]
    # v1, v2, v3 = lattice[:, indices[0]], lattice[:, indices[1]], [0, 0]
    v1, v2, v3 = lattice
    # vertices = [origin, v1, v1 + v2, v2, origin]
    vertices = [origin, v1, v1 + v2, v2, origin, v1, v1 + v3, v3, v2 + v3, v2, origin]

    # Plot the lattice by connecting vertices in the chosen plane
    ax.plot([vertices[i][indices[0]] for i in range(len(vertices))],
            [vertices[i][indices[1]] for i in range(len(vertices))], 'k-')

    if plane == 'xy':
        # BOX
        ax.scatter(atoms[:, 0], atoms[:, 1], c="violet", s=200, edgecolor="k")
        sns.histplot(x=samples[:, 0], y=samples[:, 1], bins=(40, 40), stat="count", cbar=True, ax=ax)
    elif plane == 'xz':
        ax.scatter(atoms[:, 0], atoms[:, 2], c="violet", s=100, edgecolor="k")
        sns.histplot(x=samples[:, 0], y=samples[:, 2], bins=(40, 40), stat="count", cbar=True, ax=ax)
    elif plane == 'yz':
        ax.scatter(atoms[:, 1], atoms[:, 2], c="violet", s=100, edgecolor="k")
        sns.histplot(x=samples[:, 1], y=samples[:, 2], bins=(40, 40), stat="count", cbar=True, ax=ax)

    ax.set_xlabel(f'{plane[0]}-axis')
    ax.set_ylabel(f'{plane[1]}-axis')
    # ax.legend()
    ax.grid(False)


def plot_box(ax, lattice):
    # Definir el origen (punto de partida de los vectores de red)
    origin = np.array([0, 0, 0])
    # Trazar cada vector de red desde el origen
    ax.quiver(*origin, *lattice[0], color='r', arrow_length_ratio=0.0)
    ax.quiver(*origin, *lattice[1], color='g', arrow_length_ratio=0.0)
    ax.quiver(*origin, *lattice[2], color='b', arrow_length_ratio=0.0)

    #####
    ax.quiver(
        *(lattice[1] + origin),
        *lattice[0], color='r', arrow_length_ratio=0.0
    )

    ax.quiver(
        *(lattice[2] + origin),
        *lattice[0], color='r', arrow_length_ratio=0.0
    )

    ax.quiver(
        *(lattice[1] + lattice[2] + origin),
        *lattice[0], color='r', arrow_length_ratio=0.0
    )
    #####
    ax.quiver(
        *(lattice[0] + origin),
        *lattice[1], color='g', arrow_length_ratio=0.0
    )

    ax.quiver(
        *(lattice[2] + origin),
        *lattice[1], color='g', arrow_length_ratio=0.0
    )

    ax.quiver(
        *(lattice[0] + lattice[2] + origin),
        *lattice[1], color='g', arrow_length_ratio=0.0
    )
    #####
    ax.quiver(
        *(lattice[0] + origin),
        *lattice[2], color='b', arrow_length_ratio=0.0
    )

    ax.quiver(
        *(lattice[1] + origin),
        *lattice[2], color='b', arrow_length_ratio=0.0
    )

    ax.quiver(
        *(lattice[0] + lattice[1] + origin),
        *lattice[2], color='b', arrow_length_ratio=0.0
    )


def apply_PBC_with_lattice(position, lattice_matrix):
    """Apply periodic boundary conditions to a position using a lattice matrix."""
    # Inverse of lattice matrix to obtain cell coordinates
    inv_lattice = np.linalg.inv(lattice_matrix)

    # Project the position to fractional coordinates
    fractional_coords = np.dot(inv_lattice, position)

    # Apply PBC by wrapping the coordinates within [0, 1)
    fractional_coords = fractional_coords - np.floor(fractional_coords)

    # Convert back to Cartesian coordinates
    new_position = np.dot(lattice_matrix, fractional_coords)

    return new_position


def wrap_single_position(position, lattice_matrix, pbc=True, center=(0.5, 0.5, 0.5), eps=1e-7):
    """
    Envuelve una posición en una celda periódica dada la matriz de celda.

    Parameters:
    - position: ndarray de forma (3,), coordenadas de la partícula.
    - lattice_matrix: ndarray de forma (3, 3), vectores de la celda.
    - pbc: bool o lista de 3 bool, si es True aplica condiciones de contorno periódicas.
    - center: tres float, coordenadas en la celda fraccional hacia las que se ajustará la posición.
    - eps: float, valor pequeño para prevenir ajustes negativos de coordenadas.

    Returns:
    - ndarray de forma (3,), la posición ajustada en coordenadas cartesianas dentro de la celda.
    """
    # Asegurar que center y pbc están bien formateados para una sola partícula
    if not hasattr(center, '__len__'):
        center = (center,) * 3
    if isinstance(pbc, bool):
        pbc = [pbc, pbc, pbc]

    # Transformación a coordenadas fraccionales
    shift = np.asarray(center) - 0.5 - eps
    shift = np.where(pbc, shift, 0.0)  # Solo aplicamos el shift donde pbc es True

    # Calcula las coordenadas fraccionales de la posición dada
    fractional_coords = np.linalg.solve(lattice_matrix.T, position) - shift

    # Ajuste en coordenadas fraccionales para PBC
    for i, periodic in enumerate(pbc):
        if periodic:
            fractional_coords[i] %= 1.0
            fractional_coords[i] += shift[i]

    # Convierte de nuevo a coordenadas cartesianas
    wrapped_position = np.dot(fractional_coords, lattice_matrix)
    return wrapped_position


def minimum_image_convention(vec, lattice_matrix):
    """
    Aplica la convención de imagen mínima para un vector de desplazamiento en una celda de Bravais general.

    Parameters:
    - vec: ndarray de forma (3,), el vector de desplazamiento a ajustar.
    - lattice_matrix: ndarray de forma (3, 3), representa los vectores de la celda.

    Returns:
    - vec ajustado a la imagen mínima en coordenadas cartesianas.
    """
    # Convertir el vector de diferencia a coordenadas fraccionales
    inv_lattice = np.linalg.inv(lattice_matrix)
    fractional_vec = np.dot(inv_lattice, vec)

    # Aplicar la convención de imagen mínima en coordenadas fraccionales
    fractional_vec -= np.round(fractional_vec)

    # Convertir de regreso a coordenadas cartesianas
    adjusted_vec = np.dot(lattice_matrix, fractional_vec)

    return adjusted_vec


def minimum_image_convention2(vec, lattice_matrix):
    """Apply minimum image convention."""
    cell_lengths = np.linalg.norm(lattice_matrix, axis=1)
    vec = vec.copy()

    # Ajustar cada componente
    for i in range(len(cell_lengths)):  # Para cada dimensión x, y, z
        half_cell_length = cell_lengths[i] / 2.0
        # Ajustar la diferencia
        if vec[i] > half_cell_length:
            vec[i] -= cell_lengths[i]
        elif vec[i] < -half_cell_length:
            vec[i] += cell_lengths[i]

    return vec


def V_electric_potential(r, r_center, lattice, min_dist=0.0, int_type=1):
    """Calculate the electric potential using a cutoff distance (unit Volt)."""
    r_values = r - r_center
    r_values = minimum_image_convention(r_values, lattice)
    r_norm = np.linalg.norm(r_values)

    if r_norm > 2 * min_dist:
        # attraction
        # return -1/(4*np.pi*epsilon_0_V_angstroms*r_norm)
        # repulsion
        return int_type/(4*np.pi*epsilon_0*r_norm/angstroms_per_meter)
    else:
        return np.inf


def Total_potential(r, centers, lattice, min_distances, interaction_types):
    """Calculate the total energy on the particle."""
    V = 0.0
    for center, min_dist, int_type in zip(centers, min_distances, interaction_types):
        V += V_electric_potential(r, center, lattice, min_dist, int_type)

    return V


def acceptance_probability(delta_V, T):
    """Accept new step?."""
    # Function to accept a new position according to Metropolis criteria.
    if delta_V <= 0:
        return True
    else:
        return np.random.rand() < np.exp(-delta_V / T)


def random_position_in_lattice(lattice_matrix, n_particles, dim):
    """
    Genera una posición aleatoria dentro de la celda de la lattice_matrix.

    Parameters:
        lattice_matrix (np.array): Matriz 3x3 que define los vectores de celda.

    Returns:
        np.array: Posición en coordenadas cartesianas dentro de la celda.
    """
    # Generar coordenadas fraccionarias aleatorias dentro del rango [0, 1)
    fractional_coords = np.random.rand(n_particles, dim)  # Coordenadas fraccionarias f_x, f_y, f_z

    # Convertir a coordenadas cartesianas usando la matriz de lattice
    cartesian_position = np.dot(fractional_coords, lattice_matrix)

    return cartesian_position


def monte_carlo_metropolis(n_steps, T, r_init, centers, step_size, lattice, min_dist, interaction_type):
    """Run monte carlo simulation core."""
    # Inicialización
    r_current = r_init
    history = [r_current]
    energies = []
    n_accept = 0
    current_cpu = task_function()
    step = 0
    ###NEW
    # system = Atoms(positions=[r_current], pbc=True, cell=lattice)
    # print(system)
    # system.write("test.cif")
    ###NEW

    while step < n_steps:
        # Generates a new position by disturbing the current one.
        r_new = r_current + np.random.uniform(-step_size, step_size, size=r_current.shape)
        ###NEW
        # r_new = system.positions + np.random.uniform(-step_size, step_size, system.positions.shape)
        ###NEW

        # Apply periodic boundary conditions
        # r_new = apply_PBC_with_lattice(r_new, lattice)
        ###NEW
        r_new = wrap_single_position(r_new, lattice)
        #system.set_positions(r_new)
        #system.wrap()
        #r_new = system.positions[0]
        ###NEW

        # Calculates the change in potential
        delta_V = Total_potential(r_new, centers, lattice, min_dist, interaction_type) - Total_potential(r_current, centers, lattice, min_dist, interaction_type)
        # delta_V /= len(centers)
        # if np.isnan(delta_V):
        #     continue
        # if np.isinf(delta_V):
        #     continue
        # print(f"{delta_V:.3e} volt", end=" | ")
        # print(f"{e*delta_V:.3e} eV", end="\n")

        # Aplica el criterio de aceptación
        if acceptance_probability(e*delta_V, T):
            r_current = r_new  # Acepta la nueva posición
            n_accept += 1

        if not np.isinf(delta_V):
            energies.append(e*delta_V)

        # print(f"|{current_cpu}|", step, r_new, "|", f"{delta_V:.3e} volt", "|", f"{e*delta_V:.3e} eV")

        history.append(r_current)
        step += 1

    acceptance_rate = n_accept / n_steps
    print(f"|{current_cpu}| Acceptance rate: {acceptance_rate:.2f}")

    energies = np.array(energies)
    energies = np.where(np.isinf(energies), np.nan, energies)

    return np.array(history), energies, acceptance_rate


def apply_periodic_kmeans(positions, lattice_matrix, n_clusters, random_state=0):
    # Replicamos los puntos en las direcciones positivas y negativas de cada vector de celda
    offsets = np.array([
        [0, 0, 0],  # Celda original
        [1, 0, 0], [-1, 0, 0],  # Desplazamientos en x
        [0, 1, 0], [0, -1, 0],  # Desplazamientos en y
        [0, 0, 1], [0, 0, -1],  # Desplazamientos en z
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],  # Combinaciones x-y
        [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],  # Combinaciones x-z
        [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],  # Combinaciones y-z
        [1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, 1, -1], # Combinaciones x-y-z
    ])
    
    # Expandimos las posiciones usando desplazamientos periódicos
    expanded_positions = []
    for offset in offsets:
        displacement = np.dot(offset, lattice_matrix)
        expanded_positions.append(positions + displacement)
    expanded_positions = np.vstack(expanded_positions)
    
    # Aplicamos KMeans en el conjunto de posiciones expandidas
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(expanded_positions)
    
    # Convertimos los centros de vuelta a la celda original
    centers = kmeans.cluster_centers_ % lattice_matrix.diagonal()

    # Seleccionamos las etiquetas de los clústeres para las posiciones originales
    original_labels = kmeans.labels_[:len(positions)]
    
    return centers, original_labels


def periodic_silhouette_score(positions, lattice_matrix, n_clusters, random_state=0):
    # Extiende las posiciones periódicamente
    offsets = np.array([
        [0, 0, 0],  # Celda original
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
        [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
        [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
        [1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, 1, -1]
    ])

    expanded_positions = []
    for offset in offsets:
        displacement = np.dot(offset, lattice_matrix)
        expanded_positions.append(positions + displacement)
    expanded_positions = np.vstack(expanded_positions)

    # Ejecutamos KMeans sobre las posiciones expandidas
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(expanded_positions)

    # Calculamos el `silhouette_score` usando solo las posiciones originales
    original_labels = kmeans.labels_[:len(positions)]

    # Calculamos el `silhouette_score` usando las posiciones originales
    score = silhouette_score(expanded_positions[:len(positions)], original_labels)

    return score


class DrunkenIon:
    """Stochastic alignment of an ion in a restricted space."""

    def __init__(self, mof, min_dist=None, n_steps=500000, step_size=None, T=0.01, ncpus=1, factor=1.0):
        """Initialize by defining the initial parameters."""
        # print("Parameters:")
        lattice = mof.lattice
        zone = np.linalg.norm(lattice, axis=1)
        # print("zone (anstroms):", zone)
        self.zone = zone
        self.dim = len(zone)
        # print("dimension:", self.dim)
        # r_values = np.array(zone) * np.random.rand(n_particles, self.dim)
        r_values = random_position_in_lattice(lattice, ncpus, self.dim)
        self.r_values = r_values
        # print("init r_value:", r_values)
        # print("n_particles:", n_particles)
        self.atoms_positions = mof.cart_coords
        # print(self.atoms_positions)
        # print("N steps:", n_steps)
        self.n_steps = n_steps
        if step_size is None:
            # print(mof.volume)
            # print(np.cbrt(mof.volume) * 0.45)
            step_size = np.cbrt(mof.volume) * 0.4
            step_size = round(step_size, 2)
        # print("step size:", step_size)
        self.step_size = step_size
        # print("temperature:", T)
        self.T = T
        if min_dist is None:
            # print("Using convalent radii")
            self.min_dist = np.array(mof.covalent_radii) * factor
        else:
            # print("Min distace to atoms:", min_dist)
            self.min_dist = min_dist*np.ones(len(self.atoms_positions))
        # print(self.min_dist)
        # print("Lattice:\n", lattice)
        self.lattice = lattice
        self.natoms = len(self.atoms_positions)
        # print("N atoms:", self.natoms)
        self.mof = mof
        self.ncpus = ncpus
        self.factor = factor
        self.interaction_type = mof.interaction_type

    def run_montecarlo(self):
        """Run Monte Carlo simulation."""
        print("Running MonteCarlo simulation for to position an Ion in pores...")
        time.sleep(3)
        t_i = time.time()
        info_sim = {}
        if self.ncpus == 1:
            r_init = self.r_values[0]
            print("Initial particule position:", r_init)
            # Run Monte Carlo
            trajectory, energies, acceptance_rate = monte_carlo_metropolis(
                self.n_steps,
                self.T,
                r_init,
                self.atoms_positions,
                self.step_size,
                self.lattice,
                self.min_dist,
                self.interaction_type
            )

            info_sim["traj"] = trajectory
            info_sim["ener"] = energies
            info_sim["acc_rate"] = acceptance_rate
        else:
            n_steps_by_cpu = int(self.n_steps / self.ncpus)
            args = []
            for r_init in self.r_values:
                args.append((
                    n_steps_by_cpu,
                    self.T,
                    r_init,
                    self.atoms_positions,
                    self.step_size,
                    self.lattice,
                    self.min_dist
                ))

            with Pool(processes=self.ncpus) as pool:
                # run the simulation for each particle in paralell
                results = pool.starmap(monte_carlo_metropolis, args)

            trajectories = []
            energies = []
            acc_tot = 0
            for traj, ener, acc in results:
                trajectories.append(traj)
                energies.append(ener)
                acc_tot += acc

            info_sim["traj"] = np.concatenate(trajectories, axis=0)
            info_sim["ener"] = np.concatenate(energies, axis=0)
            info_sim["acc_rate"] = acc_tot / self.ncpus

        self.info_sim = info_sim

        t_f = time.time()
        execution_time = t_f - t_i
        print("Simulation finished")
        print("done in %.3f s" % execution_time)

    def compute_Kmeans(self, k=4):
        points = self.info_sim["traj"]
        # Aplicar el algoritmo k-means
        #kmeans = KMeans(n_clusters=k)
        #kmeans.fit(points)
        self.centroids, self.cluster_labels = apply_periodic_kmeans(points, self.lattice, k)
        #self.centroids = kmeans.cluster_centers_
        #self.cluster_labels = kmeans.labels_
        # self.kmeans = kmeans
        n_steps = self.n_steps
        T = self.T
        step_size = self.step_size
        factor = self.factor

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"sim MC - nsteps: {n_steps} - T: {T} - step_size: {step_size:.2f} - factor: {factor:.2f}")
        plot_clusters(axs[0], self.atoms_positions, points, self.centroids, self.cluster_labels, self.lattice, plane='xy')
        plot_clusters(axs[1], self.atoms_positions, points, self.centroids, self.cluster_labels, self.lattice, plane='xz')
        plot_clusters(axs[2], self.atoms_positions, points, self.centroids, self.cluster_labels, self.lattice, plane='yz')
        plt.tight_layout()
        plt.show()

    def clusters_study(self, max_n_porous=8):
        print("Searching number of probable porous...")
        t_i = time.time()
        data = self.info_sim["traj"]
        """
        # Lista para almacenar la suma de errores cuadráticos (SSE)
        sse = []
        k_values = range(1, 15)  # Prueba de 1 a 10 clusters
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data)
            sse.append(kmeans.inertia_)
        # Graficar los valores de SSE para cada k
        plt.plot(k_values, sse, marker='o')
        plt.xlabel('Cluster number (k)')
        plt.ylabel('SSE (Inertia)')
        plt.title('Elbow Method')
        plt.show()
        """

        if self.ncpus == 1:
            silhouette_scores = {}
            for k in range(2, max_n_porous+1):
                print("Studying cluster number:", k, end=" - ")
                t_i_c = time.time()
                # kmeans = KMeans(n_clusters=k, random_state=0)
                # _, labels = apply_periodic_kmeans(positions, lattice_matrix, n_clusters, random_state=0)
                score = periodic_silhouette_score(data, self.lattice, k, random_state=0)
                # labels = kmeans.fit_predict(data)
                # score = silhouette_score(data, labels)
                silhouette_scores[k] = score
                t_f_c = time.time()
                execution_time_c = t_f_c - t_i_c
                print("done %.3f" % execution_time_c)

        else:
            print("TODO")
            # args = [(k, data) for k in range(2, 11)]
            # with Pool(processes=self.ncpus) as pool:
            #     silhouette_scores = pool.starmap(calcular_silhouette, args)

        """
        def k_values_generator(start=2, end=11):
            for k in range(start, end):
                yield k

        k_values = range(2, 11)
        k_gen = k_values_generator()
        n_k_values = len(k_values)
        manager = Manager()
        lock = Lock()
        silhouette_scores = manager.dict()
        completed = manager.Value('i', 0)
        processes = []

        print("Kmeans study Main BUCLE")
        while completed.value < n_k_values:
            print(f"N Process: {len(processes)}/{self.ncpus}")

            # Limpieza de procesos terminados
            # processes = [p for p in processes if p.is_alive()]

            # Condición para esperar si se ha alcanzado el límite de procesos
            while len(processes) >= self.ncpus:
                time.sleep(0.1)
                processes = [p for p in processes if p.exitcode is None]

            if completed.value < n_k_values:
                try:
                    k = next(k_gen)
                    p = Process(
                        target=calcular_silhouette,
                        args=(k, data, silhouette_scores, completed, lock)
                    )

                    p.start()
                    processes.append(p)
                except StopIteration:
                    pass

            # Espera breve para evitar que el bucle consuma demasiados recursos
            time.sleep(10.0)
            print(f"N kmeans completed: {completed.value}/{n_k_values}")

        for p in processes:
            p.join()

        # print(len(k_values))
        silhouette_scores = dict(silhouette_scores)
        """

        max_score = 0
        n_opt_k = 0
        for k in silhouette_scores:
            score = silhouette_scores[k]
            if score > max_score:
                max_score = score
                n_opt_k = k

        t_f = time.time()
        execution_time = t_f - t_i
        print("Number of cluster found: {} - score: {:.3f}".format(n_opt_k, max_score))
        print("done in %.3f s" % execution_time)

        # Graficar la puntuación de silueta
        plt.plot(range(2, max_n_porous+1), list(silhouette_scores.values()), marker='o')
        plt.xlabel('Cluster number (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score to find k')
        plt.show()

        self.n_opt_k = n_opt_k
        self.compute_Kmeans(k=n_opt_k)

    def add_ions(self, ion, n_ions=None):
        # Buscando el punto mas probable
        # Número de clusters (grupos) que se espera encontrar
        # points = self.info_sim["traj"]

        # Obtener los centroides (las coordenadas más probables en cada grupo)
        centroids = self.centroids
        if n_ions is not None:
            centroids = centroids[np.random.choice(len(centroids), n_ions, replace=False)]

        print("N porous to use:", len(centroids))

        #inertia = kmeans.inertia_
        #print(f"Inertia (Sum of squared distances to the centroids): {inertia}")

        # Obtener las distancias entre los puntos y los centroides asignados
        #_, distances = pairwise_distances_argmin_min(kmeans.cluster_centers_, points)

        # Calcular la distancia promedio a los centroides
        #average_distance = np.mean(distances)
        #print(f"Distancia promedio a los centroides: {average_distance:.3e} angtroms")

        struct = self.mof.struct.copy()
        if len(ion.atoms) == 1:
            # Solo en caso de que el ion tenga un solo atomo
            element_to_add = ion.atoms.get_chemical_symbols()[0]

            for center in centroids:
                if len(center) == 2:
                    center = np.append(center, 0)

                struct.append(
                    species=Element(element_to_add),
                    coords=center,
                    coords_are_cartesian=True
                )
        else:
            elements_to_add = ion.atoms.get_chemical_symbols()
            atoms_positions = ion.atoms.get_positions()
            center_mass = ion.atoms.get_center_of_mass()
            for center in centroids:
                # news_positions = atoms_positions - center_mass + center
                # struct.append(
                #     species=Element(element_to_add),
                #     coords=center,
                #     coords_are_cartesian=True
                # )
                new_atoms_positions = rotate_molecule(atoms_positions - center_mass)
                time.sleep(0.5)
                mol = Molecule(elements_to_add, new_atoms_positions)
                for site in mol:
                    struct.append(
                        species=Element(site.species_string),
                        coords=center + site.coords,
                        coords_are_cartesian=True
                    )

        print(struct)
        struct.to(filename=f"{self.mof.name}_{ion.name}.cif"),
        print("Ion added")

    def save_state(self, file="sampling.pkl"):
        """Save simulation information."""
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        print("State saved!")

    @classmethod
    def load_state(cls, file):
        """Load simulation information from a file."""
        with open(file, 'rb') as f:
            instance = pickle.load(f)

        print("Stated loaded")
        return instance

    def volumen_porous(self):
        # Extraer etiquetas de los clústeres
        try:
            cluster_labels = self.cluster_labels
        except AttributeError:
            print("A cluster study has not yet been performed to obtain the volume of the cavity.")
            return

        traj = self.info_sim["traj"]
        # Separar los puntos de cada clúster
        clusters = {}
        for cluster_id in np.unique(cluster_labels):
            clusters[cluster_id] = traj[cluster_labels == cluster_id]

        volumes = []
        for i in clusters:
            # Crear el envolvente convexo (ConvexHull)
            hull = ConvexHull(clusters[i])
            # Obtener el volumen
            volume = hull.volume
            print(f"Volume in porous ({i}):", volume)
            volumes.append(volume)

        print(f"Mean volume in porous :", np.mean(volumes))

    def show_plots_3D(self):
        """Show results in plots 3D."""
        lattice = self.lattice
        atoms_positions = self.atoms_positions
        info_sim = self.info_sim
        try:
            centroids = self.centroids
        except AttributeError:
            print("A cluster study to visualize the centroids has not yet been performed.")
            return

        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Z -------------------------
        ax.plot(
            np.zeros(100),
            np.zeros(100),
            np.linspace(0, np.max(axis_Z), 100),
            c="black", alpha=0.8
        )

        ax.plot(
            np.max(axis_X) + np.zeros(100),
            np.zeros(100),
            np.linspace(0, np.max(axis_Z), 100),
            c="black", alpha=0.8
        )

        ax.plot(
            np.max(axis_X) + np.zeros(100),
            np.max(axis_Y) + np.zeros(100),
            np.linspace(0, np.max(axis_Z), 100),
            c="black", alpha=0.8
        )

        ax.plot(
            np.zeros(100),
            np.max(axis_Y) + np.zeros(100),
            np.linspace(0, np.max(axis_Z), 100),
            c="black", alpha=0.8
        )

        # Y -------------------------
        ax.plot(
            np.zeros(100),
            np.linspace(0, np.max(axis_Y), 100),
            np.zeros(100),
            c="black", alpha=0.8
        )

        ax.plot(
            np.max(axis_X) + np.zeros(100),
            np.linspace(0, np.max(axis_Y), 100),
            np.zeros(100),
            c="black", alpha=0.8
        )

        ax.plot(
            np.max(axis_X) + np.zeros(100),
            np.linspace(0, np.max(axis_Y), 100),
            np.max(axis_Z) + np.zeros(100),
            c="black", alpha=0.8
        )

        ax.plot(
            np.zeros(100),
            np.linspace(0, np.max(axis_Y), 100),
            np.max(axis_Z) + np.zeros(100),
            c="black", alpha=0.8
        )

        # Y -------------------------
        ax.plot(
            np.linspace(0, np.max(np.abs(axis_X)), 100),
            np.zeros(100),
            np.zeros(100),
            c="black", alpha=0.8
        )

        ax.plot(
            np.linspace(0, np.max(np.abs(axis_X)), 100),
            np.max(axis_Y) + np.zeros(100),
            np.zeros(100),
            c="black", alpha=0.8
        )

        ax.plot(
            np.linspace(0, np.max(np.abs(axis_X)), 100),
            np.max(axis_Y) + np.zeros(100),
            np.max(axis_Z) + np.zeros(100),
            c="black", alpha=0.8
        )

        ax.plot(
            np.linspace(0, np.max(np.abs(axis_X)), 100),
            np.zeros(100),
            np.max(axis_Z) + np.zeros(100),
            c="black", alpha=0.8
        )

        ax.grid(False)
        plt.show()
        """
        # Extraer etiquetas de los clústeres
        cluster_labels = self.cluster_labels

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        plot_box(ax, lattice)

        ax.scatter(
            atoms_positions[:, 0],
            atoms_positions[:, 1],
            atoms_positions[:, 2],
            c="violet",
            s=500*np.array(self.min_dist),
            edgecolor="k", alpha=0.6,
            label="MOF skeleton"
        )

        ##### generacion de malla
        traj = info_sim["traj"]
        # sample = np.random.randint(0, len(traj), 10000)
        # ax.scatter(traj[sample, 0], traj[sample, 1], traj[sample, 2], s=200, edgecolor="k", marker=".", color="blue", label="MC sampling")
        # ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], s=50, c=self.kmeans.labels_, cmap='viridis', alpha=0.5)

        # Separar los puntos de cada clúster
        clusters = {}
        for cluster_id in np.unique(cluster_labels):
            clusters[cluster_id] = traj[cluster_labels == cluster_id]

        for i in clusters:
            # Crear el envolvente convexo (ConvexHull)
            hull = ConvexHull(clusters[i])
            # Dibujar el ConvexHull con Poly3DCollection
            for simplex in hull.simplices:
                vertices = clusters[i][simplex]
                poly = Poly3DCollection([vertices], color='cyan', alpha=0.3, edgecolor='k')
                ax.add_collection3d(poly)
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=200, alpha=0.75)

        #####
        # Configuración de límites del gráfico para visualizar la celda
        lim = np.max(np.abs(lattice)) * 1.1
        ax.set_xlim([0, lim])
        ax.set_ylim([0, lim])
        ax.set_zlim([0, lim])

        # Añadir etiquetas y leyenda
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Bonus: To get rid of the grid as well:
        ax.grid(False)
        plt.show()

        """

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Definir el origen (punto de partida de los vectores de red)
        origin = np.array([0, 0, 0])
        # Trazar cada vector de red desde el origen
        ax.quiver(*origin, *lattice[0], color='r', arrow_length_ratio=0.0)
        ax.quiver(*origin, *lattice[1], color='g', arrow_length_ratio=0.0)
        ax.quiver(*origin, *lattice[2], color='b', arrow_length_ratio=0.0)

        #####
        ax.quiver(
            *(lattice[1] + origin),
            *lattice[0], color='r', arrow_length_ratio=0.0
        )

        ax.quiver(
            *(lattice[2] + origin),
            *lattice[0], color='r', arrow_length_ratio=0.0
        )

        ax.quiver(
            *(lattice[1] + lattice[2] + origin),
            *lattice[0], color='r', arrow_length_ratio=0.0
        )
        #####
        ax.quiver(
            *(lattice[0] + origin),
            *lattice[1], color='g', arrow_length_ratio=0.0
        )

        ax.quiver(
            *(lattice[2] + origin),
            *lattice[1], color='g', arrow_length_ratio=0.0
        )

        ax.quiver(
            *(lattice[0] + lattice[2] + origin),
            *lattice[1], color='g', arrow_length_ratio=0.0
        )
        #####
        ax.quiver(
            *(lattice[0] + origin),
            *lattice[2], color='b', arrow_length_ratio=0.0
        )

        ax.quiver(
            *(lattice[1] + origin),
            *lattice[2], color='b', arrow_length_ratio=0.0
        )

        ax.quiver(
            *(lattice[0] + lattice[1] + origin),
            *lattice[2], color='b', arrow_length_ratio=0.0
        )
        #####

        ax.scatter(atoms_positions[:, 0], atoms_positions[:, 1], atoms_positions[:, 2], c="violet", s=200, edgecolor="k")

        for p_i in info_sim:
            traj = info_sim[p_i]["traj"]
            sample = np.random.randint(0, len(traj), 1000)
            # ax.scatter(traj[sample, 0], traj[sample, 1], traj[sample, 2], s=200, edgecolor="k", marker=".", color="blue")
            x = traj[sample, 0]
            y = traj[sample, 1]
            z = traj[sample, 2]
            # Crear una cuadrícula regular para usar con plot_wireframe
            grid_x, grid_y = np.meshgrid(
                np.linspace(x.min(), x.max(), 30),
                np.linspace(y.min(), y.max(), 30)
            )
            # Interpolar para obtener los valores de z en la cuadrícula
            grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

            # Dibujar el wireframe
            ax.plot_wireframe(grid_x, grid_y, grid_z, color='blue', alpha=0.5)

        ax.grid(False)
        plt.show()
        """

    def show_plots_2D(self):
        """Show results in plots."""
        lattice = self.lattice
        atoms_positions = self.atoms_positions
        info_sim = self.info_sim
        n_steps = self.n_steps
        T = self.T
        step_size = self.step_size
        traj = info_sim["traj"]
        factor = self.factor

        energies = info_sim["ener"]
        rmsd = calculate_rmsd(energies)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"sim MC - nsteps: {n_steps} - T: {T} - step_size: {step_size:.2f} - factor: {factor:.2f}, RMSD: {rmsd:0.5e}")
        ax.plot(energies)
        ax.set_xlabel("accepted steps")
        ax.set_ylabel("diff energy (eV)")
        plt.show()

        # Crear las figuras para cada plano
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"sim MC - nsteps: {n_steps} - T: {T} - step_size: {step_size:.2f} - factor: {factor:.2f}")
        plot_lattice_and_atoms(axs[0], atoms_positions, traj, lattice, plane='xy')
        plot_lattice_and_atoms(axs[1], atoms_positions, traj, lattice, plane='xz')
        plot_lattice_and_atoms(axs[2], atoms_positions, traj, lattice, plane='yz')
        plt.tight_layout()
        plt.show()

        # show_axis_3D(lattice)
        # exit()

        # Crear las figuras para cada plano
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"sim MC - nsteps: {n_steps} - T: {T} - step_size: {step_size:.2f} - factor: {factor:.2f}")
        plot_densities(axs[0], atoms_positions, traj, lattice, plane='xy')
        plot_densities(axs[1], atoms_positions, traj, lattice, plane='xz')
        plot_densities(axs[2], atoms_positions, traj, lattice, plane='yz')
        plt.tight_layout()
        plt.show()
