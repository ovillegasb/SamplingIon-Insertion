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
from pymatgen.core import Structure
from tqdm import tqdm
import spglib
import warnings

warnings.simplefilter("ignore", RuntimeWarning)

angstroms_per_meter = 1e10


def pymatgen2ase(struc):
    atoms = Atoms(symbols=struc.atomic_numbers, cell=struc.lattice.matrix)
    atoms.set_scaled_positions(struc.frac_coords)
    return atoms


def ase2pymatgen(struc):
    lattice = struc.get_cell()
    coordinates = struc.get_scaled_positions()
    species = struc.get_chemical_symbols()
    return Structure(lattice, species, coordinates)


def symmetrize_structure(struct, initial_symprec=1e-3, final_symprec=1e-8):
    ase_atoms = pymatgen2ase(struct)
    # Convertir la estructura a un formato compatible con spglib
    cell = (ase_atoms.get_cell(), ase_atoms.get_scaled_positions(), ase_atoms.get_atomic_numbers())

    # Aplicar la simetría para simetrizar la estructura
    symmetrized_cell = spglib.standardize_cell(cell, to_primitive=False, no_idealize=False, symprec=final_symprec)

    spacegroup = spglib.get_spacegroup(symmetrized_cell, symprec=initial_symprec, symbol_type=0)
    print("Spacegroup Symmetrized:", spacegroup)

    refine_cell = spglib.refine_cell(symmetrized_cell, final_symprec)
    spacegroup = spglib.get_spacegroup(refine_cell, symprec=initial_symprec, symbol_type=0)
    print("Spacegroup refine:", spacegroup)

    ase_atoms.set_cell(refine_cell[0])
    ase_atoms.set_scaled_positions(refine_cell[1])

    return ase2pymatgen(ase_atoms)


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


def plot_clusters2(ax, atoms, samples, centroids, lattice, plane='xy'):
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
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75)
    elif plane == 'xz':
        ax.scatter(atoms[:, 0], atoms[:, 2], c="violet", s=100, edgecolor="k")
        ax.scatter(centroids[:, 0], centroids[:, 2], c='red', s=200, alpha=0.75)
    elif plane == 'yz':
        ax.scatter(atoms[:, 1], atoms[:, 2], c="violet", s=100, edgecolor="k")
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


def compute_3d_density_histogram(positions, lattice_matrix, nx, ny, nz):
    """
    Calcula un histograma de densidad 3D en una celda unitaria.

    Parámetros:
    - positions: numpy array de forma (N, 3) con las posiciones de las partículas.
    - lattice_matrix: numpy array de forma (3, 3) que define la celda unitaria.
    - nx, ny, nz: número de divisiones en x, y y z respectivamente.

    Retorna:
    - hist: el histograma 3D con las densidades.
    - edges: los bordes de los bins en cada dimensión.
    """
    
    # Convertir posiciones a coordenadas dentro de la celda unitaria usando coordenadas fraccionales
    fractional_positions = np.linalg.solve(lattice_matrix.T, positions.T).T % 1.0

    # Crear el histograma de densidad 3D
    hist, edges = np.histogramdd(
        fractional_positions,
        bins=(nx, ny, nz),
        range=((0, 1), (0, 1), (0, 1))
    )

    return hist, edges


def plot_density_3d(hist, edges, density_threshold=5):
    """
    Visualiza el histograma 3D de densidad, mostrando solo los puntos con densidad
    superior a un umbral.

    Parámetros:
    - hist: el histograma 3D de densidad.
    - edges: los bordes de los bins en cada dimensión.
    - density_threshold: mínimo valor de densidad para visualizar un voxel.
    """
    # Obtenemos los centros de los bins en cada dimensión
    x_centers = (edges[0][1:] + edges[0][:-1]) / 2
    y_centers = (edges[1][1:] + edges[1][:-1]) / 2
    z_centers = (edges[2][1:] + edges[2][:-1]) / 2

    # Creamos una grilla de coordenadas 3D a partir de los centros de los bins
    x, y, z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

    # Filtramos las posiciones con densidad superior al umbral
    mask = hist > density_threshold
    x_points, y_points, z_points = x[mask], y[mask], z[mask]
    densities = hist[mask]

    # Graficamos los puntos con densidad mayor al umbral
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type='ortho')
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    img = ax.scatter(x_points, y_points, z_points, c=densities, cmap="viridis_r", marker="s", s=100)
    plt.colorbar(img, ax=ax, label="Densidad")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def vector_pbc_general(r1, r2, lattice_matrix):
    """
    Calcula el vector de desplazamiento ajustado a través de condiciones periódicas generales.

    Parámetros:
    - r1: vector de posición inicial (en coordenadas cartesianas).
    - r2: vector de posición final (en coordenadas cartesianas).
    - lattice_matrix: matriz de la celda (3x3 numpy array).
    
    Retorna:
    - Vector de desplazamiento ajustado en coordenadas cartesianas.
    """
    # Convertir las posiciones a coordenadas fraccionarias
    lattice_inv = np.linalg.inv(lattice_matrix)
    fractional_r1 = r1 @ lattice_inv
    fractional_r2 = r2 @ lattice_inv
    
    # Calcula el vector de desplazamiento en coordenadas fraccionarias
    fractional_displacement = fractional_r2 - fractional_r1
    
    # Aplica la convención de imagen mínima en coordenadas fraccionarias
    fractional_displacement = fractional_displacement - np.round(fractional_displacement)
    
    # Convierte el desplazamiento ajustado de regreso a coordenadas cartesianas
    cartesian_displacement = fractional_displacement @ lattice_matrix
    return cartesian_displacement


def periodic_distance2(r1, r2, lattice_matrix):
    return np.linalg.norm(vector_pbc_general(r1, r2, lattice_matrix))


def make_cluster_label(points, centroids, lattice_matrix):
    """
    Clasifica cada punto según el centroide más cercano en condiciones periódicas.
    
    Parámetros:
    - points: array de puntos en coordenadas cartesianas, de tamaño (N, 3).
    - centroids: array de posiciones de centroides en coordenadas cartesianas, de tamaño (M, 3).
    - lattice_matrix: matriz de la celda (3x3 numpy array).
    
    Retorna:
    - cluster_labels: lista con la etiqueta de cluster para cada punto (tamaño N).
    """
    cluster_labels = []
    for point in points:
        # Calcula la distancia a cada centroide bajo condiciones periódicas
        distances = [periodic_distance2(point, centroid, lattice_matrix) for centroid in centroids]
        
        # Encuentra el índice del centroide más cercano
        closest_centroid = np.argmin(distances)
        cluster_labels.append(closest_centroid)
    
    return cluster_labels


def periodic_distance(point, other_point, lattice_matrix):
    """
    Calcula la distancia mínima periódica entre dos puntos en un sistema periódico,
    considerando la celda unitaria y sus imágenes.

    Parámetros:
    - point: numpy array (3,) con coordenadas cartesianas del primer punto.
    - other_point: numpy array (3,) con coordenadas cartesianas del segundo punto.
    - lattice_matrix: numpy array (3, 3) que define la celda unitaria.

    Retorna:
    - La distancia periódica mínima entre los dos puntos.
    """
    # Convertir el segundo punto a coordenadas fraccionales
    frac_other = np.linalg.solve(lattice_matrix.T, other_point)

    # Considerar desplazamientos de la celda en [-1, 0, 1] en coordenadas fraccionales
    min_dist = np.inf
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                displacement = np.array([dx, dy, dz])
                shifted_frac_other = frac_other + displacement
                # Convertir a coordenadas cartesianas
                shifted_cartesian_other = np.dot(shifted_frac_other, lattice_matrix)
                dist = np.linalg.norm(shifted_cartesian_other - point)
                min_dist = min(min_dist, dist)

    return min_dist


def find_top_n_dense_regions_with_min_distance(hist, edges, lattice_matrix, n, min_distance):
    """
    Encuentra los N índices más densos en el histograma 3D, manteniendo una distancia mínima periódica entre ellos.

    Parámetros:
    - hist: el histograma 3D con las densidades.
    - edges: los bordes de los bins en cada dimensión.
    - lattice_matrix: numpy array de forma (3, 3) que define la celda unitaria.
    - n: número de regiones más densas a encontrar.
    - min_distance: distancia mínima en Å entre las regiones densas.

    Retorna:
    - selected_indices: lista de los índices de los N voxeles más densos que cumplen con la distancia mínima.
    - selected_cartesian_centers: lista de los centros de los N voxeles más densos en coordenadas cartesianas.
    - selected_densities: lista de las densidades correspondientes a estos N índices.
    """
    # Aplanar el histograma y ordenar los índices por densidad
    flat_hist = hist.ravel()
    print(flat_hist, len(flat_hist))
    sorted_indices_flat = np.argsort(flat_hist)[::-1]  # Índices en orden descendente de densidad
    print(sorted_indices_flat, len(sorted_indices_flat))
    
    selected_indices = []
    selected_cartesian_centers = []
    selected_densities = []
    
    print("N porous to find:", n)
    for idx_flat in sorted_indices_flat:
        print("idx_flat", idx_flat)
        if len(selected_indices) >= n:
            break

        # print(selected_cartesian_centers)
        
        # Convertir índice plano a 3D
        idx_3d = np.unravel_index(idx_flat, hist.shape)
        print("idx_3d", idx_3d)
        
        # Calcular el centro de este voxel en coordenadas fraccionales
        frac_coords = [(edges[i][idx_3d[i]] + edges[i][idx_3d[i] + 1]) / 2 for i in range(3)]
        print("frac_coords:", frac_coords)
        # Convertir a coordenadas cartesianas
        cartesian_coords = np.dot(frac_coords, lattice_matrix)
        print("cartesian_coords:", cartesian_coords)
        
        # Comprobar la distancia mínima periódica respecto a los puntos seleccionados
        too_close = False
        for selected_point in selected_cartesian_centers:
            dist = periodic_distance(cartesian_coords, selected_point, lattice_matrix)
            print("Distance:", dist)
            cart_coord_desp = vector_pbc_general(cartesian_coords, selected_point, lattice_matrix)
            # print(cart_coord_desp)
            # print(np.linalg.norm(cart_coord_desp))
            dist = np.linalg.norm(cart_coord_desp)
            if dist < min_distance:
                too_close = True
                break

        # print("too close:", too_close)
        # print("min_distance:", min_distance)
        
        if not too_close:
            selected_indices.append(idx_3d)
            selected_cartesian_centers.append(cartesian_coords)
            selected_densities.append(hist[idx_3d])

    return selected_indices, selected_cartesian_centers, selected_densities


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


def V_electric_potential(r, r_center, lattice, min_dist=0.0, int_type=1, radius_ion=0.2):
    """Calculate the electric potential using a cutoff distance (unit Volt)."""
    # OLD
    #r_values = r - r_center
    #r_values = minimum_image_convention(r_values, lattice)
    #r_norm = np.linalg.norm(r_values)
    # NEW
    r_norm = periodic_distance2(r, r_center, lattice)
    r_norm -= radius_ion

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
    allhistory = [r_current]
    energies = []
    n_accept = 0
    current_cpu = task_function()
    step = 0
    ###NEW
    # system = Atoms(positions=[r_current], pbc=True, cell=lattice)
    # print(system)
    # system.write("test.cif")
    ###NEW

    pbar = tqdm(desc="MC sim", total=n_steps)
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
            history.append(r_current)

        if not np.isinf(delta_V):
            energies.append(e*delta_V)

        # print(f"|{current_cpu}|", step, r_new, "|", f"{delta_V:.3e} volt", "|", f"{e*delta_V:.3e} eV")

        # # TODO Dynamic stepsize
        # # Ejemplo de ajuste dinámico del tamaño del paso:
        # if step % 100 == 0:  # Cada 100 pasos
        #     acceptance_rate = n_accept / (step + 1)
        #     if acceptance_rate < 0.3:
        #         step_size *= 0.9  # Reduce el tamaño del paso si la tasa de aceptación es baja
        #     elif acceptance_rate > 0.7:
        #         step_size *= 1.1  # Aumenta el tamaño del paso si la tasa es alta

        allhistory.append(r_current)
        step += 1
        # print(step, end="\r")
        pbar.update(1)

    pbar.close()

    acceptance_rate = n_accept / n_steps
    print(f"|{current_cpu}| Acceptance rate: {acceptance_rate:.2f}")

    energies = np.array(energies)
    energies = np.where(np.isinf(energies), np.nan, energies)

    return np.array(history), energies, acceptance_rate, np.array(allhistory)


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
        [0, 0, 0],  # Original cell
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
        [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
        [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
        [1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, 1, -1]
    ])

    # Expandimos las posiciones usando desplazamientos periódicos
    expanded_positions = []
    for offset in offsets:
        displacement = np.dot(offset, lattice_matrix)
        expanded_positions.append(positions + displacement)
    expanded_positions = np.vstack(expanded_positions)

    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # plot_lattice_and_atoms(axs[0], np.array([[0, 0, 0]]), expanded_positions, lattice_matrix, plane='xy')
    # plot_lattice_and_atoms(axs[1], np.array([[0, 0, 0]]), expanded_positions, lattice_matrix, plane='xz')
    # plot_lattice_and_atoms(axs[2], np.array([[0, 0, 0]]), expanded_positions, lattice_matrix, plane='yz')
    # plt.tight_layout()
    # plt.show()


    ######
    """
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
    ax.scatter(
        expanded_positions[:, 0],
        expanded_positions[:, 1],
        expanded_positions[:, 2],
        c="blue", marker=".", alpha=0.4,
    )
    # Añadir etiquetas y leyenda
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    # Bonus: To get rid of the grid as well:
    ax.grid(False)
    plt.show()
    """
    ######
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    # Ejecutamos KMeans sobre las posiciones expandidas
    kmeans.fit(expanded_positions)

    # Calculamos el `silhouette_score` usando solo las posiciones originales
    original_labels = kmeans.labels_[:len(positions)]
    centroids = kmeans.cluster_centers_
    # print("")
    # print(lattice_matrix)
    inv_lattice_matrix = np.linalg.inv(lattice_matrix)
    # print(inv_lattice_matrix)
    # print(centroids)
    # Convertir los centros de clústeres a coordenadas fraccionarias
    # fractional_centroids = np.linalg.solve(lattice_matrix.T, centroids.T).T
    # # print(fractional_centroids)
    fractional_centroids = np.dot(inv_lattice_matrix, centroids.T).T
    # print(fractional_centroids)
    fractional_centroids = fractional_centroids % 1.0
    # print(fractional_centroids)

    # Ajustar a la celda unitaria aplicando el módulo 1 en las coordenadas fraccionarias
    # fractional_centroids = fractional_centroids % 1.0
    # Convertir de regreso a coordenadas cartesianas
    # adjusted_centroids = fractional_centroids @ lattice_matrix
    adjusted_centroids = np.dot(fractional_centroids, lattice_matrix)
    # print(adjusted_centroids)

    # Eliminar duplicados dentro de una tolerancia pequeña
    unique_centroids = np.unique(np.round(adjusted_centroids, decimals=5), axis=0)
    # print(unique_centroids)

    # Seleccionamos las etiquetas de los clústeres para las posiciones originales
    original_labels = kmeans.labels_[:len(positions)]

    ######
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
    # ax.scatter(
    #     expanded_positions[:, 0],
    #     expanded_positions[:, 1],
    #     expanded_positions[:, 2],
    #     c="blue", marker=".", alpha=0.4,
    # )
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c="blue", marker=".", alpha=0.4,
    )
    ax.scatter(
        unique_centroids[:, 0],
        unique_centroids[:, 1],
        unique_centroids[:, 2],
        c="violet", s=200, marker=".", alpha=1.0,
    )

    plot_box(ax, lattice_matrix)
    # Añadir etiquetas y leyenda
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Bonus: To get rid of the grid as well:
    ax.grid(False)
    plt.show()
    ######

    # Calculamos el `silhouette_score` usando las posiciones originales
    score = silhouette_score(expanded_positions[:len(positions)], original_labels)

    return score, kmeans, unique_centroids, original_labels


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
            trajectory, energies, acceptance_rate, allhistory = monte_carlo_metropolis(
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
            info_sim["all"] = allhistory
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
        print("Using the better kmeans results, k:", k)
        print(self.silhouette_scores)
        # kmeans = self.silhouette_scores[k]
        # Convertimos los centros de vuelta a la celda original
        # print(kmeans.cluster_centers_)
        # centers = kmeans.cluster_centers_ % lattice_matrix.diagonal()
        centroids = self.silhouette_scores[k]["centroids"]
        self.centroids = centroids
        self.cluster_labels = self.silhouette_scores[k]["cluster_labels"]
        points = self.info_sim["traj"]
        # Aplicar el algoritmo k-means
        #kmeans = KMeans(n_clusters=k)
        #kmeans.fit(points)
        # self.centroids, self.cluster_labels = apply_periodic_kmeans(points, self.lattice, k)
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


    def compute_Kmeans2(self, k=4):
        print("Using the better kmeans results, k:", k)
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
        points = self.info_sim["traj"]
        silhouette_scores = {}
        for k in range(2, max_n_porous+1):
            print("Studying cluster number:", k, end=" - ")
            silhouette_scores[k] = {}
            t_i_c = time.time()
            # kmeans = KMeans(n_clusters=k, random_state=0)
            # _, labels = apply_periodic_kmeans(positions, lattice_matrix, n_clusters, random_state=0)
            score, kmeans, centroids, cluster_labels = periodic_silhouette_score(points, self.lattice, k, random_state=0)
            # labels = kmeans.fit_predict(data)
            # score = silhouette_score(data, labels)
            silhouette_scores[k]["score"] = score
            silhouette_scores[k]["kmeans"] = kmeans
            silhouette_scores[k]["centroids"] = centroids
            silhouette_scores[k]["cluster_labels"] = cluster_labels
            t_f_c = time.time()
            execution_time_c = t_f_c - t_i_c
            print("done %.3f" % execution_time_c)

        max_score = 0
        n_opt_k = 0

        for k in silhouette_scores:
            score = silhouette_scores[k]["score"]
            if score > max_score:
                max_score = score
                n_opt_k = k

        t_f = time.time()
        execution_time = t_f - t_i
        print("Number of cluster found: {} - score: {:.3f}".format(n_opt_k, max_score))
        print("done in %.3f s" % execution_time)

        self.n_opt_k = n_opt_k
        self.silhouette_scores = silhouette_scores.copy()
        self.compute_Kmeans(k=n_opt_k)



    def clusters_study2(self, max_n_porous=8):
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

    def find_centers(self, max_n_porous=8):
        """Search for more probable centroids.."""
        print("N centers to search:", max_n_porous)
        points = self.info_sim["traj"]
        lattice = self.lattice
        binwidth = .5  # angstroms

        # build 3D histogram
        # nx, ny, nz = (5, 5, 5)
        magnitudes = np.linalg.norm(lattice, axis=1)
        nx, ny, nz = np.round(magnitudes / binwidth + 0.5).astype(np.int64)
        hist, edges = compute_3d_density_histogram(
            points,
            lattice,
            nx, ny, nz
        )

        plot_density_3d(hist, edges, density_threshold=5)

        ######
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

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c="blue", marker=".", alpha=0.4,
        )

        # Añadir etiquetas y leyenda
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Bonus: To get rid of the grid as well:
        ax.grid(False)
        plt.show()
        ######

        n_steps = self.n_steps
        T = self.T
        step_size = self.step_size
        factor = self.factor
        
        #print(hist)
        #print(hist.shape)
        #print(edges)
        #print(len(edges))

        # Encontrar el índice del voxel con la densidad más alta
        max_density_idx = np.unravel_index(np.argmax(hist), hist.shape)
        print(f"Índice del voxel más denso: {max_density_idx}")
        print(f"Densidad máxima en el voxel: {hist[max_density_idx]}")

        # Encontrar los N puntos más densos
        # n = 5  # Número de regiones más densas a obtener
        top_n_indices, top_n_cartesian_centers, top_n_densities = find_top_n_dense_regions_with_min_distance(
            hist,
            edges,
            lattice,
            max_n_porous,
            min_distance=6.0
        )

        print("Índices de los voxeles más densos:", top_n_indices)
        print("Centros en coordenadas cartesianas de las regiones más densas:", top_n_cartesian_centers)
        print("Densidades de las regiones más densas:", top_n_densities)
        centroids = np.array(top_n_cartesian_centers)
        cluster_labels = make_cluster_label(points, centroids, lattice)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"sim MC - nsteps: {n_steps} - T: {T} - step_size: {step_size:.2f} - factor: {factor:.2f}")
        plot_clusters(axs[0], self.atoms_positions, points, np.array(top_n_cartesian_centers), cluster_labels, self.lattice, plane='xy')
        plot_clusters(axs[1], self.atoms_positions, points, np.array(top_n_cartesian_centers), cluster_labels, self.lattice, plane='xz')
        plot_clusters(axs[2], self.atoms_positions, points, np.array(top_n_cartesian_centers), cluster_labels, self.lattice, plane='yz')
        plt.tight_layout()
        plt.show()

        exit()

        v1, v2, v3 = lattice
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        plot_box(ax, lattice)
        ax.scatter(
            self.atoms_positions[:, 0],
            self.atoms_positions[:, 1],
            self.atoms_positions[:, 2],
            c="violet",
            s=500*np.array(self.min_dist),
            edgecolor="k", alpha=0.8,
            label="MOF skeleton"
        )

        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c='red',
            s=200,
            alpha=1.
        )

        # Etiquetas y leyenda
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.legend()
        plt.show()
        self.centroids = centroids
        self.cluster_labels = cluster_labels

    def add_ions(self, ion, n_ions=None):
        # Buscando el punto mas probable
        # Número de clusters (grupos) que se espera encontrar
        # points = self.info_sim["traj"]

        # Obtener los centroides (las coordenadas más probables en cada grupo)
        centroids = self.centroids
        cluster_labels = self.cluster_labels

        print("N porous to use:", len(centroids))

        if n_ions != len(centroids):
            print("Using a new cluster centers")
            centroids = self.silhouette_scores[n_ions]["centroids"]

        if n_ions is not None:
            centroids = centroids[np.random.choice(len(centroids), n_ions, replace=False)]

        points = self.info_sim["traj"]
        #####fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        #####fig.suptitle(f"sim MC - nsteps: {self.n_steps} - T: {self.T} - step_size: {self.step_size:.2f} - factor: {self.factor:.2f}")
        #####plot_clusters(axs[0], self.atoms_positions, points, centroids, cluster_labels, self.lattice, plane='xy')
        #####plot_clusters(axs[1], self.atoms_positions, points, centroids, cluster_labels, self.lattice, plane='xz')
        #####plot_clusters(axs[2], self.atoms_positions, points, centroids, cluster_labels, self.lattice, plane='yz')
        #####plt.tight_layout()
        #####plt.show()

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
        symm_struct = symmetrize_structure(struct)
        struct.to(filename=f"{self.mof.name}_{ion.name}.cif")
        symm_struct.to(filename=f"{self.mof.name}_{ion.name}_symm.cif"),
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
        print(f"Sum volume in porous :", np.sum(volumes))

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
        acc = info_sim["acc_rate"]
        factor = self.factor

        energies = info_sim["ener"]
        rmsd = calculate_rmsd(energies)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"sim MC - nsteps: {n_steps} - T: {T} - step_size: {step_size:.2f} - factor: {factor:.2f}, RMSD: {rmsd:0.5e}, ACC: {acc:0.5f}")
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
