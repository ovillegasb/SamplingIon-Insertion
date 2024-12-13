"""Module that defines the structures to be used."""

from pymatgen.core import Structure
from ase.data import covalent_radii
import ase.io


class MOF:
    """Object representing a MOF."""

    def __init__(self, file):
        """Initialize class by reading the structure."""
        self.file = file
        self.name = file.split("/")[-1].split(".")[0]

        # Get the structure
        struct = Structure.from_file(file)
        self.struct = struct
        self.cart_coords = struct.cart_coords
        self.lattice = struct.lattice.matrix
        # angstroms^3
        self.volume = struct.volume

        # Get atoms numbers
        self.atomic_numbers = [site.specie.number for site in struct]
        self.covalent_radii = [covalent_radii[site.specie.number] for site in struct]
        interaction_type = []
        for number in self.atomic_numbers:
            if number in [-8]:
                interaction_type.append(-1)
            else:
                interaction_type.append(+1)

        self.interaction_type = interaction_type

    def __str__(self):
        """MOF information."""
        info = "MOF class\n"
        info += f"File: {self.file}\n"
        return info


class ION:
    """Object representing a ION (atomic or molecular)."""

    def __init__(self, file):
        """Initialize class by reading the structure."""
        self.file = file
        self.name = file.split("/")[-1].split(".")[0]
        self.atoms = ase.io.read(file)

    def __str__(self):
        """ION information."""
        info = "ION class\n"
        info += f"File: {self.file}\n"
        return info
