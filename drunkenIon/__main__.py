#!/bin/env python
# -*- coding: utf-8 -*-


"""Run DrukenIon in command line."""

import argparse
from drunkenIon.core import TITLE
from drunkenIon.structures import MOF, ION
from drunkenIon.sampling import DrunkenIon


def options():
    """Generate command line interface."""
    parser = argparse.ArgumentParser(
        prog="tobacco",
        usage="%(prog)s [-options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Enjoy the program!"
    )

    fileinput = parser.add_argument_group(
        "\033[1;36mInitial settings\033[m")

    fileinput.add_argument(
        "-i", "--input",
        help="Specify the input crystal file.",
        type=str,
        default=None,
        dest="mof",
        metavar="crystal.cif"
    )

    RunSampling = parser.add_argument_group(
        "\033[1;36mOptions to run sampling\033[m")

    RunSampling.add_argument(
        "--min_dist",
        help="Minimal distance in anstroms between test particle and MOF atoms.",
        type=float,
        default=None,
        metavar="value"
    )

    RunSampling.add_argument(
        "-step_size",
        help="Step size in anstroms.",
        type=float,
        default=None,
        metavar="value"
    )

    RunSampling.add_argument(
        "-n_steps",
        help="Number of simulation steps.",
        type=int,
        default=100,
        metavar="steps"
    )

    RunSampling.add_argument(
        "-n_cpu",
        help="Number of cpus.",
        type=int,
        default=1,
        metavar="n",
        dest="cpu"
    )

    RunSampling.add_argument(
        "-T",
        help="Temperature (k_B * 300K ~ 0.02585 eV).",
        type=float,
        default=0.1,
        metavar="temp"
    )

    RunSampling.add_argument(
        "-factor",
        help="Scaled Covalent radii (1.0).",
        type=float,
        default=1.0,
        metavar="factor"
    )

    RunSampling.add_argument(
        "--run_sampling",
        help="Run sampling on the MOF structure.",
        action="store_true",
        default=False
    )

    RunSampling.add_argument(
        "--show_plots",
        help="Show plots 2D and 3D.",
        action="store_true",
        default=False
    )

    RunSampling.add_argument(
        "--cluster_study",
        help="A kmeans study is make.",
        action="store_true",
        default=False
    )

    RunSampling.add_argument(
        "--hist_study",
        help="A hist 3D study is make.",
        action="store_true",
        default=False
    )

    RunSampling.add_argument(
        "-load",
        help="Load results from a file.",
        type=str,
        metavar="file.pkl",
        default=None
    )

    RunSampling.add_argument(
        "-n_porous",
        help="Maximum number of pores to search.",
        type=int,
        metavar="N",
        default=11
    )

    RunSampling.add_argument(
        "-n_centers",
        help="Specify number of centroids to search.",
        type=int,
        metavar="N",
        default=None
    )

    RunSampling.add_argument(
        "--save_plots",
        help="Save 2D and 3D graphs of the process evolution.",
        action="store_true",
        default=False
    )

    AddIon = parser.add_argument_group(
        "\033[1;36mOptions for adding the ion to the system\033[m")

    AddIon.add_argument(
        "-ion", "--add_ion",
        help="Specify the input ion file.",
        type=str,
        default=None,
        dest="ion",
        metavar="ion.xyz"
    )

    AddIon.add_argument(
        "-n_ions",
        help="Number of ions to incorporate.",
        type=int,
        metavar="N_ions",
        default=None
    )

    return vars(parser.parse_args())


def main():
    """Run main function."""
    print(TITLE)
    args = options()
    if args["mof"] is not None:
        # load the mof
        mof = MOF(args["mof"])
        print(mof)

    if args["run_sampling"]:
        alignIon = DrunkenIon(
            mof=mof,
            min_dist=args["min_dist"],
            n_steps=args["n_steps"],
            step_size=args["step_size"],
            T=args["T"],  # k_B * 300K ~ 0.02585 eV
            ncpus=args["cpu"],
            factor=args["factor"]
        )
        alignIon.run_montecarlo()
        alignIon.save_state()

    if args["load"] is not None:
        alignIon = DrunkenIon.load_state(file=args["load"])
        alignIon.ncpus = args["cpu"]

    if args["cluster_study"]:
        alignIon.clusters_study(max_n_porous=args["n_porous"])
        alignIon.save_state()

    if not args["cluster_study"] and args["n_centers"] is not None:
        alignIon.compute_Kmeans(k=args["n_centers"])
        alignIon.save_state()

    if args["hist_study"]:
        alignIon.find_centers(max_n_porous=args["n_porous"])
        alignIon.save_state()

    if args["show_plots"]:
        alignIon.show_plots_2D()
        alignIon.show_plots_3D()

    if args["ion"] is not None:
        ion = ION(args["ion"])
        print(ion)
        alignIon.add_ions(ion, args["n_ions"])

    alignIon.volumen_porous()
    print("Done!")


# RUN
if __name__ == '__main__':
    main()
