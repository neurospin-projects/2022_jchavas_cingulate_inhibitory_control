#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Validates and clusterizes

"""
######################################################################
# Imports and global variables definitions
######################################################################
import sys
import argparse
import six
from os.path import abspath
from contrastive.evaluation.loop_validate_and_clusterize import loop_over_directory
from contrastive.evaluation.plot_loss_silhouette_score_vs_latent_dimension import plot_loss_silhouette_score

def parse_args(argv):
    """Parses command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='synthesize_results.py',
        description='Analyzes all deep lerning subdirectories')
    parser.add_argument(
        "-s", "--src_dir", type=str, required=True,
        help='Source deep learning directory.')
    parser.add_argument(
        "-c", "--csv_file", type=str, required=True,
        help='csv file on which is done the evaluation.')

    args = parser.parse_args(argv)

    return args


def main(argv):
    """Reads argument line and launches postprocessing_results on each

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        args = parse_args(argv)
        src_dir = abspath(args.src_dir)
        csv_file = abspath(args.csv_file)

        loop_over_directory(src_dir, csv_file)
        plot_loss_silhouette_score(src_dir)
    except SystemExit as exc:
        if exc.code != 0:
            six.reraise(*sys.exc_info())


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
