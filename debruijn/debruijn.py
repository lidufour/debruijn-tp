#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
    draw_networkx_nodes,
    draw_networkx_edges,
)
import matplotlib
from operator import itemgetter
import random

random.seed(9001)
from random import randint
from statistics import stdev, mean
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

matplotlib.use("Agg")

__author__ = "Laura DUFOUR"
__copyright__ = "Universite Paris Cité"
__credits__ = ["Laura DUFOUR"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Laura DUFOUR"
__email__ = "laura.dufour@etu.u-paris.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """

    with open(fastq_file, 'r') as f:
        for line in f:
            yield next(f).strip() #on skip la 1e ligne qui correspond au header
            next(f) #on skip les deux lignes suivantes
            next(f)


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(0, len(read)-kmer_size+1):
        current_read = read[i:i+kmer_size]
        yield current_read


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    read_list = list(read_fastq(fastq_file))
    for read in read_list:
        kmer_list = list(cut_kmer(read, kmer_size))
        for kmer in kmer_list:
            if kmer not in kmer_dict.keys():
                kmer_dict[kmer] = 1 # on initialise le kmer, on lui associe une occurence
            else:
                kmer_dict[kmer] += 1 # on incrémente de 1 son occurence s'il existe déjà
    return kmer_dict

def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()

    for kmer, weight in kmer_dict.items():
        # print(kmer, weight)
        prefix = kmer[:-1]
        suffix = kmer[1:]
        # print(f"prefix : {prefix}\nsuffix: {suffix}")
        graph.add_edge(prefix, suffix, weight=weight)


    return graph
    


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    input_graph = graph.copy()
    nodes_to_remove = set()

    for path in path_list:
        if not path:
            continue

        start_idx = 0 if delete_entry_node else 1
        end_idx = len(path) if delete_sink_node else len(path) - 1

        nodes_to_remove.update(path[start_idx:end_idx])

    nodes_present = [n for n in nodes_to_remove if n in input_graph]
    input_graph.remove_nodes_from(nodes_present)

    return input_graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if not path_list:
        return graph

    # Choix du meilleur chemin
    if len(weight_avg_list) > 1 and stdev(weight_avg_list) > 0:
        best_idx = max(range(len(path_list)), key=lambda i: weight_avg_list[i])
    elif len(path_length) > 1 and stdev(path_length) > 0:
        best_idx = max(range(len(path_list)), key=lambda i: path_length[i])
    else:
        best_idx = randint(0, len(path_list) - 1)

    G = graph.copy()

    # Suppressions ciblées selon les flags
    for i, path in enumerate(path_list):
        if i == best_idx or not path:
            continue

        if delete_entry_node and not delete_sink_node:
            # couper la pointe d'entrée : on enlève tout sauf le dernier (noeud commun d'arrivée)
            nodes_to_remove = path[:-1]
        elif delete_sink_node and not delete_entry_node:
            # couper la pointe de sortie : on enlève tout sauf le premier (noeud commun de départ)
            nodes_to_remove = path[1:]
        elif not delete_entry_node and not delete_sink_node:
            # seulement les noeuds internes
            nodes_to_remove = path[1:-1]
        else:  # delete_entry_node and delete_sink_node
            nodes_to_remove = path[:]

        G.remove_nodes_from([n for n in nodes_to_remove if n in G])

    return G


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    paths: List[List[str]] = list(all_simple_paths(graph, ancestor_node, descendant_node))
    if len(paths) <= 1:
        return graph

    lengths = [len(p) for p in paths]
    weights = [path_average_weight(graph, p) for p in paths]  # uses Graph.subgraph(path).edges(data=True)

    return select_best_path(
        graph,
        path_list=paths,
        path_length=lengths,
        weight_avg_list=weights,
        delete_entry_node=False,
        delete_sink_node=False,
    )


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    for n in list(graph.nodes):  # itérer sur une copie car le graphe peut être modifié
        preds = list(graph.predecessors(n))  # graph.predecessors(node)
        if len(preds) > 1:
            # combinaisons uniques (i, j) sans itertools
            for i in range(len(preds) - 1):
                for j in range(i + 1, len(preds)):
                    anc = lowest_common_ancestor(graph, preds[i], preds[j])  # LCA
                    if anc is not None and anc != n:
                        # On a détecté une bulle entre anc (ancêtre) et n (descendant)
                        new_graph = solve_bubble(graph, anc, n)
                        # La simplification peut supprimer des nœuds/arrêtes -> récursif
                        return simplify_bubbles(new_graph)

    # Aucune bulle détectée
    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for n in list(graph.nodes):
        # Sélection des start nodes qui atteignent n
        reachable_starts = []
        for s in starting_nodes:
            if s in graph and n in graph and has_path(graph, s, n):  # nx.has_path
                reachable_starts.append(s)

        if len(reachable_starts) >= 2:
            # Construire tous les chemins simples start -> n
            paths, lengths, weights = [], [], []
            for s in reachable_starts:
                for p in all_simple_paths(graph, s, n):  # nx.all_simple_paths
                    paths.append(p)
                    lengths.append(len(p))
                    # Moyenne des poids des arêtes du chemin via Graph.subgraph(path).edges(data=True)
                    ew = []
                    for u, v, d in graph.subgraph(p).edges(data=True):
                        ew.append(d.get("weight", 1))
                    weights.append(sum(ew) / len(ew) if ew else 0.0)

            if len(paths) > 1:
                # On supprime les mauvaises pointes d'entrée (on ne supprime pas le noeud de sortie)
                new_graph = select_best_path(
                    graph,
                    path_list=paths,
                    path_length=lengths,
                    weight_avg_list=weights,
                    delete_entry_node=True,   # supprimer les bouts d'entrée indésirables
                    delete_sink_node=False,   # conserver le noeud d'arrivée commun
                )
                # La simplification modifie le graphe -> récursion
                return solve_entry_tips(new_graph, starting_nodes)

    # Rien à simplifier
    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for n in list(graph.nodes):  # itérer sur une copie car le graphe peut être modifié
        # Sélection des end nodes atteignables depuis n
        reachable_ends = []
        for e in ending_nodes:
            if n in graph and e in graph and has_path(graph, n, e):  # nx.has_path
                reachable_ends.append(e)

        if len(reachable_ends) >= 2:
            # Construire tous les chemins simples n -> end
            paths, lengths, weights = [], [], []
            for e in reachable_ends:
                for p in all_simple_paths(graph, n, e):  # nx.all_simple_paths
                    paths.append(p)
                    lengths.append(len(p))
                    # Moyenne des poids via Graph.subgraph(path).edges(data=True)
                    ew = []
                    for u, v, d in graph.subgraph(p).edges(data=True):
                        ew.append(d.get("weight", 1))
                    weights.append(sum(ew) / len(ew) if ew else 0.0)

            if len(paths) > 1:
                # Supprimer les mauvaises pointes de sortie :
                # ne PAS supprimer le noeud de départ commun (entry), mais supprimer les sinks indésirables
                new_graph = select_best_path(
                    graph,
                    path_list=paths,
                    path_length=lengths,
                    weight_avg_list=weights,
                    delete_entry_node=False,  # conserver le noeud de départ commun
                    delete_sink_node=True,    # supprimer les noeuds de sortie des chemins non retenus
                )
                # La simplification modifie le graphe => récursif
                return solve_out_tips(new_graph, ending_nodes)

    # Rien à simplifier
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starts: List[str] = []
    for n in graph.nodes:
        it = graph.predecessors(n)
        try:
            next(it)
            has_pred = True
        except StopIteration:
            has_pred = False

        if not has_pred:
            starts.append(n)

    return sorted(starts)


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sinks: List[str] = []
    for node in graph.nodes:
        if len(list(graph.successors(node))) == 0:
            sinks.append(node)
    return sorted(sinks)

def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs: List[tuple[str, int]] = []

    for start in sorted(starting_nodes):
        for end in sorted(ending_nodes):
            if has_path(graph, start, end):
                for path in all_simple_paths(graph, start, end):
                    seq = path[0]
                    for node in path[1:]:
                        seq += node[-1]
                    contigs.append((seq, len(seq)))

    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with output_file.open("w") as filout:
        for i, item in enumerate(contigs_list):
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                seq, length = str(item[0]), int(item[1])
            else:
                seq = str(item)
                length = len(seq)

            filout.write(f">contig_{i} len={length}\n")
            filout.write(textwrap.fill(seq, width=80) + "\n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)
    dico = build_kmer_dict(args.fastq_file, 3)
    build_graph(dico)


if __name__ == "__main__":  # pragma: no cover
    main()
