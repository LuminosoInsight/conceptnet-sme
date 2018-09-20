"""
Program to take a trained SME model and produce a list of assertions which 
that model gives high scores.
"""

import numpy as np
import torch
import torch.nn as nn
import sys

from collections import defaultdict

from semantic_matching import SemanticMatchingModel
from semantic_matching import RELATION_INDEX, SYMMETRIC_RELATIONS
from semantic_matching import _MessageWriter
from conceptnet5.util import get_data_filename
from conceptnet5.uri import get_uri_language, uri_prefix

MODEL_FILENAME = get_data_filename("sme/sme_ace_0.0865_20180829_restarted.model")
EDGE_FILENAME = get_data_filename("collated/sorted/edges-shuf.csv")
LOG_FILENAME = get_data_filename("sme/sme_ace_0.0865_20180829_restarted_eval_log.txt")

_message_writer = None

def log_message(msg):
    global _message_writer
    print(msg)
    if _message_writer is not None:
        _message_writer.write(msg)


@torch.no_grad()
def nearest_neighbors(model, term_vector, n_best=1):
    """
    Find the terms in the model's index whose embedding vectors are closest 
    to the given term vector.  The term "vector" must in fact be a torch 
    tensor.  The return value is a list containing the neighboring terms' 
    indices (in the model's index).
    """
    term_vectors = model.term_vecs.weight
    term_vectors.sub_(term_vector)
    norms = torch.norm(term_vectors, p=2, dim=1)
    term_vectors.add_(term_vector)
    
    if n_best > term_vectors.size()[0]:
        n_best = term_vectors.size()[0]  # can't find more neighbors than that
    big_norm = torch.max(norms) + 1.0
    
    neighbors = []
    while len(neighbors) < n_best:
        min_norm, i_neighbor = torch.min(norms, dim=0)
        norms[i_neighbor] = big_norm  # remove from further consideration
        neighbors.append(i_neighbor.item())

    return neighbors


@torch.no_grad()
def high_scoring_assertions(model, rel, n_top=2, n_singular_vectors_to_use=10,
                            n_neighbors_to_use=40, use_multiple_gpus=False):
    """
    Search for n_top (default 2) many assertions with the given relation which 
    will be given high scores by the model.  The search is done by looking for 
    pairs of terms in the model's index which are close to the pairs of vectors 
    maximizing the model score (which should usually give high scoring pairs of 
    terms, but with no guarantee that it will), and then taking the best n_top 
    of the assertions found that way.
    
    Returns a list of (score, (rel, left, right)) tuples.
    """
    scale_factor = model.truth_multiplier
    term_vectors = model.term_vecs.weight
    i_rel = torch.tensor(RELATION_INDEX.get_loc(rel), device=model.device)
    rel_vector = model.rel_vecs(i_rel)
    assoc_tensor = model.assoc_tensor.weight
    rel_assoc_product = scale_factor * torch.einsum('i,ijk->jk', (rel_vector, assoc_tensor))
    rel_assoc_product = rel_assoc_product.cpu().numpy()
    U, s, Vt = np.linalg.svd(rel_assoc_product, full_matrices=True, compute_uv=True)

    # Compute and print the minimum and maximum score values attainable by
    # edges with this relation.  Note that neither bound may be attained by
    # assertions whose endpoints are actual terms in the model index (but will
    # be attained by vectors in the embedding space that may not have such
    # corresponding terms).
    score_offset = model.truth_offset + \
                   model.truth_multiplier * torch.dot(rel_vector, model.assoc_tensor.bias)
    max_vector_norm = torch.max(torch.norm(term_vectors, 2, 1)).cpu().item()
    score_lower_bound = score_offset.cpu().item() - s[0] * max_vector_norm * max_vector_norm
    score_upper_bound = score_offset.cpu().item() + s[0] * max_vector_norm * max_vector_norm
    log_message('Scores for relation {} fall in [{}, {}].'.format(
        rel, score_lower_bound, score_upper_bound))

    # Iterate over the pairs of singular vectors of the bilinear form on the
    # term vectors' space induced by the relation vector and the model's
    # association tensor, and find the terms in the model's index which
    # have vectors nearest to the singular vectors.  Return edges built from
    # the relation and these terms for scoring.
    
    if n_singular_vectors_to_use > U.shape[1]:
        n_singular_vectors_to_use = U.shape[1]
        
    def edge_iterator():
        previous_edges = set()  # don't repeat edges
        for i_vector in range(n_singular_vectors_to_use):
            u = U[:,i_vector].reshape((U.shape[0],))
            v = Vt[i_vector,:].reshape((U.shape[0],))
            u_tensor = torch.tensor(u, device=model.device)
            v_tensor = torch.tensor(v, device=model.device)
            for sign in [1, -1]:
                u_tensor *= sign
                v_tensor *= sign
                u_neighbors = nearest_neighbors(model, u_tensor, n_best=n_neighbors_to_use)
                v_neighbors = nearest_neighbors(model, v_tensor, n_best=n_neighbors_to_use)
                for i_u_neighbor in u_neighbors:
                    u_neighbor_term = model.index[i_u_neighbor]
                    if get_uri_language(u_neighbor_term) != 'en':
                        continue
                    for i_v_neighbor in v_neighbors:
                        v_neighbor_term = model.index[i_v_neighbor]
                        if get_uri_language(v_neighbor_term) != 'en':
                            continue
                        if u_neighbor_term == v_neighbor_term:
                            continue
                        if (i_rel, i_u_neighbor, i_v_neighbor) in previous_edges:
                            continue
                        previous_edges.add((i_rel, i_u_neighbor, i_v_neighbor))
                        if rel in SYMMETRIC_RELATIONS:
                            previous_edges.add((i_rel, i_v_neighbor, i_u_neighbor))
                        yield i_rel, i_u_neighbor, i_v_neighbor, \
                            rel, u_neighbor_term, v_neighbor_term

    scored_edges = list(model.score_edges(
        edge_iterator(),
        edges_given_as_indices=True,
        use_multiple_gpus=use_multiple_gpus
    ))
    scored_edges = [(score, (rel, left, right))
                     for score, (i_rel, i_left, i_right, rel, left, right)
                     in scored_edges]
    permutation = list(np.argsort([score for score, edge in scored_edges]))
    if n_top > len(permutation):
        n_top = len(permutation)
    top_n_scored_edges = [scored_edges[i_edge] for i_edge in permutation[-n_top:]]
    return top_n_scored_edges


@torch.no_grad()
def high_scoring_assertions_for_common_terms(model, rel, edge_file, n_top=2):
    """
    Finds the most common left- and right-hand endpoint terms for the given 
    relation, and for each returns the assertions with the highest scores 
    that can be formed by filling in the other endpoint with terms from the 
    model's index.
    """
    # Find the most commonly occuring left and right-hand terms.
    left_counts = defaultdict(int)
    right_counts = defaultdict(int)
    edge_set = set()
    with open(edge_file, 'rt', encoding='utf-8') as fp:
        for line in fp:
            _assertion, rel1, left, right, _rest = line.split('\t')
            if rel1 != rel:
                continue
            left = uri_prefix(left)
            right = uri_prefix(right)
            if left not in model.index:
                continue
            if right not in model.index:
                continue
            if get_uri_language(left) != 'en':
                continue
            if get_uri_language(right) != 'en':
                continue
            left_counts[left] += 1
            right_counts[right] += 1
            edge_set.add((rel, left, right))
    lefts_with_counts = sorted(left_counts.items(), key=lambda p: -p[1])[:n_top]
    rights_with_counts = sorted(right_counts.items(), key=lambda p: -p[1])[:n_top]
    common_lefts = [left for left, count in lefts_with_counts]
    common_rights = [right for right, count in rights_with_counts]

    # For each common term, find the highest-scoring edges not in the graph.
    rel_idx = RELATION_INDEX.get_loc(rel)
    
    def left_iterator(left, left_idx):
        for right_idx, right in enumerate(model.index):
            if (rel, left, right) in edge_set:
                continue
            if get_uri_language(right) != 'en':
                continue
            if left_idx == right_idx:
                continue
            yield rel_idx, left_idx, right_idx, rel, left, right

    def right_iterator(right, right_idx):
        for left_idx, left in enumerate(model.index):
            if (rel, left, right) in edge_set:
                continue
            if get_uri_language(left) != 'en':
                continue
            if left_idx == right_idx:
                continue
            yield rel_idx, left_idx, right_idx, rel, left, right

    def find_best_edges(edge_iterator):
        best_scored_edges = [(-sys.float_info.max, None)] * n_top
        for score, (rel_idx, left_idx, right_idx, rel, left, right) in \
                model.score_edges(edge_iterator,
                                  edges_given_as_indices=True,
                                  batch_size=64):
            best_scored_edges.append((score, (rel, left, right)))
            # Sort in descending order of score by using -score as the key.
            permutation = list(np.argsort([-score for score, edge in best_scored_edges]))
            best_scored_edges = [best_scored_edges[i_edge] for i_edge in permutation[:-1]]
        return best_scored_edges

    best_edges_for_lefts = []
    for left in common_lefts:
        left_idx = model.index.get_loc(left)
        best_edges = find_best_edges(left_iterator(left, left_idx))
        best_edges_for_lefts.extend(best_edges)

    best_edges_for_rights = []
    for right in common_rights:
        right_idx = model.index.get_loc(right)
        best_edges = find_best_edges(right_iterator(right, right_idx))
        best_edges_for_rights.extend(best_edges)

    return best_edges_for_lefts, best_edges_for_rights


def main(model_filename, edge_filename):
    global _message_writer
    _message_writer = _MessageWriter(LOG_FILENAME)
    model = SemanticMatchingModel.load_model(model_filename)
    scored_assertions = set()
    for rel in RELATION_INDEX:
        new_assertions = high_scoring_assertions(model, rel)
        for new_assertion in new_assertions:
            scored_assertions.add(new_assertion)
        
    log_message('High scoring assertions:')
    scored_assertions = sorted(scored_assertions,
                               key=lambda pair: pair[0],
                               reverse=True)
    for score, (rel, left, right) in scored_assertions:
        log_message('{}\t{}\t{}\t{}'.format(rel, left, right, score))

    log_message('\nHigh scoring assertions for common terms:')
    for rel in RELATION_INDEX:
        best_for_common_lefts, best_for_common_rights = \
            high_scoring_assertions_for_common_terms(model, rel, edge_filename)
        log_message('Edges for relation {} with common left terms:'.format(rel))
        for score, (rel, left, right) in best_for_common_lefts:
            log_message('{}\t{}\t{}\t{}'.format(rel, left, right, score))
        log_message('Edges for relation {} with common right terms:'.format(rel))
        for score, (rel, left, right) in best_for_common_rights:
            log_message('{}\t{}\t{}\t{}.'.format(rel, left, right, score))


if __name__ == '__main__':
    main(MODEL_FILENAME, EDGE_FILENAME)
