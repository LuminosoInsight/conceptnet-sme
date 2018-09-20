"""
Utilities to evaluate different ConceptNet data sources by comparing the 
semantic energies (as defined by a trained SME model) they assign to edges.
"""

import numpy as np
import torch
import torch.nn as nn

import json
import msgpack
import os

from collections import defaultdict

from conceptnet5.uri import uri_prefix
from semantic_matching import SemanticMatchingModel, stopwatch, TimeAccumulator
from conceptnet5.util import get_data_filename

def read_edges(root):
    """
    Read all csv files (assumed to be ConceptNet edge files) under the given 
    root (e.g. $CONCEPTNET_DATA/edges or a single edge file) and generate one 
    five-tuple for every edge they contain, consisting of the relation, the 
    (uri-prefixes of the) left and right endpoints, the dataset, and the data 
    sources (as a string).
    """
    if not os.path.isdir(root):
        files = [root]
    else:
        files = []
        for root_dir, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith('.csv'):
                    path = os.path.join(root_dir, filename)
                    files.append(path)
    for path in files:
        with open(path, 'rt', encoding='utf-8') as fp:
            for line in fp:
                _, rel, left, right, json_data = line.split('\t')
                left = uri_prefix(left)
                right = uri_prefix(right)
                data = json.loads(json_data, encoding='utf-8')
                dataset = data['dataset']
                source = data['sources']
                yield (rel, left, right, dataset, source)


def evaluate_sources(model, root, **kwargs):
    """
    Read all csv files (assumed to be ConceptNet edge files) under the given 
    root (typically $CONCEPTNET_DATA/edges or a single edge csv file) and 
    return two dicts, one mapping each dataset found to a dict mapping edges 
    from that dataset to their scores as returned by the model's score_edges 
    method, and the other similarly mapping each source found to edge scores.  
    Other (optional) keyword arguments will be passed to the model's 
    score_edges method.
    """
    scores_by_dataset = defaultdict(dict)
    scores_by_source = defaultdict(dict)
    for score, (rel, left, right, dataset, sources) in model.score_edges(
            read_edges(root),
            **kwargs
    ):
        edge = (rel, left, right)
        scores_by_dataset[dataset][edge] = score
        for source_dict in sources:
            for source in source_dict.values():
                scores_by_source[source][edge] = score
    return scores_by_dataset, scores_by_source

def save_scores(scores, path):
    """
    Serialize scores as returned by evaluate_sources to a msgpack file at 
    the given path.  
    """
    canonical_scores = {source: {edge:float(score)
                                 for edge, score in source_scores.items()}
                        for source, source_scores in scores.items()}
    with open(path, 'wb') as fp:
        msgpack.pack(canonical_scores, fp)

def load_scores(path):
    """
    Deserialize scores serialized by save_scores to the given path.
    """
    with open(path, 'rb') as fp:
        canonical_scores = msgpack.unpack(fp, use_list=False, encoding='utf-8')
    scores = {source: {tuple(edge):np.float32(score)
                       for edge, score in source_scores.items()}
              for source, source_scores in canonical_scores.items()}
    return scores

def composite_scores(raw_scores):
    """
    Convert a dict of per-data-source score dicts (as returned by evaluate_sources) 
    into two single scores per data source: (1) the mean score assigned to edges 
    from that data source, and (2) the ratio of the total of the scores assigned 
    to edges from that source to the total over all edges from any source (which 
    in the case of probability rather than logit scaled scores correspond roughly 
    to precision and (approximately) recall for each source, assuming the model 
    assigns fairly accurate estimates of the probability of truth to each edge).
    """
    result = {}
    pooled_edge_scores = {}
    for source, source_scores in raw_scores.items():
        source_mean = np.mean(list(source_scores.values()))
        source_sum = source_mean * len(list(source_scores.values()))
        result[source] = (source_mean, source_sum)
        pooled_edge_scores.update(source_scores)
    pooled_sum = np.sum(list(pooled_edge_scores.values()))
    result = {source: (source_mean, source_sum/pooled_sum)
              for source, (source_mean, source_sum) in result.items()}
    return result

def save_composite_scores(scores, path):
    """
    Serialize composite scores as returned by composite_scores to a msgpack file 
    atthe given path.
    """
    canonical_scores = {source: (float(s0), float(s1)) for source, (s0, s1) in scores.items()}
    with open(path, 'wb') as fp:
        msgpack.pack(canonical_scores, fp)

def load_composite_scores(path):
    """
    Deserialize composite scores serialized by save_composite_scores to the 
    given path.
    """
    with open(path, 'rb') as fp:
        canonical_scores = msgpack.unpack(fp, use_list=False, encoding='utf-8')
    scores = {source: (np.float32(s0), np.float32(s1))
              for source, (s0, s1) in canonical_scores.items()}
    return scores


if __name__ == '__main__':
    accumulator = TimeAccumulator()
    edges_filename = get_data_filename('collated/sorted/edges-shuf.csv')
    vectors_filename = get_data_filename('vectors/numberbatch-biased.h5')
    model_filename = get_data_filename('sme/sme.model')
    scores_by_dataset_filename = get_data_filename('sme/scores_by_dataset.msgpack')
    scores_by_source_filename = get_data_filename('sme/scores_by_source.msgpack')
    dataset_scores_filename = get_data_filename('sme/dataset_composite_scores.msgpack')
    source_scores_filename = get_data_filename('sme/source_composite_scores.msgpack')
    model = SemanticMatchingModel.load_model(model_filename)
    print('Scoring edges....')
    with stopwatch(accumulator):
        scores_by_dataset, scores_by_source = evaluate_sources(
            model, edges_filename, convert_logits_to_probas=True
        )
    accumulator.print('Time to collect scores:', accumulated_time=0.0)
    save_scores(scores_by_dataset, scores_by_dataset_filename)
    save_scores(scores_by_source, scores_by_source_filename)
    dataset_scores = composite_scores(scores_by_dataset)
    source_scores = composite_scores(scores_by_source)
    save_composite_scores(dataset_scores, dataset_scores_filename)
    save_composite_scores(source_scores, source_scores_filename)
