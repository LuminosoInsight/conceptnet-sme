import torch
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from conceptnet5.vectors.formats import save_hdf, save_npy
import numpy as np
import pandas as pd
import copy
import random
import re
import pathlib
import os
import contextlib
import time

from conceptnet_sme.relations import (
    COMMON_RELATIONS, ALL_RELATIONS, SYMMETRIC_RELATIONS, ENTAILED_RELATIONS,
)
from conceptnet5.uri import uri_prefix, assertion_uri, get_uri_language
from conceptnet5.util import get_data_filename
from conceptnet5.vectors.formats import load_hdf
from conceptnet5.vectors.transforms import l2_normalize_rows


RELATION_INDEX = pd.Index(COMMON_RELATIONS)
N_RELS = len(RELATION_INDEX)
INITIAL_VECS_FILENAME = get_data_filename('vectors/numberbatch-biased.h5')
EDGES_FILENAME = get_data_filename('collated/sorted/edges-shuf.csv')
MODEL_FILENAME = get_data_filename('sme/sme.model')
NEG_SAMPLES = 5
LANGUAGES_TO_USE = [
    'en', 'fr', 'de', 'it', 'es', 'ru', 'pt', 'ja', 'zh', 'nl',
    'ar', 'fa', 'ko', 'ms', 'no', 'pl', 'sv', 'mul'
]
BATCH_SIZE=128

random.seed(0)


def coin_flip():
    return random.choice([False, True])


def _make_rel_chart():
    """
    When we produce positive and negative examples from ConceptNet edges, we produce
    some examples that involve changing the relation of the edge.

    For positive examples, we can replace a relation with a relation that it entails.
    For negative examples, we can replace a relation with one it doesn't entail.

    We store the index numbers of possible replacement relations in `entailed_map`
    and `unrelated_map`.
    """
    entailed_map = {}
    unrelated_map = {}
    for rel in ALL_RELATIONS:
        entailed = [rel]
        entailed_rel = rel
        while entailed_rel in ENTAILED_RELATIONS:
            entailed_rel = ENTAILED_RELATIONS[entailed_rel]
            entailed.append(entailed_rel)
        entailed_map[rel] = [
            i for (i, rel) in enumerate(COMMON_RELATIONS)
            if rel in entailed
        ]
        unrelated_map[rel] = [
            i for (i, rel) in enumerate(COMMON_RELATIONS)
            if rel not in entailed
        ]
    return entailed_map, unrelated_map


ENTAILED_INDICES, UNRELATED_INDICES = _make_rel_chart()


@contextlib.contextmanager
def stopwatch(consumer):
    '''
    After executing the managed block of code, call the provided consumer with 
    two arguments, the start and end times of the execution of the block.
    '''
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    if consumer is not None:
        consumer(start_time, end_time)
    return

class TimeAccumulator:
    '''
    A simple consumer for use with stopwatches, that accumulates the total 
    elapsed time over multiple calls, and has a convenience method for printing 
    the total time (and optionally resetting it).
    '''
    def __init__(self, initial_time=0.0):
        self.accumulated_time = initial_time
        return
    def __call__(self, start_time, end_time):
        self.accumulated_time += end_time - start_time
        return
    def print(self, caption, accumulated_time=None):
        print('{} {} sec.'.format(caption, self.accumulated_time))
        if accumulated_time is not None:
            self.accumulated_time = accumulated_time
        return


class EdgeDataset(Dataset):
    '''
    Wrapper class (around an iteration over ConceptNet edges) that enables us 
    to use a torch DataLoader to parallelize generation of training batches.
    '''
    def __init__(self, filename, index):
        '''
        Construct an edge dataset from a filename (of a tab-separated 
        ConceptNet edge file, which should be in random order), and a (pandas) 
        index mapping (row) numbers to terms of the ConceptNet vocabulary. 
        '''
        super().__init__()
        self.filename = filename
        self.index = index
        self.n_terms = len(self.index)
        # Cache the edges in CPU memory as torch LongTensors,
        # skipping edges we don't intend to process.
        rel_indices = []
        left_indices = []
        right_indices = []
        weights = []
        for count, (rel, left, right, weight) in \
          enumerate(self.iter_edges_once()):
            if count % 1000000 == 0:
                print('Read {} edges.'.format(count))
            if rel not in COMMON_RELATIONS:
                continue
            if not ENTAILED_INDICES[rel]:
                continue
            rel_idx = RELATION_INDEX.get_loc(rel)
            left = uri_prefix(left)
            right = uri_prefix(right)
            try:
                left_idx = self.index.get_loc(left)
                right_idx = self.index.get_loc(right)
            except KeyError:
                continue
            rel_indices.append(rel_idx)
            left_indices.append(left_idx)
            right_indices.append(right_idx)
            weights.append(weight)
        if len(rel_indices) < 1:
            print('No edges survived filtering; fitting is impossible!')
            raise ValueError
        self.rel_indices = torch.LongTensor(rel_indices)
        self.left_indices = torch.LongTensor(left_indices)
        self.right_indices = torch.LongTensor(right_indices)
        self.edge_weights = torch.FloatTensor(weights)
        return

    def __len__(self):
        return len(self.rel_indices)
    
    def iter_edges_once(self):
        '''Return the edges from the data file.'''
        with open(self.filename, encoding='utf-8') as fp:
            for line in fp:
                _assertion, relation, concept1, concept2, _rest = \
                    line.split('\t', 4)
                yield (relation, concept1, concept2, 1.)
        return

    def __getitem__(self, i_edge):
        '''
        Produce a positive example and weight and a batch of (of size 
        NEG_SAMPLES) of negative examples, derived from the edge at the 
        given index.  The return values are three torch tensors containing 
        the positive example, the negative examples, and the weights.
        '''
        rel_idx = self.rel_indices[i_edge]
        left_idx = self.left_indices[i_edge]
        right_idx = self.right_indices[i_edge]
        weight = self.edge_weights[i_edge]
        
        rel = RELATION_INDEX[rel_idx]
        
        # Possibly swap the sides of a relation
        if coin_flip() and rel in SYMMETRIC_RELATIONS:
            left_idx, right_idx = right_idx, left_idx
        
        # Possibly replace a relation with a more general relation
        if coin_flip():
            rel_idx = random.choice(ENTAILED_INDICES[rel])
            rel = COMMON_RELATIONS[rel_idx]
        
        pos_rels = [rel_idx]
        pos_left = [left_idx]
        pos_right= [right_idx]
        weights = [weight]
        
        neg_rels = []
        neg_left = []
        neg_right = []

        for iter in range(NEG_SAMPLES):
            corrupt_rel_idx = rel_idx
            corrupt_left_idx = left_idx
            corrupt_right_idx = right_idx
            
            corrupt_which = random.randrange(5)
            if corrupt_which == 0:
                if rel not in SYMMETRIC_RELATIONS and coin_flip():
                    corrupt_left_idx = right_idx
                    corrupt_right_idx = left_idx
                else:
                    corrupt_rel_idx = random.choice(UNRELATED_INDICES[rel])
            elif corrupt_which == 1 or corrupt_which == 2:
                while corrupt_left_idx == left_idx:
                    corrupt_left_idx = random.randrange(self.n_terms)
            else:
                while corrupt_right_idx == right_idx:
                    corrupt_right_idx = random.randrange(self.n_terms)

            neg_rels.append(corrupt_rel_idx)
            neg_left.append(corrupt_left_idx)
            neg_right.append(corrupt_right_idx)

        pos_data = dict(rel=torch.LongTensor(pos_rels),
                        left=torch.LongTensor(pos_left),
                        right=torch.LongTensor(pos_right))
        neg_data = dict(rel=torch.LongTensor(neg_rels),
                        left=torch.LongTensor(neg_left),
                        right=torch.LongTensor(neg_right))
        weights = torch.FloatTensor(weights)
        return dict(positive_data=pos_data,
                    negative_data=neg_data,
                    weights=weights)

    def collate_batch(self, batch):
        '''
        Collates batches (as returned by a DataLoader that batches 
        the outputs of calls to __getitem__) into tensors (as required 
        by the train method of SemanticMatchingModel).
        '''
        pos_rels = torch.cat(list(x['positive_data']['rel'] for x in batch))
        pos_left = torch.cat(list(x['positive_data']['left'] for x in batch))
        pos_right = torch.cat(list(x['positive_data']['right'] for x in batch))
        neg_rels = torch.cat(list(x['negative_data']['rel'] for x in batch))
        neg_left = torch.cat(list(x['negative_data']['left'] for x in batch))
        neg_right = torch.cat(list(x['negative_data']['right'] for x in batch))
        weights = torch.cat(list(x['weights'] for x in batch))
        pos_data = (pos_rels, pos_left, pos_right)
        neg_data = (neg_rels, neg_left, neg_right)
        return pos_data, neg_data, weights


class CyclingSampler(Sampler):
    '''
    Like a sequential sampler, but these samplers cycle repeatedly over the 
    data source, and so have infinite length.
    '''
    def __init__(self, data_source):
        self.data_source = data_source
        self.index = 0
        return
    def _next_index(self):
        result = self.index
        self.index = (self.index + 1) % len(self.data_source)
        return result
    def __iter__(self):
        return iter(self._next_index, None) # never raises StopIteration!
    def __len__(self):
        return -1 # what else can we do?



class SemanticMatchingModel(nn.Module):
    """
    The PyTorch model for semantic matching energy over ConceptNet.
    """
    def __init__(self, index, use_cuda=True, term_dim=300, relation_dim=10,
                 batch_size=BATCH_SIZE):
        """
        Parameters:

        `index`: a pandas Index of words corresponding to the rows of the 
        term embedding to be learned.
        
        `term_dim`: the number of dimensions in the term embedding.

        `use_cuda`: whether to use GPU-accelerated PyTorch objects.

        `relation_dim`: the number of dimensions in the relation embeddings.
        Unlike SME as published, this can differ from the dimensionality of
        the term embeddings.

        `batch_size`: how many positive examples to use in each batch.
        The number of negative examples is NEG_SAMPLES times batch_size.
        """
        super().__init__()
        self.batch_size = batch_size

        # Initialize term embeddings, including the index that converts terms
        # from strings to row numbers
        self.index = index
        self.term_vecs = nn.Embedding(len(self.index), term_dim,
                                      sparse=True)

        # The assoc_tensor is a (k_2 x k_1 x k_1) tensor that represents
        # interactions between the relations and the terms. k_1 is the
        # dimensionality of the term embeddings, and k_2 is the dimensionality
        # of the relation embeddings.
        #
        # We pass the dimensions in the order (k_1, k_1, k_2) because the
        # PyTorch convention is that inputs come before outputs, but the
        # resulting tensor is (k_2 x k_1 x k_1) because the math convention
        # is that outputs are on the left.
        self.assoc_tensor = nn.Bilinear(
            term_dim, term_dim, relation_dim, bias=True
        )
        self.rel_vecs = nn.Embedding(N_RELS, relation_dim, sparse=True)

        # Using CUDA to run on the GPU requires different data types
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.register_buffer('identity_slice', 
                             torch.tensor(np.eye(term_dim),
                                          dtype=torch.float32,
                                          device=self.device))
        self.reset_synonym_relation()

        # Learnable priors for how confident to be in arbitrary statements.
        # These are used to convert the tensor products linearly into logits.
        # The initial values of these are set to numbers that appeared
        # reasonable based on a previous run.
        self.truth_multiplier = nn.Parameter(torch.tensor(
            5.0, dtype=torch.float32, device=self.device))
        self.truth_offset = nn.Parameter(torch.tensor(
            -3.0, dtype=torch.float32, device=self.device))

        self.to(self.device) # make sure everything is on the right device

    @classmethod
    def from_frame(cls, frame, use_cuda=True, relation_dim=10,
                   batch_size=BATCH_SIZE):
        """
        Parameters:

        `frame`: a pandas DataFrame of pre-trained word embeddings over the
        vocabulary of ConceptNet. `conceptnet5.vectors.formats.load_hdf`
        can load these.

        `use_cuda`: whether to use GPU-accelerated PyTorch objects.

        `relation_dim`: the number of dimensions in the relation embeddings.
        Unlike SME as published, this can differ from the dimensionality of
        the term embeddings.

        `batch_size`: how many positive examples to use in each batch.
        The number of negative examples is NEG_SAMPLES times batch_size.
        """
        model = cls(frame.index, term_dim=frame.values.shape[1],
                    use_cuda=use_cuda, relation_dim=relation_dim,
                    batch_size=batch_size)
        model.term_vecs.weight.data.copy_(
            torch.from_numpy(frame.values)
        )
        return model
        

    def reset_synonym_relation(self):
        """
        The first layer of the assoc tensor is fixed to be the identity matrix.
        The embedding of the "Synonym" relation is fixed to use only this
        layer.

        In effect, this is saying that when term vectors are compared directly,
        instead of via a relation matrix, that should represent the "Synonym"
        relation.
        """
        self.assoc_tensor.weight.data[0] = self.identity_slice
        self.rel_vecs.weight.data[0, :] = 0
        self.rel_vecs.weight.data[0, 0] = 1

    def forward(self, rels, terms_L, terms_R):
        """
        The forward step of this model takes in a batch of relations and the
        corresponding terms on the left and right of them, and produces a
        batch of truth judgments.

        Truth judgments are on the logit scale. Positive numbers should
        represent true assertions, and higher values represent more confidence.

        A truth judgment is a logit, and can be converted to a probability
        using the sigmoid function.
        """
        # Get relation vectors for the whole batch, with shape (b, i)
        rels_b_i = self.rel_vecs(rels)
        # Get left term vectors for the whole batch, with shape (b, L)
        terms_b_L = self.term_vecs(terms_L)
        # Get right term vectors for the whole batch, with shape (b, R)
        terms_b_R = self.term_vecs(terms_R)

        # Get the interaction of the terms in relation-embedding space, with
        # shape (b, i).
        inter_b_i = self.assoc_tensor(terms_b_L, terms_b_R)

        # Multiply our (b * i) term elementwise by rels_b_i. This indicates
        # how well the interaction between term L and term R matches each
        # component of the relation vector.
        relmatch_b_i = inter_b_i * rels_b_i

        # Add up the components for each item in the batch
        energy_b = torch.sum(relmatch_b_i, 1)

        return energy_b * self.truth_multiplier + self.truth_offset

    def show_debug(self, batch, energy, positive):
        """
        On certain iterations, we show the training examples and what the model
        believed about them.
        """
        truth_values = energy
        rel_indices, left_indices, right_indices = batch
        if positive:
            print("POSITIVE")
        else:
            print("\nNEGATIVE")
        for i in range(len(energy)):
            rel = RELATION_INDEX[int(rel_indices.data[i])]
            left = self.index[int(left_indices.data[i])]
            right = self.index[int(right_indices.data[i])]
            value = truth_values.data[i]
            print("[%4.4f] %s %s %s" % (value, rel, left, right))

    @staticmethod
    def load_initial_frame():
        """
        Load the pre-computed embeddings that form our starting point.
        """
        accumulator = TimeAccumulator()
        with stopwatch(accumulator):
            print('Loading frame from file {}.'.format(INITIAL_VECS_FILENAME))
            frame = load_hdf(INITIAL_VECS_FILENAME)
            labels = [
                label for label in frame.index
                if get_uri_language(label) in LANGUAGES_TO_USE
            ]
            frame = frame.loc[labels]
        accumulator.print('Loading frame took')
        return frame.astype(np.float32)

    @staticmethod
    def load_model(filename):
        """
        Load the SME model from a file.
        """
        accumulator = TimeAccumulator()
        with stopwatch(accumulator):
            print('Loading model from file {}.'.format(filename))
            frame = SemanticMatchingModel.load_initial_frame()
            model = SemanticMatchingModel(frame.index)
            # We have adopted the convention that models are moved to the
            # cpu before saving their state (as else loading would need
            # twice the gpu memory).  We (currently) do not support building
            # a model on the gpu and loading it on the cpu (or conversely).
            model.cpu()
            model.load_state_dict(torch.load(filename))
            model.to(model.device)
        accumulator.print('Total time to load model:')
        return model

    def evaluate_conceptnet(self, dataset, cutoff_value=-1,
                            output_filename=None):
        """
        Use the SME model to "sanity-check" existing edges in ConceptNet.
        We particularly highlight edges that get a value of -1 or less, which
        may be bad or unhelpful edges.
        """
        self.eval() # in case someday we add submodules where the mode matters
            
        if output_filename:
            out = open(output_filename, 'w', encoding='utf-8')
        else:
            out = None
        for rel, left, right, weight in dataset.iter_edges_once():
            try:
                rel_idx = RELATION_INDEX.get_loc(rel)
                left_idx = self.index.get_loc(left)
                right_idx = self.index.get_loc(right)
            except KeyError:
                continue

            rel_tensor = torch.tensor([rel_idx], dtype=torch.int64,
                                      device=self.device)
            left_tensor = torch.tensor([left_idx], dtype=torch.int64,
                                       device=self.device)
            right_tensor = torch.tensor([right_idx], dtype=torch.int64,
                                        device=self.device)

            model_output = self(rel_tensor, left_tensor, right_tensor)
 
            value = model_output.item()
            assertion = assertion_uri(rel, left, right)
            if value < cutoff_value:
                print("%4.4f\t%s" % (value, assertion))
            if out is not None:
                print("%4.4f\t%s" % (value, assertion), file=out)

    def export(self, dirname):
        """
        Convert the model into HDF5 / NumPy files that can be loaded from
        other code.
        """
        # terms-similar.h5 contains the term embeddings. (The name indicates
        # that comparing these embeddings directly gets you similarity or
        # synonymy.)
        path = pathlib.Path(dirname)
        term_mat = self.term_vecs.weight.data.float().cpu().numpy()
        term_frame = pd.DataFrame(term_mat, index=self.index)
        save_hdf(term_frame, str(path / "terms-similar.h5"))

        # relations.h5 contains the relation embeddings.
        rel_mat = self.rel_vecs.weight.data.float().cpu().numpy()
        rel_frame = pd.DataFrame(rel_mat, index=RELATION_INDEX)
        save_hdf(rel_frame, str(path / "relations.h5"))

        # assoc.npy contains the tensor of interactions.
        assoc_t = self.assoc_tensor.weight.data.float().cpu().numpy()
        save_npy(assoc_t, str(path / "assoc.npy"))

        # terms-related.h5 is something to experiment with; it contains the
        # term embeddings acted upon by the operator corresponding to
        # /r/RelatedTo. Our goal is to distinguish similarity from relatedness,
        # such as for the SimLex evaluation.
        rel_vec = rel_mat[1]
        related_mat = np.einsum('i,ijk->jk', rel_vec, assoc_t)
        related_mat = (related_mat + related_mat.T) / 2
        related_terms = term_frame.dot(related_mat)
        save_hdf(related_terms, str(path / "terms-related.h5"))



def clip_grad_norm(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters, just like 
    nn.utils.clip_grad_norm_, but works even if some parameters are sparse.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor]): an iterable of Tensors that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        def L_inf_norm(tensor):
            if tensor.layout == torch.sparse_coo:
                if tensor._nnz() > 0:
                    return tensor.coalesce()._values().abs().max()
                else:
                    return 0.0
            else:
                return tensor.abs().max()
        total_norm = max(L_inf_norm(p.grad.data) for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if p.grad.data.layout == torch.sparse_coo:
                param_norm = p.grad.data.coalesce()._values().norm(norm_type)
            else:
                param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = float(max_norm / (total_norm + 1e-6))
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm



class SGD_Sparse(optim.SGD):
    """
    Just like optim.SGD except that this class handles non-zero 
    weight decay and momentum even when some of the parameters 
    being optimized have sparse gradient storage.
    
    NOTE:  Currently there seem to be gpu memory leaks when using 
    Nesterov momentum; use that at your own risk.
    """
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p_is_sparse = (d_p.layout == torch.sparse_coo)
                if weight_decay != 0:
                    if not d_p_is_sparse:
                        d_p.add_(weight_decay, p.data)
                    else:
                        # Update the gradient via weight decay only at
                        # locations where it is already non-zero; this
                        # isn't quite the same as actual weight decay but
                        # it avoids ballooning the memory consumption.
                        d_p = d_p.coalesce()
                        p.grad.data = d_p
                        if len(d_p._indices().size()) < 2: # empty sparse tensor
                            pass
                        else:
                            # Find the indices of the nonzero entries in the
                            # gradient, assuming row-major storage order, by
                            # finding the index of each nonzero subtensor's
                            # first entry and adding offsets to the other
                            # entries.
                            assert p.data.size() == d_p.size()
                            n_sparse_dims, n_values = d_p._indices().size()
                            values_shape = tuple(d_p._values().size())
                            assert len(values_shape) > 0
                            assert n_values == values_shape[0]
                            value_size = int(np.prod(np.array(
                                values_shape[1:])))
                            factors = np.array(d_p.size())
                            factors = np.flip(factors, axis=0)
                            factors = np.append(np.array([1]), factors[:-1])
                            factors = np.cumprod(factors)
                            factors = np.flip(factors, axis=0)
                            factors = factors.tolist() # lose negative strides
                            assert n_sparse_dims <= len(factors)
                            flat_indices0 = torch.zeros(
                                n_values, dtype=torch.int64, device=d_p.device)
                            for i_sparse_dim in range(n_sparse_dims):
                                flat_indices0.add_(
                                    factors[i_sparse_dim],
                                    d_p._indices()[i_sparse_dim])
                            flat_indices0.unsqueeze_(1)
                            offsets = torch.arange(value_size, 
                                dtype=torch.int64, device=d_p.device).\
                                unsqueeze(0)
                            flat_indices = \
                                torch.index_select(
                                    flat_indices0, 1,
                                    torch.zeros(
                                        value_size, 
                                        dtype=torch.int64,
                                        device=d_p.device)) \
                                + \
                                torch.index_select(
                                    offsets, 0,
                                    torch.zeros(
                                        n_values,
                                        dtype=torch.int64,
                                        device=d_p.device))
                            flat_indices.resize_(values_shape)
                            del flat_indices0, offsets
                            ctor = eval(d_p.type())
                            vals = torch.take(p.data, flat_indices)
                            try: 
                                sparse_data = ctor(
                                    d_p._indices(), vals, d_p.size())
                            except RuntimeError:
                                # This looks like a bug in pytorch; sparse
                                # tensors evidently cannot be created on
                                # every gpu, even if all the data is on the
                                # same gpu.  The workaround:  construct the 
                                # sparse tensor on the cpu then move it.
                                ctor_cpu = eval(
                                    re.sub('\.cuda', '', d_p.type()))
                                sparse_data = ctor_cpu(
                                    d_p._indices().cpu(), vals.cpu(),
                                    d_p.size()).to(d_p.device)
                            d_p.add_(weight_decay, sparse_data)
                            del flat_indices, sparse_data, vals
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(d_p)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if d_p_is_sparse:
                        buf = buf.coalesce()
                        param_state['momentum_buffer'] = buf
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                if d_p_is_sparse:
                    d_p = d_p.coalesce()
                    p.grad.data = d_p

                p.data.add_(-group['lr'], d_p)

        return loss


class DataParallelizedModule(nn.Module):
    """
    Similarly to nn.DataParallel, this class of modules serves to wrap 
    other modules and run them in parallel over multiple gpus, splitting 
    training (or testing/application) batches between the gpus (over the 
    first dimension of the batch).  
    
    Unlike nn.DataParallel, all of the wrapped module's parameters will be 
    copied to all gpus during construction of the wrapper, and only gradient 
    data will be copied from gpu to gpu during training (no data should need 
    to be copied between gpus during forward application).  However, training 
    should only use loss functions that can be expressed as averages over all 
    data points of a batch of some per-data-point loss.  Also, the set of 
    parameters of the wrapped module is assumed not to change (of course their 
    values can change) over the lifetime of the wrapper object.
    
    Note that during training of a wrapped module, it is necessary to call 
    the wrapper's broadcast_gradients method immediately following the 
    backpropagation of gradients (i.e. typically right after calling 
    .backward on some loss tensor), in order to share corrections to the 
    computed gradients between the gpus.
    """
    def __init__(self, module, device, copy_fn=copy.deepcopy):
        """
        Construct a parallelizing wrapper for the given module (an instance 
        of nn.Module).  The module will be moved to the given device (if 
        not already there) and copies will be made using the given copy 
        function (defaulting to copy.deepcopy) and placed on all other 
        available gpus.
        """
        super().__init__()
        device_ids = list(range(torch.cuda.device_count()))
        devices = [torch.device('cuda:{}'.format(d_id)) for d_id in device_ids]
        if device not in devices:
            device = devices[0]
        self.children = [module]
        self.devices = [device]
        self.add_module('child:{}'.format(device.index), module)
        for dev in devices:
            if dev != device:
                self.devices.append(dev)
        for dev in self.devices[1:]:
            module.cpu()
            module_copy = copy_fn(module)
            module_copy.to(dev)
            self.children.append(module_copy)
            self.add_module('child:{}'.format(dev.index), module_copy)
        module.to(device)
        self.chunk_sizes = torch.zeros(len(self.children), dtype=torch.int64)

    def forward(self, *args):
        """
        Scatter the supplied args (assumed to be a list of tensors) across 
        the child modules, and gather their outputs (assumed to be single 
        tensors) back to the first gpu.  Also, accumulate the sizes of the 
        scattered chunks (for later use in updating parameter gradients).
        """
        if len(self.children) <= 1:
            return self.children[0](*args)
        device_ids = [device.index for device in self.devices]
        chunk_lists = list(list() for i_device in device_ids)
        for arg in args:
            chunks = torch.cuda.comm.scatter(arg, device_ids)
            for i_child, chunk in enumerate(chunks):
                chunk_lists[i_child].append(chunk)
        chunks = list(tuple(chunk_list) for chunk_list in chunk_lists)
        outputs = []
        for i_child, (module, chunk) in enumerate(zip(self.children, chunks)):
            chunk_size = chunk[0].size()[0]
            self.chunk_sizes[i_child] += chunk_size
            output = module(*chunk)
            outputs.append(output)
        assert len(self.children) == len(outputs)
        output = torch.cuda.comm.gather(
            outputs, destination=self.devices[0].index)
        return output

    def zero_grad(self):
        """
        In addition to the ordinary zeroing of gradient data, reset the 
        chunk size data.
        """
        super().zero_grad()
        self.chunk_sizes = torch.zeros(len(self.children), dtype=torch.int64)

    def broadcast_gradients(self):
        """
        Compute a single value, for all the child modules, of the gradient 
        of each module parameter (as a convex combination of the gradients 
        in the individual children, with coefficients based on the batch 
        sizes from the forward computation), and distribute these common 
        gradients back to all the children.
        """
        if len(self.children) <= 1:
            return
        weights = self.chunk_sizes.to(torch.float32) / \
                  self.chunk_sizes.sum().to(torch.float32)
        weights = [weights[ii].item() for ii, dev in
                   enumerate(self.devices)]
        for name, param in self.children[0].named_parameters():
            if param.grad is None:
                continue
            param_copies = [param]
            for other_module in self.children[1:]:
                other_module_params = list(
                    p for other_name, p in other_module.named_parameters()
                    if other_name == name)
                assert len(other_module_params) == 1
                param_copies.append(other_module_params[0])
            param_grad = torch.cuda.comm.reduce_add(
                list(param_copy.grad.mul_(weight) for param_copy, weight in
                     zip(param_copies, weights)), 
                destination=self.devices[0].index)
            for ii, param_copy in enumerate(param_copies):
                param_copy.grad = param_grad.to(self.devices[ii])



def train_model(model, dataset):
    """
    Incrementally train the model on the given dataset.

    As it is, this function will never return. It writes the results so far
    to `sme/sme.model` every 5000 iterations, and you can run it for as
    long as you want.
    """
    model.train()
    
    def model_copier(model):
        model.cpu()
        new_model = SemanticMatchingModel(model.index, use_cuda=False,
                                          batch_size=model.batch_size)
        new_model.load_state_dict(model.state_dict())
        return new_model

    if torch.cuda.is_available()and torch.cuda.device_count() > 1:
        parallel_model = DataParallelizedModule(
            model, model.device, copy_fn=model_copier)
    else:
        parallel_model = model
        
    # Relative loss says that the positive examples should outrank their
    # corresponding negative examples, with a difference of at least 1
    # logit between them. If the difference is less than this (especially
    # if it's negative), this adds to the relative loss.
    relative_loss_function = nn.MarginRankingLoss(margin=1)

    # Absolute loss measures the cross-entropy of the predictions:
    # true statements should get positive values, false statements should
    # get negative values, and the sigmoid of those values should be a
    # probability that accurately reflects the model's confidence.
    absolute_loss_function = nn.BCEWithLogitsLoss()

    optimizer = SGD_Sparse(parallel_model.parameters(), lr=0.1,
                           weight_decay=1e-9)
    losses = []
    true_target = torch.ones([model.batch_size], dtype=torch.float32,
                             device=model.device)
    false_target = torch.zeros([model.batch_size], dtype=torch.float32,
                               device=model.device)
    steps = 0

    # Note that you want drop_last=False with a CyclingSampler.
    data_loader = DataLoader(dataset, batch_size=model.batch_size,
                             drop_last=False, num_workers=10,
                             collate_fn=dataset.collate_batch,
                             sampler=CyclingSampler(dataset),
                             pin_memory=True)
    for pos_batch, neg_batch, weights in data_loader:
        if model.device != torch.device('cpu'):
            pos_batch = tuple(
                x.cuda(device=model.device, non_blocking=True)
                for x in pos_batch)
            neg_batch = tuple(
                x.cuda(device=model.device, non_blocking=True)
                for x in neg_batch)
            weights = weights.cuda(device=model.device, non_blocking=True)
        parallel_model.zero_grad()
        pos_energy = parallel_model(*pos_batch)
        neg_energy = parallel_model(*neg_batch)
        
        abs_loss = absolute_loss_function(pos_energy, true_target)
        rel_loss = 0
        for neg_index in range(NEG_SAMPLES):
            neg_energy_slice = neg_energy[neg_index::NEG_SAMPLES]
            rel_loss += relative_loss_function(pos_energy, neg_energy_slice, true_target)
            abs_loss += absolute_loss_function(neg_energy_slice, false_target)

        loss = abs_loss + rel_loss
        loss.backward()

        if model != parallel_model:
            parallel_model.broadcast_gradients()
            for model_copy in parallel_model.children:
                clip_grad_norm(model_copy.parameters(), 1)
        else:
            clip_grad_norm(model.parameters(), 1)
        
        optimizer.step()
        
        if model == parallel_model:
            model.reset_synonym_relation()
        else:
            for model_copy in parallel_model.children:
                model_copy.reset_synonym_relation()

        losses.append(loss.data.cpu().item())
        steps += 1

        if steps in (1, 10, 20, 50, 100) or steps % 100 == 0:
            model.show_debug(neg_batch, neg_energy, False)
            model.show_debug(pos_batch, pos_energy, True)
            avg_loss = np.mean(losses)
            print("%d steps, loss=%4.4f, abs=%4.4f, rel=%4.4f" % (
                steps, avg_loss, abs_loss, rel_loss
            ))
            losses.clear()

        if steps % 5000 == 0:
            model.cpu()
            torch.save(model.state_dict(), MODEL_FILENAME)
            model.to(model.device)
            print("saved")
            # Sanity check:  if the model has been parallelized across
            # multiple gpu's, the parameters across all gpu's should agree.
            if model != parallel_model:
                for child in parallel_model.children:
                    for name, param in model.named_parameters():
                        child_params = list(
                            p for other_name, p in child.named_parameters()
                            if name == other_name)
                        assert len(child_params) == 1
                        assert torch.norm((param.data.cpu() -
                                           child_params[0].data.cpu())) < 1e-6
        
    print()



def get_model():
    """
    Instantiate a model, either by loading it from the saved checkpoint, or
    by creating it from scratch if nothing is there.
    """
    if os.access(MODEL_FILENAME, os.F_OK):
        model = SemanticMatchingModel.load_model(MODEL_FILENAME)
    else:
        frame = SemanticMatchingModel.load_initial_frame()
        model = SemanticMatchingModel.from_frame(l2_normalize_rows(frame))
    return model


if __name__ == '__main__':
    model = get_model()
    print('Initializing edge dataset ....')
    dataset_accumulator = TimeAccumulator()
    with stopwatch(dataset_accumulator):
        dataset = EdgeDataset(EDGES_FILENAME, model.index)
    dataset_accumulator.print('Edge dataset initialization took')
    print('Edge dataset contains {} edges.'.format(len(dataset)))
    train_model(model, dataset)
    # model.evaluate_conceptnet(dataset)
