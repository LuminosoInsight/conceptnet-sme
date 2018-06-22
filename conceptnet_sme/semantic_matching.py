import torch

# Set USE_MULTIPLE_GPUS = True to enable parallelization of training over
# multiple GPUs (if available; setting it True when there are no or only one
# GPU won't hurt, but it won't help either).

USE_MULTIPLE_GPUS = True

# Set NUM_BATCH_WORKERS > 0 to enable parallelization of generation of training
# batches over multiple CPU cores (by spawning that number of worker processes;
# 10 is a reasonable choice if you have multiple CPUs).

NUM_BATCH_WORKERS = 10
if NUM_BATCH_WORKERS > 0:
    try:
        torch.multiprocessing.set_start_method("spawn")
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
import pathlib
import os
import contextlib
import time

from conceptnet_sme.relations import (
    COMMON_RELATIONS,
    ALL_RELATIONS,
    SYMMETRIC_RELATIONS,
    ENTAILED_RELATIONS,
)
from conceptnet5.uri import uri_prefix, assertion_uri, get_uri_language
from conceptnet5.util import get_data_filename
from conceptnet5.vectors.formats import load_hdf
from conceptnet5.vectors.transforms import l2_normalize_rows


RELATION_INDEX = pd.Index(COMMON_RELATIONS)
N_RELS = len(RELATION_INDEX)
INITIAL_VECS_FILENAME = get_data_filename("vectors/numberbatch-biased.h5")
EDGES_FILENAME = get_data_filename("collated/sorted/edges-shuf.csv")
MODEL_FILENAME = get_data_filename("sme/sme.model")
NEG_SAMPLES = 5
LANGUAGES_TO_USE = [
    "en",
    "fr",
    "de",
    "it",
    "es",
    "ru",
    "pt",
    "ja",
    "zh",
    "nl",
    "ar",
    "fa",
    "ko",
    "ms",
    "no",
    "pl",
    "sv",
    "mul",
]
BATCH_SIZE = 160

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
            i for (i, rel) in enumerate(COMMON_RELATIONS) if rel in entailed
        ]
        unrelated_map[rel] = [
            i for (i, rel) in enumerate(COMMON_RELATIONS) if rel not in entailed
        ]
    return entailed_map, unrelated_map


ENTAILED_INDICES, UNRELATED_INDICES = _make_rel_chart()


@contextlib.contextmanager
def stopwatch(consumer):
    """
    After executing the managed block of code, call the provided consumer with 
    two arguments, the start and end times of the execution of the block.
    """
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    if consumer is not None:
        consumer(start_time, end_time)
    return


class TimeAccumulator:
    """
    A simple consumer for use with stopwatches, that accumulates the total 
    elapsed time over multiple calls, and has a convenience method for printing 
    the total time (and optionally resetting it).
    """

    def __init__(self, initial_time=0.0):
        self.accumulated_time = initial_time
        return

    def __call__(self, start_time, end_time):
        self.accumulated_time += end_time - start_time
        return

    def print(self, caption, accumulated_time=None):
        print("{} {} sec.".format(caption, self.accumulated_time))
        if accumulated_time is not None:
            self.accumulated_time = accumulated_time
        return


class EdgeDataset(Dataset):
    """
    Wrapper class (around an iteration over ConceptNet edges) that enables us 
    to use a torch DataLoader to parallelize generation of training batches.
    """

    def __init__(self, filename, index):
        """
        Construct an edge dataset from a filename (of a tab-separated 
        ConceptNet edge file, which should be in random order), and a (pandas) 
        index mapping (row) numbers to terms of the ConceptNet vocabulary. 
        """
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
        for count, (rel, left, right, weight) in enumerate(self.iter_edges_once()):
            if count % 500000 == 0:
                print("Read {} edges.".format(count))
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
            print("No edges survived filtering; fitting is impossible!")
            raise ValueError
        self.rel_indices = torch.LongTensor(rel_indices)
        self.left_indices = torch.LongTensor(left_indices)
        self.right_indices = torch.LongTensor(right_indices)
        self.edge_weights = torch.FloatTensor(weights)
        return

    def __len__(self):
        return len(self.rel_indices)

    def iter_edges_once(self):
        """Return the edges from the data file."""
        with open(self.filename, encoding="utf-8") as fp:
            for line in fp:
                _assertion, relation, concept1, concept2, _rest = line.split("\t", 4)
                yield (relation, concept1, concept2, 1.)
        return

    def __getitem__(self, i_edge):
        """
        Produce a positive example and weight and a batch of (of size 
        NEG_SAMPLES) of negative examples, derived from the edge at the 
        given index.  The return values are three torch tensors containing 
        the positive example, the negative examples, and the weights.
        """
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
        pos_right = [right_idx]
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

        pos_data = dict(
            rel=torch.LongTensor(pos_rels),
            left=torch.LongTensor(pos_left),
            right=torch.LongTensor(pos_right),
        )
        neg_data = dict(
            rel=torch.LongTensor(neg_rels),
            left=torch.LongTensor(neg_left),
            right=torch.LongTensor(neg_right),
        )
        weights = torch.FloatTensor(weights)
        return dict(positive_data=pos_data, negative_data=neg_data, weights=weights)

    def collate_batch(self, batch):
        """
        Collates batches (as returned by a DataLoader that batches 
        the outputs of calls to __getitem__) into tensors (as required 
        by the train method of SemanticMatchingModel).
        """
        pos_rels = torch.cat(list(x["positive_data"]["rel"] for x in batch))
        pos_left = torch.cat(list(x["positive_data"]["left"] for x in batch))
        pos_right = torch.cat(list(x["positive_data"]["right"] for x in batch))
        neg_rels = torch.cat(list(x["negative_data"]["rel"] for x in batch))
        neg_left = torch.cat(list(x["negative_data"]["left"] for x in batch))
        neg_right = torch.cat(list(x["negative_data"]["right"] for x in batch))
        weights = torch.cat(list(x["weights"] for x in batch))
        pos_data = (pos_rels, pos_left, pos_right)
        neg_data = (neg_rels, neg_left, neg_right)
        return pos_data, neg_data, weights


class CyclingSampler(Sampler):
    """
    Like a sequential sampler, but these samplers cycle repeatedly over the 
    data source, and so have infinite length.
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.index = 0
        return

    def _next_index(self):
        result = self.index
        self.index = (self.index + 1) % len(self.data_source)
        return result

    def __iter__(self):
        return iter(self._next_index, None)  # never raises StopIteration!

    def __len__(self):
        """
        Length makes no sense for a cycling sampler; it is effectively infiinte.
        """
        raise NotImplementedError


class SemanticMatchingModel(nn.Module):
    """
    The PyTorch model for semantic matching energy over ConceptNet.
    """

    def __init__(self, frame, use_cuda=True, term_dim=300, relation_dim=10):
        """
        Parameters:

        `frame`: a pandas DataFrame of pre-trained word embeddings over the
        vocabulary of ConceptNet. `conceptnet5.vectors.formats.load_hdf`
        can load these.

        `term_dim`: the number of dimensions in the term embedding.

        `use_cuda`: whether to use GPU-accelerated PyTorch objects.

        `relation_dim`: the number of dimensions in the relation embeddings.
        Unlike SME as published, this can differ from the dimensionality of
        the term embeddings.
        """
        super().__init__()

        # Initialize term embeddings, including the index that converts terms
        # from strings to row numbers
        print("Intializing term embeddings.")
        self.index = frame.index
        self.term_vecs = nn.Embedding(len(self.index), term_dim, sparse=True)
        self.term_vecs.weight.data.copy_(torch.from_numpy(frame.values))

        # The assoc_tensor is a (k_2 x k_1 x k_1) tensor that represents
        # interactions between the relations and the terms. k_1 is the
        # dimensionality of the term embeddings, and k_2 is the dimensionality
        # of the relation embeddings.
        #
        # We pass the dimensions in the order (k_1, k_1, k_2) because the
        # PyTorch convention is that inputs come before outputs, but the
        # resulting tensor is (k_2 x k_1 x k_1) because the math convention
        # is that outputs are on the left.
        print("Initializing association tensor and relation embedding.")
        self.assoc_tensor = nn.Bilinear(term_dim, term_dim, relation_dim, bias=True)
        self.rel_vecs = nn.Embedding(N_RELS, relation_dim, sparse=True)

        # Using CUDA to run on the GPU requires different data types
        print("Setting model device (cpu/gpu).")
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Create (and register) the identity tensor used to reset the synonym
        # relation.  (Register it so it will be included in the state dict.)
        print("Initializing identity slice.")
        self.register_buffer(
            "identity_slice",
            torch.tensor(np.eye(term_dim), dtype=torch.float32, device=self.device),
        )
        self.reset_synonym_relation()

        # Learnable priors for how confident to be in arbitrary statements.
        # These are used to convert the tensor products linearly into logits.
        # The initial values of these are set to numbers that appeared
        # reasonable based on a previous run.
        print("Initializing logit scale factors.")
        self.truth_multiplier = nn.Parameter(
            torch.tensor(5.0, dtype=torch.float32)
        )
        self.truth_offset = nn.Parameter(
            torch.tensor(-3.0, dtype=torch.float32)
        )
        
        # Make sure everything is on the right device.
        print("Moving model to selected device (cpu/gpu).")
        self.to(self.device)

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
        total_accum = TimeAccumulator()
        incremental_accum = TimeAccumulator()
        print("Loading frame from file {}.".format(INITIAL_VECS_FILENAME))
        with stopwatch(total_accum), stopwatch(incremental_accum):
            frame = load_hdf(INITIAL_VECS_FILENAME)
        incremental_accum.print("Reading frame from file took", accumulated_time=0.0)
        print("Filtering terms in frame by language.")
        with stopwatch(total_accum), stopwatch(incremental_accum):
            labels = [
                label
                for label in frame.index
                if get_uri_language(label) in LANGUAGES_TO_USE
            ]
            frame = frame.loc[labels]
        incremental_accum.print("Filtering took", accumulated_time=0.0)
        print("Casting frame to float32.")
        with stopwatch(total_accum), stopwatch(incremental_accum):
            frame = frame.astype(np.float32)
        incremental_accum.print("Casting took", accumulated_time=0.0)
        total_accum.print("Loading frame (including reading, filtering, casting) took")
        return frame

    @staticmethod
    def load_model(filename):
        """
        Load the SME model from a file.
        """
        total_accum = TimeAccumulator()
        incremental_accum = TimeAccumulator()
        print("Loading model from file {}.".format(filename))
        with stopwatch(total_accum):
            frame = SemanticMatchingModel.load_initial_frame()
        print("Constructing model from frame.")
        with stopwatch(total_accum), stopwatch(incremental_accum):
            model = SemanticMatchingModel(frame)
        incremental_accum.print("Constructing model took", accumulated_time=0.0)
        print("Restoring model state from file.")
        with stopwatch(total_accum), stopwatch(incremental_accum):
            # We have adopted the convention that models are moved to the
            # cpu before saving their state (as else loading would need
            # twice the gpu memory).  We (currently) do not support building
            # a model on the gpu and loading it on the cpu (or conversely).
            model.cpu()
            model.load_state_dict(torch.load(filename))
            model.to(model.device)
        incremental_accum.print("Restoring took", accumulated_time=0.0)
        total_accum.print("Total time to load model:")
        return model

    def evaluate_conceptnet(self, dataset, cutoff_value=-1, output_filename=None):
        """
        Use the SME model to "sanity-check" existing edges in ConceptNet.
        We particularly highlight edges that get a value of -1 or less, which
        may be bad or unhelpful edges.
        """
        self.eval()  # in case someday we add submodules where the mode matters

        if output_filename:
            out = open(output_filename, "w", encoding="utf-8")
        else:
            out = None
        for rel, left, right, weight in dataset.iter_edges_once():
            try:
                rel_idx = RELATION_INDEX.get_loc(rel)
                left_idx = self.index.get_loc(left)
                right_idx = self.index.get_loc(right)
            except KeyError:
                continue

            rel_tensor = torch.tensor([rel_idx], dtype=torch.int64, device=self.device)
            left_tensor = torch.tensor(
                [left_idx], dtype=torch.int64, device=self.device
            )
            right_tensor = torch.tensor(
                [right_idx], dtype=torch.int64, device=self.device
            )

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
        related_mat = np.einsum("i,ijk->jk", rel_vec, assoc_t)
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
    # Torch sparse tensors may be stored un-coalesced (i.e., there can be
    # more than one value assigned to each non-zero entry; the actual
    # value of the tensor at that entry is the sum of those assigned values).
    # Before computing the norm of a sparse tensor it must be coalesced, and
    # after that it is only necessary to iterate over the non-zero entries to
    # find the norm.
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float("inf"):

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
            if p.grad.data.layout == torch.sparse_coo:
                p.grad.data = p.grad.data.coalesce()
    return total_norm


class DataParallelizedModule(nn.Module):
    """
    Similarly to nn.DataParallel, this class of modules serves to wrap 
    other modules and run them in parallel over multiple gpus, splitting 
    training (or testing/application) batches between the gpus (over the 
    first dimension of the batch, which is assumed to correspond to data 
    points within the batch).  
    
    Unlike nn.DataParallel, all of the wrapped module's parameters will be 
    copied to all gpus during construction of the wrapper, and only gradient 
    data will be copied from gpu to gpu during training (no data should need 
    to be copied between gpus during forward application).  However, training 
    should only use loss functions that can be expressed as averages over all 
    data points of a batch of some per-data-point loss.  Each training batch 
    must be presented as a tuple of tensors all of which have equal sizes (or 
    at least sizes in fixed proportions) in dimension 0 (corresponding to the 
    points in the batch).  Also, the set of parameters of the wrapped module 
    is assumed not to change (of course their values can change) over the 
    lifetime of the wrapper object.
    
    Note that during training of a wrapped module, it is necessary to call 
    the wrapper's broadcast_gradients method immediately following the 
    backpropagation of gradients (i.e. typically right after calling 
    .backward on some loss tensor), in order to share corrections to the 
    computed gradients between the gpus.
    
    Specifically, if L is the average loss over an entire (mini-)batch, of 
    size n data points, that batch is scattered over k gpus as chunks of 
    sizes n_0, ..., n_(k-1) (with sum equal to n), and L_i is the average loss 
    over the i-th chunk, then L = w_0 * L_0 + ... + w_k-1 * L_(k-1), where 
    w_i = n_i / n, and so for any parameter p
    
        dL/dp = w_0 dL_0/dp + ... + w_(k-1) * dL_(k-1)/dp.
    
    The broadcast_gradients method collects the individual pieces of gradient 
    data dL_i/dp from all the gpus, computes the unified gradient data dL/dp 
    (for every parameter) and updates the gradients on every gpu.
    """

    def __init__(self, module, device, copy_fn, parallelize=True):
        """
        Construct a parallelizing wrapper for the given module (an instance 
        of nn.Module).  The module will be moved to the given device (if 
        not already there) and copies will be made using the given copy 
        function (which should accept a module and a device, and return a 
        copy of the module suitable for placement on that device) and 
        moved to all other available gpus.
        
        If the (optional) parallelize argument is set to False, or if the 
        requested device is the cpu, or if multiple gpus are not available, 
        the model will be moved to the given device (if possible) for execution 
        there.
        """
        super().__init__()

        # If the requested device is the cpu, or there is only one gpu,
        # or parallelization is turned off, don't parallelize.
        if (
            device == torch.device("cpu")
            or not torch.cuda.is_available()
            or torch.cuda.device_count() < 2
            or not parallelize
        ):
            self.children = [module]
            self.devices = [device]
            module.to(device)
            self.add_module("child:{0}", module)
            self.chunk_sizes = torch.zeros(1, dtype=torch.int64)
            return

        device_ids = list(range(torch.cuda.device_count()))
        devices = [torch.device("cuda:{}".format(d_id)) for d_id in device_ids]
        if device not in devices:
            device = devices[0]
        self.children = [module]  # put the original copy first
        self.devices = [device]  # the original copy will end up on this device

        # Register the child as a submodule of the wrapper, so that the
        # autograd machinery will update its parameters when the forward
        # method of the wrapper is called.
        self.add_module("child:{}".format(device.index), module)

        # Save a list of all available devices (with the one corresponding to
        # the original wrapped module first).
        for dev in devices:
            if dev != device:
                self.devices.append(dev)

        # Make a copy for each additional device, put it on the corresponding
        # device, and register it as a submodule.
        for dev in self.devices[1:]:
            module.cpu()  # in case the copy_fn moved it
            module_copy = copy_fn(module, dev)
            module_copy.to(dev)  # in case the copy_fn didn't move the copy
            self.children.append(module_copy)
            self.add_module("child:{}".format(dev.index), module_copy)

        # Put the original copy on the requested device.
        module.to(device)

        # During forward evaluation of the wrapper it is necessary to keep
        # track of the sizes of the chunks of the input batch(es) that are
        # handed off to each child.  So we create a 1D tensor holding those
        # sizes.
        self.chunk_sizes = torch.zeros(len(self.children), dtype=torch.int64)

    @property
    def device(self):
        """
        The nominal device of a parallelized module is the first device used.
        """
        return self.devices[0]

    def forward(self, *args):
        """
        Scatter the supplied args (assumed to be a list of tensors) across 
        the child modules, and gather their outputs (assumed to be single 
        tensors) back to the first gpu.  Also, accumulate the sizes of the 
        scattered chunks (for later use in updating parameter gradients).
        """
        # Data is scattered into chunks by splitting on dimension 0.
        split_dimension = 0

        # We assume the input argument tensors have proportional sizes in
        # the splitting dimension, so any of them can be used as representative
        # of the size of the input batch, and it chunks as representative of
        # the sizes of the chunks
        representative = 0

        if len(self.children) <= 1:
            return self.children[0](*args)

        device_ids = [device.index for device in self.devices]

        # Scatter each arg across the (multiple) devices.  For each arg,
        # calling comm.scatter gives us a tuple of chunks, one chunk on each
        # device.  We populate a list (with length equal to the number of
        # devices) whose entries are lists (with length equal to the number
        # of args) of the chunks on each device.
        chunk_lists = list(list() for i_device in device_ids)
        for arg in args:
            chunks = torch.cuda.comm.scatter(arg, device_ids)
            for i_child, chunk in enumerate(chunks):
                chunk_lists[i_child].append(chunk)

        # In order to apply the child modules to the appropriate collections
        # of arg chunks, convert each list of chunks on one device to a tuple.
        chunks = list(tuple(chunk_list) for chunk_list in chunk_lists)

        # Now we can apply the children modules to the chunks of data.  We
        # collect the outputs in a list, and also update the running tally
        # of the sizes of the chunks processed by each child.
        outputs = []
        for i_child, (module, chunk) in enumerate(zip(self.children, chunks)):
            chunk_size = chunk[representative].size()[split_dimension]
            self.chunk_sizes[i_child] += chunk_size
            output = module(*chunk)
            outputs.append(output)
        assert len(self.children) == len(outputs)

        # Finally, we put the separate outputs of the children together into
        # a unified (concatenated) output, and place it on the first device.
        output = torch.cuda.comm.gather(outputs, destination=self.devices[0].index)
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
        in the individual children, with coefficients proportional to the 
        batch chunk sizes from the forward computation), and distribute these 
        common gradients back to all the children.
        """
        if len(self.children) <= 1:
            return

        # Compute the coefficients of the convex combination, proportional to
        # the sizes of the chunks of the batch that were processed by each
        # child.
        weights = self.chunk_sizes.to(torch.float32) / self.chunk_sizes.sum().to(
            torch.float32
        )
        weights = [
            weights[i_device].item() for i_device, device in enumerate(self.devices)
        ]

        # Update each parameter's gradients on all children.
        for name, param in self.children[0].named_parameters():
            if param.grad is None:
                continue
            param_copies = [param]

            # Collect the other children's copies of this parameter in a list.
            for other_module in self.children[1:]:
                # Find the other module's parameter of the same name.
                other_module_params = list(
                    p
                    for other_name, p in other_module.named_parameters()
                    if other_name == name
                )
                assert len(other_module_params) == 1
                param_copies.append(other_module_params[0])

            # Find the sum, over all child modules, of the gradient for this
            # parameter in the child, multiplied by the corresponding weights
            # (as determined above by the relative sizes of the batch chunks
            # processed by each child), and place it on the first device.
            param_grad = torch.cuda.comm.reduce_add(
                list(
                    param_copy.grad.mul_(weight)
                    for param_copy, weight in zip(param_copies, weights)
                ),
                destination=self.devices[0].index,
            )

            # Now send the weighted sum to all the child modules on their
            # devices, replacing their values of the parameter's gradient.
            for i_param_copy, param_copy in enumerate(param_copies):
                param_copy.grad = param_grad.to(self.devices[i_param_copy])

    def synchronize_children(self, tolerance=5e-6):
        """
        In principle, if broadcast_gradients is called on every training step, 
        the child modules should always agree on all parameters.  In practice, 
        some optimizers sometimes introduce slight discrepancies (e.g. 
        optim.SGD with sparse gradients, which does not coalesce such gradients 
        at every step).  This method can be called periodically to reset the 
        parameters of all children to the values of the first child (the 
        original module), and to print warnings if the parameters have diverged 
        by more than the given (absolute) tolerance.
        """
        msg = "Warning: {} differs between parallelized models by {}."
        model = self.children[0]
        for child, device in zip(self.children[1:], self.devices[1:]):
            for name, param in model.named_parameters():
                param_data_cpu = param.data.cpu()
                # Find the param on the child of the same name.
                child_params = list(
                    p
                    for other_name, p in child.named_parameters()
                    if name == other_name
                )
                assert len(child_params) == 1
                # Find its difference from the param on the first child.
                difference = torch.norm(param_data_cpu - child_params[0].data.cpu())
                if difference > tolerance:
                    print(msg.format(name, difference))
                # Copy the param to the child (but first free up space).
                child_params[0].data = child_params[0].data.new_empty((0,))
                child_params[0].data = param_data_cpu.to(device)


def train_model(model, dataset, num_batch_workers=0, use_multiple_gpus=False):
    """
    Incrementally train the model on the given dataset.

    If a positive number of batch worker processes is requested, generation 
    of training data batches will be done in parallel across multiple CPU cores 
    by spawning that number of child processes.

    If use of multiple GPUs is requested (and they are available), evaluation 
    of batches will be parallelized across all available GPUs.

    As it is, this function will never return. It writes the results so far
    to `sme/sme.model` every 5000 iterations, and you can run it for as
    long as you want.
    """
    print("Starting model training.")
    model.train()

    print("Making parallelized model.")
    def model_copier(model, device):
        model.cpu()
        frame = pd.DataFrame(model.term_vecs.weight.data.numpy(), index=model.index)
        new_model = SemanticMatchingModel(frame, use_cuda=False)
        new_model.load_state_dict(model.state_dict())
        new_model.device = device
        new_model.to(device)
        return new_model

    parallel_model = DataParallelizedModule(model, model.device, copy_fn=model_copier, parallelize=use_multiple_gpus)

    # Relative loss says that the positive examples should outrank their
    # corresponding negative examples, with a difference of at least 1
    # logit between them. If the difference is less than this (especially
    # if it's negative), this adds to the relative loss.
    print("Making loss functions.")
    relative_loss_function = nn.MarginRankingLoss(margin=1)

    # Absolute loss measures the cross-entropy of the predictions:
    # true statements should get positive values, false statements should
    # get negative values, and the sigmoid of those values should be a
    # probability that accurately reflects the model's confidence.
    absolute_loss_function = nn.BCEWithLogitsLoss()

    print("Making optimizer.")
    optimizer = optim.SGD(parallel_model.parameters(), lr=0.1)
    losses = []

    print("Making loss targets.")
    true_target = torch.ones(
        [BATCH_SIZE], dtype=torch.float32, device=parallel_model.device
    )
    false_target = torch.zeros(
        [BATCH_SIZE], dtype=torch.float32, device=parallel_model.device
    )
    steps = 0

    # Note that you want drop_last=False with a CyclingSampler.
    print("Making data loader.")
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=num_batch_workers,
        collate_fn=dataset.collate_batch,
        sampler=CyclingSampler(dataset),
        pin_memory=True,
    )

    print("Entering training (batch) loop.")
    for pos_batch, neg_batch, weights in data_loader:
        if parallel_model.device != torch.device("cpu"):
            pos_batch = tuple(
                x.cuda(device=parallel_model.device, non_blocking=True)
                for x in pos_batch
            )
            neg_batch = tuple(
                x.cuda(device=parallel_model.device, non_blocking=True)
                for x in neg_batch
            )
            weights = weights.cuda(device=parallel_model.device, non_blocking=True)

        parallel_model.zero_grad()

        pos_energy = parallel_model(*pos_batch)
        neg_energy = parallel_model(*neg_batch)

        abs_loss = absolute_loss_function(pos_energy, true_target)
        rel_loss = 0
        for neg_index in range(NEG_SAMPLES):
            neg_energy_slice = neg_energy[neg_index::NEG_SAMPLES]
            rel_loss += relative_loss_function(
                pos_energy, neg_energy_slice, true_target
            )
            abs_loss += absolute_loss_function(neg_energy_slice, false_target)

        loss = abs_loss + rel_loss
        loss.backward()

        parallel_model.broadcast_gradients()
        for model_copy in parallel_model.children:
            clip_grad_norm(model_copy.parameters(), 1)

        optimizer.step()

        for model_copy in parallel_model.children:
            model_copy.reset_synonym_relation()

        losses.append(loss.data.cpu().item())
        steps += 1

        if steps in (1, 10, 20, 50, 100) or steps % 100 == 0:
            model.show_debug(neg_batch, neg_energy, False)
            model.show_debug(pos_batch, pos_energy, True)
            avg_loss = np.mean(losses)
            print(
                "%d steps, loss=%4.4f, abs=%4.4f, rel=%4.4f"
                % (steps, avg_loss, abs_loss, rel_loss)
            )
            losses.clear()

        if steps % 5000 == 0:
            incremental_accum = TimeAccumulator()
            print("Saving model.")
            with stopwatch(incremental_accum):
                model.cpu()
            incremental_accum.print("Moving model to cpu took", accumulated_time=0.0)
            with stopwatch(incremental_accum):
                torch.save(model.state_dict(), MODEL_FILENAME)
            incremental_accum.print("Saving to disk took", accumulated_time=0.0)

            with stopwatch(incremental_accum):
                model.to(model.device)
            incremental_accum.print("Moving model back to its device took", accumulated_time=0.0)
            print("saved")

            # With optim.SGD as the optimizer and when the model is
            # parallelized across multiple GPU's we see slight divergences
            # over time between the parallel models.  (This did not happen
            # with a custom optimizer that coalesces sparse gradients at
            # every training iteration.)  So we force agreement between the
            # children periodically.

            print("Synchronizing parallel model.")
            with stopwatch(incremental_accum):
                parallel_model.synchronize_children()
            incremental_accum.print("Synchonizing took", accumulated_time=0.0)

    print()


def get_model():
    """
    Instantiate a model, either by loading it from the saved checkpoint, or
    by creating it from scratch if nothing is there.
    """
    if os.access(MODEL_FILENAME, os.F_OK):
        print("Loading previously saved model.")
        model = SemanticMatchingModel.load_model(MODEL_FILENAME)
    else:
        print("Creating a new model.")
        total_accum = TimeAccumulator()
        incremental_accum = TimeAccumulator()
        with stopwatch(total_accum):
            frame = SemanticMatchingModel.load_initial_frame()
        print("Normalizing initial frame.")
        with stopwatch(total_accum), stopwatch(incremental_accum):
            frame = l2_normalize_rows(frame)
        incremental_accum.print('Normalizing frame took', accumulated_time=0.0)
        print('Constructing model from frame.')
        with stopwatch(total_accum), stopwatch(incremental_accum):
            model = SemanticMatchingModel(frame)
        incremental_accum.print("Construction took", accumulated_time=0.0)
        total_accum.print('Total model creation time:')
    return model


if __name__ == "__main__":
    print("Starting semantic matching.")
    model = get_model()
    print("Initializing edge dataset ....")
    dataset_accumulator = TimeAccumulator()
    with stopwatch(dataset_accumulator):
        dataset = EdgeDataset(EDGES_FILENAME, model.index)
    dataset_accumulator.print("Edge dataset initialization took")
    print("Edge dataset contains {} edges.".format(len(dataset)))
    train_model(model, dataset, num_batch_workers=NUM_BATCH_WORKERS, use_multiple_gpus=USE_MULTIPLE_GPUS)
    # model.evaluate_conceptnet(dataset)
