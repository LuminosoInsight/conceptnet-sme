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
import random
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
    def __init__(self, filename, index, n_terms, batch_size):
        '''
        Construct an edge dataset from a filename (of a tab-separated 
        ConceptNet edge file, which should be in random order), a (pandas) 
        index mapping (row) numbers to terms of the ConceptNet vocabulary, 
        the (total) number of such terms, and a batch size. 
        '''
        super().__init__()
        self.filename = filename
        self.index = index
        self.n_terms = n_terms
        self.batch_size = batch_size
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
        given index.  The return values are three torch variables containing 
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

        pos_data = dict(rel=autograd.Variable(torch.LongTensor(pos_rels)),
                        left=autograd.Variable(torch.LongTensor(pos_left)),
                        right=autograd.Variable(torch.LongTensor(pos_right)))
        neg_data = dict(rel=autograd.Variable(torch.LongTensor(neg_rels)),
                        left=autograd.Variable(torch.LongTensor(neg_left)),
                        right=autograd.Variable(torch.LongTensor(neg_right)))
        weights = autograd.Variable(torch.FloatTensor(weights))
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
    def __init__(self, frame, use_cuda=True, relation_dim=10, batch_size=4096):
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
        super().__init__()
        self.n_terms, self.term_dim = frame.shape
        self.relation_dim = relation_dim
        self.batch_size = batch_size

        # Initialize term embeddings, including the index that converts terms
        # from strings to row numbers
        self.index = frame.index
        self.term_vecs = nn.Embedding(frame.shape[0], self.term_dim)
        self.term_vecs.weight.data.copy_(
            torch.from_numpy(frame.values)
        )

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
            self.term_dim, self.term_dim, self.relation_dim, bias=True
        )
        self.rel_vecs = nn.Embedding(N_RELS, self.relation_dim)

        # Using CUDA to run on the GPU requires different data types
        if use_cuda and torch.cuda.is_available():
            self.float_type = torch.cuda.FloatTensor
            self.int_type = torch.cuda.LongTensor
            self.term_vecs = self.term_vecs.cuda()
            self.rel_vecs = self.rel_vecs.cuda()
            self.assoc_tensor = self.assoc_tensor.cuda()
        else:
            self.float_type = torch.FloatTensor
            self.int_type = torch.LongTensor

        print('Initializing edge dataset ....')
        dataset_accumulator = TimeAccumulator()
        with stopwatch(dataset_accumulator):
            self.dataset = EdgeDataset(
                EDGES_FILENAME,
                self.index,
                self.n_terms, 
                self.batch_size
            )
        dataset_accumulator.print('Edge dataset initialization took')
        print('Edge dataset contains {} edges.'.format(len(self.dataset)))

        self.identity_slice = self.float_type(np.eye(self.term_dim))
        self.reset_synonym_relation()

        # Learnable priors for how confident to be in arbitrary statements.
        # These are used to convert the tensor products linearly into logits.
        # The initial values of these are set to numbers that appeared
        # reasonable based on a previous run.
        self.truth_multiplier = nn.Parameter(self.float_type([5.]))
        self.truth_offset = nn.Parameter(self.float_type([-3.]))

        if use_cuda and torch.cuda.is_available():
            self.cuda()
            if torch.cuda.device_count() > 1:
                self = nn.DataParallel(self)

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
        frame = load_hdf(INITIAL_VECS_FILENAME)
        labels = [
            label for label in frame.index
            if get_uri_language(label) in LANGUAGES_TO_USE
        ]
        frame = frame.loc[labels]
        return frame.astype(np.float32)

    @staticmethod
    def load_model(filename):
        """
        Load the SME model from a file.
        """
        frame = SemanticMatchingModel.load_initial_frame()
        model = SemanticMatchingModel(l2_normalize_rows(frame))
        model.load_state_dict(torch.load(filename))
        return model

    def ltvar(self, numbers):
        """
        This is something we have to do a lot: take a list or a numpy array
        of integers, and turn it into a Variable containing a LongTensor.
        """
        return autograd.Variable(self.int_type(numbers))

    def evaluate_conceptnet(self, cutoff_value=-1, output_filename=None):
        """
        Use the SME model to "sanity-check" existing edges in ConceptNet.
        We particularly highlight edges that get a value of -1 or less, which
        may be bad or unhelpful edges.
        """
        if output_filename:
            out = open(output_filename, 'w', encoding='utf-8')
        else:
            out = None
        for rel, left, right, weight in self.dataset.iter_edges_once():
            try:
                rel_idx = RELATION_INDEX.get_loc(rel)
                left_idx = self.index.get_loc(left)
                right_idx = self.index.get_loc(right)
            except KeyError:
                continue

            model_output = self(self.ltvar([rel_idx]),
                                self.ltvar([left_idx]),
                                self.ltvar([right_idx]))
            value = model_output.data[0]
            assertion = assertion_uri(rel, left, right)
            if value < cutoff_value:
                print("%4.4f\t%s" % (value, assertion))
            if out is not None:
                print("%4.4f\t%s" % (value, assertion), file=out)

    def train(self):
        """
        Incrementally train the model.

        As it is, this method will never return. It writes the results so far
        to `sme/sme.model` every 5000 iterations, and you can run it for as
        long as you want.
        """
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

        optimizer = optim.SGD(self.parameters(), lr=0.1, weight_decay=1e-9)
        losses = []
        true_target = autograd.Variable(self.float_type([1] * self.batch_size))
        false_target = autograd.Variable(self.float_type([0] * self.batch_size))
        steps = 0

        # Note that you want drop_last=False with a CyclingSampler.
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                 drop_last=False, num_workers=10,
                                 collate_fn=self.dataset.collate_batch,
                                 sampler=CyclingSampler(self.dataset),
                                 pin_memory=True)
        for pos_batch, neg_batch, weights in data_loader:
            if self.int_type == torch.cuda.LongTensor:
                pos_batch = tuple(x.cuda(async=True) for x in pos_batch)
                neg_batch = tuple(x.cuda(async=True) for x in neg_batch)
            if self.float_type == torch.cuda.FloatTensor:
                weights = weights.cuda(async=True)
            self.zero_grad()
            pos_energy = self(*pos_batch)
            neg_energy = self(*neg_batch)

            abs_loss = absolute_loss_function(pos_energy, true_target)
            rel_loss = 0
            for neg_index in range(NEG_SAMPLES):
                neg_energy_slice = neg_energy[neg_index::NEG_SAMPLES]
                rel_loss += relative_loss_function(pos_energy, neg_energy_slice, true_target)
                abs_loss += absolute_loss_function(neg_energy_slice, false_target)

            loss = abs_loss + rel_loss
            loss.backward()

            nn.utils.clip_grad_norm(self.parameters(), 1)
            optimizer.step()
            self.reset_synonym_relation()

            losses.append(loss.data[0])
            steps += 1
            
            if steps in (1, 10, 20, 50, 100) or steps % 100 == 0:
                self.show_debug(neg_batch, neg_energy, False)
                self.show_debug(pos_batch, pos_energy, True)
                avg_loss = np.mean(losses)
                print("%d steps, loss=%4.4f, abs=%4.4f, rel=%4.4f" % (
                    steps, avg_loss, abs_loss, rel_loss
                ))
                losses.clear()
            if steps % 5000 == 0:
                torch.save(self.state_dict(), MODEL_FILENAME)
                print("saved")
        print()

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


def get_model():
    """
    Instantiate a model, either by loading it from the saved checkpoint, or
    by creating it from scratch if nothing is there.
    """
    if os.access(MODEL_FILENAME, os.F_OK):
        model = SemanticMatchingModel.load_model(MODEL_FILENAME)
    else:
        frame = SemanticMatchingModel.load_initial_frame()
        model = SemanticMatchingModel(l2_normalize_rows(frame))
    return model


if __name__ == '__main__':
    model = get_model()
    model.train()
    # model.evaluate_conceptnet()
