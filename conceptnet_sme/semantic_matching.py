import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from conceptnet5.vectors.formats import save_hdf, save_npy
import numpy as np
import pandas as pd
import random
import pathlib
import os

from conceptnet_sme.relations import (
    COMMON_RELATIONS, ALL_RELATIONS, SYMMETRIC_RELATIONS, ENTAILED_RELATIONS,
)
from conceptnet5.uri import uri_prefix, assertion_uri
from conceptnet5.nodes import get_uri_language
from conceptnet5.util import get_data_filename
from conceptnet5.vectors.formats import load_hdf
from conceptnet5.vectors.transforms import l2_normalize_rows


RELATION_INDEX = pd.Index(COMMON_RELATIONS)
N_RELS = len(RELATION_INDEX)
INITIAL_VECS_FILENAME = get_data_filename('vectors/numberbatch-biased.h5')
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


def iter_edges_forever(filename):
    """
    Turn an iterator of ConceptNet edges into an iterator that runs forever
    in a cycle.
    """
    while True:
        yield from iter_edges_once(filename)


def iter_edges_once(filename):
    """
    Iterate through the edges in ConceptNet, given an "edges-shuf.csv" file
    that contains tab-separated edge data in random order.
    """
    for line in open(filename, encoding='utf-8'):
        _assertion, relation, concept1, concept2, _rest = line.split('\t', 4)
        yield (relation, concept1, concept2, 1.)


class SemanticMatchingModel(nn.Module):
    """
    The PyTorch model for semantic matching energy over ConceptNet.
    """
    def __init__(self, frame, use_cuda=True, relation_dim=10, batch_size=100):
        """
        Parameters:

        `frame`: a pandas DataFrame of pre-trained word embeddings over the
        vocabulary of ConceptNet. `conceptnet5.vectors.formats.load_hdf`
        can load these.

        `use_cuda`: whether to use GPU-accelerated PyTorch objects.

        `relation_dim`: the number of dimensions in the relation embeddings.
        Unlike SME as published, this can differ from the dimensionality of
        the term embeddings.

        `batch_size`: how many positive and neg
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
        self.assoc_tensor = nn.Bilinear(
            self.term_dim, self.term_dim, self.relation_dim, bias=True
        )
        self.rel_vecs = nn.Embedding(N_RELS, self.relation_dim)

        # Using CUDA to run on the GPU requires different data types
        if use_cuda:
            self.float_type = torch.cuda.FloatTensor
            self.int_type = torch.cuda.LongTensor
            self.term_vecs = self.term_vecs.cuda()
            self.rel_vecs = self.rel_vecs.cuda()
            self.assoc_tensor = self.assoc_tensor.cuda()
        else:
            self.float_type = torch.FloatTensor
            self.int_type = torch.LongTensor

        self.identity_slice = self.float_type(np.eye(self.term_dim))
        self.reset_synonym_relation()

        # Learnable priors for how confident to be in arbitrary statements.
        # These are used to convert the tensor products linearly into logits.
        self.truth_multiplier = nn.Parameter(self.float_type([5.]))
        self.truth_offset = nn.Parameter(self.float_type([-3.]))

    def ltvar(self, numbers):
        """
        This is something we have to do a lot: take a list or a numpy array
        of integers, and turn it into a Variable containing a LongTensor.
        """
        return autograd.Variable(self.int_type(numbers))

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

    def positive_negative_batch(self, edge_iterator):
        """
        Produce a batch of positive examples and a batch of negative examples,
        expressed as PyTorch variables containing 1-D LongTensors (that is,
        vectors of integers).

        The negative batch will be NEG_SAMPLES times as large as the positive
        batch.

        Returns the positive batch, the negative batch, and the weights for
        the batch (though currently these weights are always 1).
        """
        pos_rels = []
        pos_left = []
        pos_right = []

        neg_rels = []
        neg_left = []
        neg_right = []

        weights = []

        for rel, left, right, weight in edge_iterator:
            try:
                if rel not in COMMON_RELATIONS:
                    continue
                if not ENTAILED_INDICES[rel]:
                    continue

                left = uri_prefix(left)
                right = uri_prefix(right)

                # Possibly swap the sides of a relation
                if coin_flip() and rel in SYMMETRIC_RELATIONS:
                    left, right = right, left

                rel_idx = RELATION_INDEX.get_loc(rel)
                left_idx = self.index.get_loc(left)
                right_idx = self.index.get_loc(right)

                # Possibly replace a relation with a more general relation
                if coin_flip():
                    rel_idx = random.choice(ENTAILED_INDICES[rel])
                    rel = COMMON_RELATIONS[rel_idx]

            except KeyError:
                continue

            pos_rels.append(rel_idx)
            pos_left.append(left_idx)
            pos_right.append(right_idx)
            weights.append(weight)

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

            if len(weights) == self.batch_size:
                break

        pos_data = (self.ltvar(pos_rels), self.ltvar(pos_left), self.ltvar(pos_right))
        neg_data = (self.ltvar(neg_rels), self.ltvar(neg_left), self.ltvar(neg_right))
        weights = autograd.Variable(self.float_type(weights))
        return pos_data, neg_data, weights

    def make_batches(self, edge_iterator):
        """
        An infinite iterator of training batches.
        """
        while True:
            yield self.positive_negative_batch(edge_iterator)

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
        for rel, left, right, weight in iter_edges_once(get_data_filename('collated/sorted/edges-shuf.csv')):
            try:
                rel_idx = RELATION_INDEX.get_loc(rel)
                left_idx = self.index.get_loc(left)
                right_idx = self.index.get_loc(right)
            except KeyError:
                continue

            model_output = self(self.ltvar([rel_idx]), self.ltvar([left_idx]), self.ltvar([right_idx]))
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
        relative_loss_function = nn.MarginRankingLoss(margin=1)
        absolute_loss_function = nn.BCEWithLogitsLoss()

        optimizer = optim.SGD(self.parameters(), lr=0.1, weight_decay=1e-9)
        losses = []
        true_target = autograd.Variable(self.float_type([1] * self.batch_size))
        false_target = autograd.Variable(self.float_type([0] * self.batch_size))
        steps = 0
        for pos_batch, neg_batch, weights in self.make_batches(
            iter_edges_forever(get_data_filename('collated/sorted/edges-shuf.csv'))
        ):
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
