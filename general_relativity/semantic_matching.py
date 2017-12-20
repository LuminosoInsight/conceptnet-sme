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

from general_relativity.relations import (
    COMMON_RELATIONS, ALL_RELATIONS, SYMMETRIC_RELATIONS, ENTAILED_RELATIONS,
    reverse_relation
)
from conceptnet5.vectors.debias import GENDERED_WORDS, GENDER_NEUTRAL_WORDS, MALE_WORDS, FEMALE_WORDS
from conceptnet5.uri import uri_prefix, assertion_uri
from conceptnet5.nodes import standardized_concept_uri
from conceptnet5.util import get_data_filename
from conceptnet5.vectors.formats import load_hdf
from conceptnet5.vectors.transforms import l2_normalize_rows


RELATION_INDEX = pd.Index(COMMON_RELATIONS)
N_RELS = len(RELATION_INDEX)
INITIAL_VECS_FILENAME = get_data_filename('vectors/numberbatch-biased.h5')
MODEL_FILENAME = get_data_filename('vectors/sme.model')
NEG_SAMPLES = 5


random.seed(0)


def coin_flip():
    return random.choice([False, True])


def _make_rel_chart():
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
    while True:
        yield from iter_edges_once(filename)


def iter_edges_once(filename):
    for line in open(filename, encoding='utf-8'):
        _assertion, relation, concept1, concept2, _rest = line.split('\t', 4)
        yield (relation, concept1, concept2, 1.)


def inappropriateness_loss(bias_vals, appropriateness_vals):
    weights = torch.clamp(torch.tanh(-appropriateness_vals), 0, 1)
    weights /= torch.sum(weights)
    weighted_mse = torch.sum(weights * bias_vals ** 2)
    return weighted_mse


class SemanticMatchingModel(nn.Module):
    def __init__(self, frame, use_cuda=True, relation_dim=10, batch_size=100):
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

        self.truth_multiplier = nn.Parameter(self.float_type([5.]))
        self.truth_offset = nn.Parameter(self.float_type([-3.]))

        self.gender_bias = nn.Linear(self.term_dim, 1, bias=True).cuda()
        self.gender_appropriateness = nn.Linear(self.term_dim, 1, bias=True).cuda()

        self.bias_indices = {
            'gendered': self.precompute_term_indices(GENDERED_WORDS),
            'neutral': self.precompute_term_indices(GENDER_NEUTRAL_WORDS),
            'female': self.precompute_term_indices(FEMALE_WORDS),
            'male': self.precompute_term_indices(MALE_WORDS),
        }

    def precompute_term_indices(self, terms, language='en'):
        return self.ltvar(
            [
                self.index.get_loc(standardized_concept_uri(language, term))
                for term in terms
                if standardized_concept_uri(language, term) in self.index
            ]
        )

    def reset_synonym_relation(self):
        self.assoc_tensor.weight.data[0] = self.identity_slice
        self.rel_vecs.weight.data[0, :] = 0
        self.rel_vecs.weight.data[0, 0] = 1

        related_mat = torch.sum(self.assoc_tensor.weight * self.rel_vecs.weight[1].unsqueeze(1).unsqueeze(2), 0).data
        related_mat[range(self.term_dim), range(self.term_dim)] = 0.
        self.related_mat = autograd.Variable(self.float_type(related_mat), requires_grad=False)

    def forward(self, rels, terms_L, terms_R):
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

    def forward_corrupt(self, rels, terms_L, terms_R):
        # Generate corrupt versions of the term vectors that are vaguely related
        # but not similar, and return the matching energy when corrupting the
        # left term or the right term. We want this energy to be low.
        rels_b_i = self.rel_vecs(rels)
        terms_b_L = self.term_vecs(terms_L)
        terms_b_R = self.term_vecs(terms_R)

        related_b_L = torch.matmul(terms_b_L, self.related_mat)
        dist_b_L = torch.sqrt(torch.sum((related_b_L - terms_b_L) ** 2, 1, keepdim=True))
        noise_b_L = autograd.Variable(self.float_type(np.random.normal(size=related_b_L.shape)), requires_grad=False)
        corrupt_b_L = related_b_L + noise_b_L * dist_b_L / (self.term_dim ** 0.5)

        related_b_R = torch.matmul(terms_b_R, self.related_mat)
        dist_b_R = torch.sqrt(torch.sum((related_b_R - terms_b_R) ** 2, 1, keepdim=True))
        noise_b_R = autograd.Variable(self.float_type(np.random.normal(size=related_b_R.shape)), requires_grad=False)
        corrupt_b_R = related_b_R + noise_b_R * dist_b_R / (self.term_dim ** 0.5)

        inter_b_i_L = self.assoc_tensor(corrupt_b_L, terms_b_R)
        inter_b_i_R = self.assoc_tensor(terms_b_L, corrupt_b_R)
        relmatch_b_i_L = inter_b_i_L * rels_b_i
        relmatch_b_i_R = inter_b_i_R * rels_b_i
        relmatch_2b_i = torch.cat((relmatch_b_i_L, relmatch_b_i_R))
        energy_2b = torch.sum(relmatch_2b_i, 1)
        return energy_2b * self.truth_multiplier + self.truth_offset

    def measure_bias(self):
        # Train our predictors to recognize gender distinctions in term vectors
        gendered_vecs = self.term_vecs(self.bias_indices['gendered']).detach()
        gendered_batch = self.gender_appropriateness(gendered_vecs)
        neutral_vecs = self.term_vecs(self.bias_indices['neutral']).detach()
        neutral_batch = -self.gender_appropriateness(neutral_vecs)
        female_vecs = self.term_vecs(self.bias_indices['female']).detach()
        female_batch = self.gender_bias(female_vecs)
        male_vecs = self.term_vecs(self.bias_indices['male']).detach()
        male_batch = -self.gender_bias(male_vecs)

        all_terms_gender = self.gender_bias(self.term_vecs.weight)
        all_terms_appropriateness = self.gender_appropriateness(self.term_vecs.weight)

        return (
            torch.cat((female_batch, male_batch)).float(),
            torch.cat((gendered_batch, neutral_batch)).float(),
            all_terms_gender.float(),
            all_terms_appropriateness.float()
        )

    def positive_negative_batch(self, edge_iterator):
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
        while True:
            yield self.positive_negative_batch(edge_iterator)

    def show_debug(self, batch, energy, positive):
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
    def load_model(filename):
        frame = load_hdf(INITIAL_VECS_FILENAME)
        model = SemanticMatchingModel(l2_normalize_rows(frame.astype(np.float32)))
        model.load_state_dict(torch.load(filename))
        return model

    def evaluate_conceptnet(self, cutoff_value=-1, output_filename=None):
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
        relative_loss_function = nn.MarginRankingLoss(margin=1)
        absolute_loss_function = nn.BCEWithLogitsLoss()
        one_side_loss_function = nn.MarginRankingLoss(margin=0)

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

            sem_loss = absolute_loss_function(pos_energy, true_target)
            for neg_index in range(NEG_SAMPLES):
                neg_energy_slice = neg_energy[neg_index::NEG_SAMPLES]
                sem_loss += relative_loss_function(pos_energy, neg_energy_slice, true_target)
                sem_loss += absolute_loss_function(neg_energy_slice, false_target)

            gender_predictions, approp_predictions, gender_vals, approp_vals = self.measure_bias()
            ones_like_gender = autograd.Variable(torch.ones_like(gender_predictions.data))
            ones_like_approp = autograd.Variable(torch.ones_like(approp_predictions.data))

            measurement_loss = absolute_loss_function(gender_predictions, ones_like_gender)
            measurement_loss += absolute_loss_function(approp_predictions, ones_like_approp)

            prejudice_loss = inappropriateness_loss(gender_vals, approp_vals)

            loss = sem_loss + measurement_loss * 10 + prejudice_loss
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
                print("%d steps, loss=%4.4f, sem=%4.4f, measure=%4.4f, prej=%4.4f" % (
                    steps, avg_loss, sem_loss, measurement_loss * 10, prejudice_loss
                ))
                losses.clear()
            if steps % 5000 == 0:
                torch.save(self.state_dict(), MODEL_FILENAME)
                print("saved")
        print()

    def export(self, dirname):
        path = pathlib.Path(dirname)
        term_mat = self.term_vecs.weight.data.float().cpu().numpy()
        term_frame = pd.DataFrame(term_mat, index=self.index)
        save_hdf(term_frame, str(path / "terms-similar.h5"))

        rel_mat = self.rel_vecs.weight.data.float().cpu().numpy()
        rel_frame = pd.DataFrame(rel_mat, index=RELATION_INDEX)

        save_hdf(rel_frame, str(path / "relations.h5"))
        assoc_t = self.assoc_tensor.weight.data.float().cpu().numpy()
        save_npy(assoc_t, str(path / "assoc.npy"))

        rel_vec = rel_mat[1]
        related_mat = np.einsum('i,ijk->jk', rel_vec, assoc_t)
        related_mat = (related_mat + related_mat.T) / 2
        related_terms = term_frame.dot(related_mat)
        save_hdf(related_terms, str(path / "terms-related.h5"))


    def ltvar(self, numbers):
        return autograd.Variable(self.int_type(numbers))


def get_model():
    if os.access(MODEL_FILENAME, os.F_OK):
        model = SemanticMatchingModel.load_model(MODEL_FILENAME)
    else:
        frame = load_hdf(get_data_filename(INITIAL_VECS_FILENAME))
        model = SemanticMatchingModel(l2_normalize_rows(frame.astype(np.float32)))
    return model


if __name__ == '__main__':
    model = get_model()
    model.train()
    # model.evaluate_conceptnet()
