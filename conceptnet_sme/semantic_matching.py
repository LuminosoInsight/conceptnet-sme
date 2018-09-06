import torch

# Set USE_MULTIPLE_GPUS = True to enable parallelization of training over
# multiple GPUs (if available; setting it True when there are no or only one
# GPU won't hurt, but it won't help either).

USE_MULTIPLE_GPUS = True

# Set NUM_BATCH_WORKERS > 0 to enable parallelization of generation of training
# batches over multiple CPU cores (by spawning that number of worker processes;
# 10 is a reasonable choice if you have multiple CPUs).

NUM_BATCH_WORKERS = 0
if NUM_BATCH_WORKERS > 0:
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from conceptnet5.vectors.formats import save_hdf, save_npy
from conceptnet5.formats.msgpack_stream import MsgpackStreamWriter
import numpy as np
import pandas as pd
import random
import pathlib
import os
import contextlib
import datetime
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
INITIAL_VECS_FILENAME = get_data_filename("vectors/numberbatch.h5")
#INITIAL_VECS_FILENAME = get_data_filename("vectors/mini.h5")
EDGES_FILENAME = get_data_filename("collated/sorted/edges-shuf.csv")
VALIDATION_FILENAME = get_data_filename("collated/sorted/edges-shuf-validation.csv")
MODEL_FILENAME = get_data_filename("sme/sme.model")
LOG_FILENAME = get_data_filename("sme/sme.log")
NEG_SAMPLES = 5
ADVERSARIAL_SAMPLES = 3
PREDICT_SHARDS = 100
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


class _MessageWriter:
    def __init__(self, filename=LOG_FILENAME):
        self.file = open(filename, 'at', encoding='utf-8')
        timestamp = datetime.datetime.utcnow().isoformat()
        print("Logging started at {}".format(timestamp), file=self.file)

    def __del__(self):
        self.file.close()

    def write(self, msg):
        print(msg, file=self.file)

_message_writer = None

def log_message(msg):
    global _message_writer
    print(msg)
    if _message_writer is not None:
        _message_writer.write(msg)


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
        log_message("{} {} sec.".format(caption, self.accumulated_time))
        if accumulated_time is not None:
            self.accumulated_time = accumulated_time
        return


class EdgeDataset(Dataset):
    """
    Wrapper class (around an iteration over ConceptNet edges) that enables us
    to use a torch DataLoader to parallelize generation of training batches.
    """

    def __init__(self, filename, model):
        """
        Construct an edge dataset from a filename (of a tab-separated
        ConceptNet edge file, which should be in random order), and a (pandas)
        index mapping (row) numbers to terms of the ConceptNet vocabulary.
        """
        super().__init__()
        self.filename = filename
        self.model = model
        self.index = model.index
        self.n_terms = len(self.index)
        self.edge_set = set()
        # Cache the edges in CPU memory as torch LongTensors,
        # skipping edges we don't intend to process.
        rel_indices = []
        left_indices = []
        right_indices = []
        weights = []

        for count, (rel, left, right, weight) in enumerate(self.iter_edges_once()):
            if count % 500000 == 0 and count > 0:
                log_message("Read {} edges.".format(count))
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

            self.edge_set.add((rel_idx, left_idx, right_idx))
            entailed_rels = ENTAILED_INDICES[RELATION_INDEX[rel_idx]]
            for entailed in entailed_rels:
                self.edge_set.add((entailed, left_idx, right_idx))
            if rel in SYMMETRIC_RELATIONS:
                self.edge_set.add((rel_idx, right_idx, left_idx))
                for entailed in entailed_rels:
                    self.edge_set.add((entailed, right_idx, left_idx))

            weights.append(weight)
        if len(rel_indices) < 1:
            log_message("No edges survived filtering; fitting is impossible!")
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
        Produce a positive example, followed by a number of generated examples that
        are presumed to be negative (but might accidentally be positive). Return these
        examples collated into tensors.
        """
        rel_idx = self.rel_indices[i_edge]
        orig_rel_idx = rel_idx
        left_idx = self.left_indices[i_edge]
        right_idx = self.right_indices[i_edge]
        weight = self.edge_weights[i_edge]

        examples = []

        rel = RELATION_INDEX[rel_idx]

        # Possibly swap the sides of a relation
        if coin_flip() and rel in SYMMETRIC_RELATIONS:
            left_idx, right_idx = right_idx, left_idx

        # Possibly replace a relation with a more general relation
        if coin_flip():
            rel_idx = random.choice(ENTAILED_INDICES[rel])
            rel = COMMON_RELATIONS[rel_idx]

        examples.append((int(rel_idx), int(left_idx), int(right_idx), weight))

        n_neg = NEG_SAMPLES + ADVERSARIAL_SAMPLES * 2

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

            examples.append((int(corrupt_rel_idx), int(corrupt_left_idx), int(corrupt_right_idx), weight / n_neg))

        best_terms_right, best_values_right = self.model.predict_terms(
            torch.LongTensor([rel_idx]),
            torch.LongTensor([left_idx]),
            PREDICT_SHARDS, random.randrange(PREDICT_SHARDS), forward=True, topk=ADVERSARIAL_SAMPLES
        )
        best_terms_left, best_values_left = self.model.predict_terms(
            torch.LongTensor([rel_idx]),
            torch.LongTensor([right_idx]),
            PREDICT_SHARDS, random.randrange(PREDICT_SHARDS), forward=False, topk=ADVERSARIAL_SAMPLES
        )

        for row in best_terms_left:
            for predicted_left in row:
                examples.append((int(orig_rel_idx), int(predicted_left), int(right_idx), weight / n_neg))
        for row in best_terms_right:
            for predicted_right in row:
                examples.append((int(orig_rel_idx), int(left_idx), int(predicted_right), weight / n_neg))

        targets = [example[:3] in self.edge_set for example in examples]
        assert examples[0][:3] in self.edge_set
        rels, lefts, rights, weights = zip(*examples)
        data = dict(
            rels=torch.LongTensor(rels),
            lefts=torch.LongTensor(lefts),
            rights=torch.LongTensor(rights),
            weights=torch.FloatTensor(weights),
            targets=torch.FloatTensor(targets)
        )

        lastk = ADVERSARIAL_SAMPLES * 2
        eval_batch = rels[-lastk:], lefts[-lastk:], rights[-lastk:], weights[-lastk:], targets[-lastk:]
        if random.random() < 0.01:
            self.model.show_debug(eval_batch)
        return data

    def collate_batch(self, batch):
        """
        Collates batches (as returned by a DataLoader that batches
        the outputs of calls to __getitem__) into tensors (as required
        by the train method of SemanticMatchingModel).
        """
        rels = torch.cat([x["rels"] for x in batch])
        lefts = torch.cat([x["lefts"] for x in batch])
        rights = torch.cat([x["rights"] for x in batch])
        weights = torch.cat([x["weights"] for x in batch])
        targets = torch.cat([x["targets"] for x in batch])
        return (rels, lefts, rights, weights, targets)


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
        Length makes no sense for a cycling sampler; it is effectively infinite.
        """
        raise NotImplementedError


class SemanticMatchingModel(nn.Module):
    """
    The PyTorch model for semantic matching energy over ConceptNet.
    """

    def __init__(self, frame, use_cuda=True, relation_dim=10):
        """
        Parameters:

        `frame`: a pandas DataFrame of pre-trained word embeddings over the
        vocabulary of ConceptNet. `conceptnet5.vectors.formats.load_hdf`
        can load these.

        `use_cuda`: whether to use GPU-accelerated PyTorch objects.

        `relation_dim`: the number of dimensions in the relation embeddings.
        Unlike SME as published, this can differ from the dimensionality of
        the term embeddings.
        """
        super().__init__()

        # Initialize term embeddings, including the index that converts terms
        # from strings to row numbers and the index converting relations to
        # row numbers in the relation vector embedding.
        log_message("Intializing term embeddings.")
        self.index = frame.index
        n_frame_terms, term_dim = frame.values.shape
        self.term_vecs = nn.Embedding(n_frame_terms, term_dim, sparse=True)
        self.term_vecs.weight.data.copy_(torch.from_numpy(frame.values))

        # Create a mapping from languages to index numbers
        self.index_by_language = {}
        for i, term in enumerate(frame.index):
            lang = get_uri_language(term)
            self.index_by_language.setdefault(lang, []).append(i)

        # The assoc_tensor is a (k_2 x k_1 x k_1) tensor that represents
        # interactions between the relations and the terms. k_1 is the
        # dimensionality of the term embeddings, and k_2 is the dimensionality
        # of the relation embeddings.
        #
        # We pass the dimensions in the order (k_1, k_1, k_2) because the
        # PyTorch convention is that inputs come before outputs, but the
        # resulting tensor is (k_2 x k_1 x k_1) because the math convention
        # is that outputs are on the left.
        log_message("Initializing association tensor and relation embedding.")
        self.assoc_tensor = nn.Bilinear(term_dim, term_dim, relation_dim, bias=True)
        self.rel_vecs = nn.Embedding(N_RELS, relation_dim, sparse=True)

        # Using CUDA to run on the GPU requires different data types
        log_message("Setting model device (cpu/gpu).")
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Create (and register) the identity tensor used to reset the synonym
        # relation.  (Register it so it will be included in the state dict.)
        log_message("Initializing identity slice.")
        self.register_buffer(
            "identity_slice",
            torch.tensor(np.eye(term_dim), dtype=torch.float32, device=self.device),
        )
        self.reset_synonym_relation()

        # Learnable priors for how confident to be in arbitrary statements.
        # These are used to convert the tensor products linearly into logits.
        # The initial values of these are set to numbers that appeared
        # reasonable based on a previous run.
        log_message("Initializing logit scale factors.")
        self.truth_multiplier = nn.Parameter(
            torch.tensor(5.0, dtype=torch.float32)
        )
        self.truth_offset = nn.Parameter(
            torch.tensor(-3.0, dtype=torch.float32)
        )

        # Make sure everything is on the right device.
        log_message("Moving model to selected device (cpu/gpu).")
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
        # Get relation vectors for the whole batch, with shape (b, k)
        rels_b_k = self.rel_vecs(rels)
        # Get left term vectors for the whole batch, with shape (b, L)
        terms_b_L = self.term_vecs(terms_L)
        # Get right term vectors for the whole batch, with shape (b, R)
        terms_b_R = self.term_vecs(terms_R)

        # Get the interaction of the terms in relation-embedding space, with
        # shape (b, k).
        inter_b_k = self.assoc_tensor(terms_b_L, terms_b_R)

        # Multiply our (b * k) term elementwise by rels_b_k. This indicates
        # how well the interaction between term L and term R matches each
        # component of the relation vector.
        relmatch_b_k = inter_b_k * rels_b_k

        # Add up the components for each item in the batch
        energy_b = torch.sum(relmatch_b_k, 1)

        return energy_b * self.truth_multiplier + self.truth_offset

    def score_edges(
            self,
            edges,
            batch_size=2,
            convert_logits_to_probas=False,
            device=None,
            use_multiple_gpus=False
    ):
        """
        Given a SemanticMatchingModel and an iteratable of edges (in the form of 
        tuples starting with a relation, a left endpoint, and a right endpoint, 
        where the endpoints have been run through uri_prefix), return an iterator 
        yielding the same tuples paired with the model's scores to the edge, 
        for all of the given edges whose endpoints are present in the model's 
        term index and whose relation is in the model's relation index.
        
        Scores are returned in units of logits, unless conversion to probabilities 
        is requested by setting the optional paramenter convert_logits_to_probas 
        to True.  If a device is specified (with the device parameter) then the 
        model will be moved to that device before scoring takes place; otherwise 
        the model will be moved to the device given by its device attribute (if not 
        already there).  After that, if use_multiple_gpus (default False) is 
        set to True, the model will be parallelized across all available gpu's 
        (using DataParalleizedModule).  Edges are scored in batches, with a default 
        size of 2 (which may be changed by setting the batch_size parameter).
        """
        # If a device is specified, move the model there before applying it.
        # Regardless, input for the model will be constructed in batches on the
        # cpu then moved to the device where the model is, and outputs will be
        # moved from there to the cpu.
        if device is None:
            device = self.device
        self.to(device)

        # Put the model on all available gpu's, if so requested.
        parallel_model = DataParallelizedModule(
            self,
            device,
            copy_fn=_model_copier,
            parallelize=use_multiple_gpus)

        # Put the model(s) in evaluation mode.
        for model_copy in parallel_model.children:
            model_copy.eval()

        # We'll batch the edges together before handing them to the model for
        # evaluation, for efficiency's sake.  As torch tensors are stored row-major,
        # and the model expects three tensors as input, we make batches of size
        # 3 x b, then split them by rows to feed them to the model.
        input_batch = torch.empty(
            [3, batch_size], dtype=torch.int64, requires_grad=False)
        input_batch.pin_memory() # speed up any transfer to gpu
        input_batch_on_device = torch.empty_like(input_batch, device=device)
        input_batch_edges = []

        position_in_batch = 0
        for edge in edges:
            # Translate the string fields of the edge to indices the model knows.
            rel = edge[0]
            left = edge[1]
            right = edge[2]
            try:
                rel_idx = RELATION_INDEX.get_loc(rel)
                left_idx = self.index.get_loc(left)
                right_idx = self.index.get_loc(right)
            except KeyError:
                continue # the model can't handle this edge

            # Insert the new edge's data into the growing batch.
            input_batch[0, position_in_batch] = rel_idx
            input_batch[1, position_in_batch] = left_idx
            input_batch[2, position_in_batch] = right_idx
            input_batch_edges.append(edge)

            # Keep track of where to insert, and when a batch is full.
            position_in_batch += 1
            if position_in_batch < batch_size:
                continue

            # Handle a full batch buffer.

            input_batch_on_device = input_batch.cuda(device, non_blocking=True)
            output_batch = parallel_model(
                input_batch_on_device[0], input_batch_on_device[1], input_batch_on_device[2]
            )
            output_batch.detach_() # requiring grad prevents calling numpy()
            if convert_logits_to_probas:
                output_batch.sigmoid_()
            output_batch = output_batch.cpu().numpy()
            for i_edge in range(batch_size):
                yield output_batch[i_edge], input_batch_edges[i_edge]
            position_in_batch = 0 # setup next batch
            input_batch_edges = []

        # Handle any final (shorter than normal) batch.
        
        if position_in_batch > 0:
            input_batch_on_device = input_batch.cuda(device, non_blocking=True)
            output_batch = parallel_model(
                input_batch_on_device[0, 0:position_in_batch],
                input_batch_on_device[1, 0:position_in_batch],
                input_batch_on_device[2, 0:position_in_batch]
            )
            output_batch.detach_() # requiring grad prevents calling numpy()
            if convert_logits_to_probas:
                output_batch.sigmoid_()
            output_batch = output_batch.cpu().numpy()
            for i_edge in range(position_in_batch):
                yield output_batch[i_edge], input_batch_edges[i_edge]

    def make_edge_score_file(self, input_filename, output_filename, **kwargs):
        """
        Read edges from the given input file, score them via score_edges, and 
        write 4-tuples (rel, left, right, score) as a msgpack stream to the 
        given output file.  Any additional kwargs given are passed on to 
        score_edges.
        """
        def edge_iterator(input_filename):
            with open(input_filename, "rt", encoding="utf-8") as fp:
                for line in fp:
                    fields = line.split('\t')
                    # fields[0] is the assertion
                    rel = fields[1]
                    left = uri_prefix(fields[2])
                    right = uri_prefix(fields[3])
                    yield rel, left, right
        writer = MsgpackStreamWriter(output_filename)
        for score, (rel, left, right) in self.score_edges(edge_iterator(input_filename), **kwargs):
            writer.write((rel, left, right, float(score)))
    
    def predict_wrapper(self, relname, termname, forward=True, topk=10):
        rel_idx = RELATION_INDEX.get_loc(relname)
        term_idx = self.index.get_loc(termname)
        best_terms, best_values = self.predict_terms(
            torch.LongTensor([rel_idx]),
            torch.LongTensor([term_idx]),
            1, 0, forward=forward, topk=topk
        )

        best_terms = best_terms[0].cpu().numpy()
        best_values = best_values[0].cpu().numpy()

        best_term_names = self.index[best_terms]
        return pd.DataFrame(best_values, index=best_term_names)

    def predict_terms(self, rels, terms, nshards=1, offset=0, forward=True, topk=1):
        with torch.no_grad():
            # Indices here use different letters to represent different dimensions, like
            # in Einstein notation.
            #
            # b: items in the batch
            # k: relation vectors
            # L and R: term vectors, distinguishing left terms from right terms
            # T: one of L or R, whichever one we have as input
            # m: terms in the subset of vocabulary we're using

            # Get a subset of terms that we'll try to predict. This has dimensions (m x R),
            # as it maps the vocabulary to term vectors.
            lang = get_uri_language(self.index[terms[0]])
            lang_indices = self.index_by_language[lang]
            candidate_indices = torch.LongTensor([
                idx for idx in lang_indices[offset::nshards]
                if idx != int(terms[0])
            ])

            candidate_terms_m_T = self.term_vecs.weight[candidate_indices]

            rels_b_k = self.rel_vecs(rels.to(self.device))
            assoc_k_L_R = self.assoc_tensor.weight
            terms_b_T = self.term_vecs(terms.to(self.device))

            # And now that we've got all these indices in Einstein notation, we can use
            # Einstein notation to describe exactly the operation that multiplies them,
            # giving us a (b x m) batch of term predictions.
            if forward:
                predictions_b_m = torch.einsum('bk,bl,klr,mr->bm', (rels_b_k, terms_b_T, assoc_k_L_R, candidate_terms_m_T))
            else:
                predictions_b_m = torch.einsum('bk,br,klr,ml->bm', (rels_b_k, terms_b_T, assoc_k_L_R, candidate_terms_m_T))

            best_values, best_indices = torch.topk(predictions_b_m, topk, dim=1)
            best_terms_reindexed = torch.take(candidate_indices, best_indices.cpu())
            return best_terms_reindexed, best_values

    def show_debug(self, batch):
        """
        On certain iterations, we show the training examples and what the model
        believed about them.
        """
        rel_indices, left_indices, right_indices, weights, targets = batch
        index_order = np.arange(len(weights))

        for i in index_order:
            rel = RELATION_INDEX[int(rel_indices[i])]
            left = self.index[int(left_indices[i])]
            right = self.index[int(right_indices[i])]
            if get_uri_language(left) == 'en':
                target = int(targets[i])
                print(f'{target}  {rel:<20} {left:<20} {right:<20}')

    @staticmethod
    def load_initial_frame():
        """
        Load the pre-computed embeddings that form our starting point.
        """
        total_accum = TimeAccumulator()
        incremental_accum = TimeAccumulator()
        log_message("Loading frame from file {}.".format(INITIAL_VECS_FILENAME))
        with stopwatch(total_accum), stopwatch(incremental_accum):
            frame = load_hdf(INITIAL_VECS_FILENAME)
        incremental_accum.print("Reading frame from file took", accumulated_time=0.0)
        log_message("Filtering terms in frame by language.")
        with stopwatch(total_accum), stopwatch(incremental_accum):
            labels = [
                label
                for label in frame.index
                if get_uri_language(label) in LANGUAGES_TO_USE
            ]
            frame = frame.loc[labels]
        incremental_accum.print("Filtering took", accumulated_time=0.0)
        log_message("Casting frame to float32.")
        with stopwatch(total_accum), stopwatch(incremental_accum):
            frame = frame.astype(np.float32)
        incremental_accum.print("Casting took", accumulated_time=0.0)
        total_accum.print("Loading frame (including reading, filtering, casting) took")
        return frame

    @staticmethod
    def load_model(filename, use_cuda=True):
        """
        Load the SME model from a file.  If use_cuda is True (the default), the 
        model will be placed on a gpu, otherwise on the cpu.
        """
        total_accum = TimeAccumulator()
        incremental_accum = TimeAccumulator()
        log_message("Loading model from file {}.".format(filename))
        with stopwatch(total_accum):
            frame = SemanticMatchingModel.load_initial_frame()
        log_message("Constructing model from frame.")
        with stopwatch(total_accum), stopwatch(incremental_accum):
            model = SemanticMatchingModel(frame, use_cuda=use_cuda)
        incremental_accum.print("Constructing model took", accumulated_time=0.0)
        log_message("Restoring model state from file.")
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

    def evaluate_edge(self, rel_idx, left_idx, right_idx):
        """
        Return the value assigned by the (presumably trained) model to the 
        edge with the given relation, left, and right term indices.
        """
        rel_tensor = torch.tensor([rel_idx], dtype=torch.int64, device=self.device)
        left_tensor = torch.tensor(
            [left_idx], dtype=torch.int64, device=self.device
        )
        right_tensor = torch.tensor(
            [right_idx], dtype=torch.int64, device=self.device
        )
        model_output = self(rel_tensor, left_tensor, right_tensor)
        value = model_output.item()
        return value

    def evaluate_conceptnet(self, input_file, cutoff_value=-1, output_filename=None):
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

        with open(input_file, "rt", encoding="utf-8") as fp:
            for line in fp:
                _assertion, rel, left, right, _rest = line.split("\t", 4)
                try:
                    rel_idx = RELATION_INDEX.get_loc(rel)
                    left_idx = self.index.get_loc(left)
                    right_idx = self.index.get_loc(right)
                except KeyError:
                    continue

                value = self.evaluate_edge(rel_idx, left_idx, right_idx)
                assertion = assertion_uri(rel, left, right)
                if value < cutoff_value:
                    print("%4.4f\t%s" % (value, assertion))
                if out is not None:
                    print("%4.4f\t%s" % (value, assertion), file=out)

    def evaluate_statistics(self, input_filename, output_filename=None,
                            n_edges=-1, n_perturbations_per_edge=99, random_state=None):
        """
        Read edges from the given input file, compute figures of merit based 
        on the (presumably trained) model's preditions compared to those edges, 
        or to a sample of them, print a summary of the results to stdout, and, 
        if an output filename is given, to that file (in json format).  
        
        The number of edges in the sample to use is given by the parameter 
        n_edges (default -1, meaning use all the edges, i.e. do not sample).
        
        For each edge in the sample, a set (of size n_perturbations_per_edge, 
        default 99) of perturbations of that edge is generated by randomly 
        altering either the left or right term (chosen with probability 1/2) 
        to another term from the model's index of terms (chosen uniformly at 
        random).  The quantile of the original edge's value as assigned by the 
        model among all the values assigned to perturbations not in the input 
        file is computed.  The statistics reported are the median and mean, 
        over all edges of the sample, of this quantile, and the fraction of 
        edges of the sample for which the quantile is at least 90% ("precision 
        at 10%").  Also, we compute the excess of the score value assigned by 
        the model to each of the orignal edges over the median score of its 
        perturbations, and report the 10th, 50th, and 90th percentiles (over 
        all the original edges) of this excess score.
        
        The random state used for sampling (and perturbation) may be specified 
        as the value of the parameter random_state; if unspecified a non-determnistic 
        seed will be used.
        """
        self.eval()
        if random_state is None:
            random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        print("Reading edge file.")
        edge_set = set()
        edge_list = []
        with open(input_filename, 'rt', encoding='utf-8') as fp:
            if (len(edge_list) + 1) % 500000 == 0:
                print("Reading edge {}".format(len(edge_list) + 1))
            for line in fp:
                _assertion, rel, left, right, _rest = line.split("\t", 4)
                try:
                    rel_idx = RELATION_INDEX.get_loc(rel)
                    left_idx = self.index.get_loc(left)
                    right_idx = self.index.get_loc(right)
                except KeyError:
                    continue
                edge_set.add((rel_idx, left_idx, right_idx))
                edge_list.append((rel_idx, left_idx, right_idx))

        if not 0 <= n_edges <= len(edge_set):
            n_edges = len(edges)

        print("Compiling statistics.")
        quantiles = []
        precision_at_10_count = 0
        n_totals = []
        excesses_over_median = []
        for i_edge in range(n_edges):
            if (i_edge + 1) % 500 == 0:
                print("Processing edge {} (of {}).".format(i_edge + 1, n_edges))
            rel_idx, left_idx, right_idx = edge_list[random_state.choice(len(edge_list))]
            value = self.evaluate_edge(rel_idx, left_idx, right_idx)
            n_bad_below = 0
            n_total = 1 # count the original edge
            new_values = []
            for i_new_edge in range(n_perturbations_per_edge):
                new_term_idx = random_state.choice(len(self.index))
                if random_state.choice(2) == 0:
                    new_edge = (rel_idx, new_term_idx, right_idx)
                else:
                    new_edge = (rel_idx, left_idx, new_term_idx)
                if new_edge in edge_set:
                    continue
                n_total += 1
                new_value = self.evaluate_edge(*new_edge)
                new_values.append(new_value)
                if new_value < value:
                    n_bad_below += 1
            quantile = float(n_bad_below) / float(n_total) 
            quantiles.append(quantile)
            if quantile >= 0.9:
                precision_at_10_count += 1
            n_totals.append(n_total)
            median_new_value = np.median(new_values)
            excess_over_median = value - median_new_value
            excesses_over_median.append(excess_over_median)

        median_quantile = np.median(quantiles)
        mean_quantile = np.mean(quantiles)
        precision_at_10 = precision_at_10_count / n_edges
        median_n_totals = np.median(n_totals)
        mean_n_totals = np.mean(n_totals)
        quantiles_of_excess_over_median = np.percentile(excesses_over_median, [10, 50, 90])

        print("Number of true edges examined is {}.".format(n_edges))
        print("Number of generated edges per true edge is {}.".format(
            n_perturbations_per_edge))
        print("Median total number of edges compared to compute quantiles is {}.".format(
            median_n_totals))
        print("Mean total number of edges compared to compute quantiles is {}.".format(
            mean_n_totals))
        print("Median quantile is {}.".format(median_quantile))
        print("Mean quantile is {}.".format(mean_quantile))
        print("Precision @ 10% is {}.".format(precision_at_10))
        print("10th percentile of excess of known positive edge score over median perturbed edge score is {}".format(
            quantiles_of_excess_over_median[0]))
        print("50th percentile of excess of known positive edge score over median perturbed edge score is {}".format(
            quantiles_of_excess_over_median[1]))
        print("90th percentile of excess of known positive edge score over median perturbed edge score is {}".format(
            quantiles_of_excess_over_median[2]))

        if output_filename is not None:
            import json
            with open(output_filename, 'wt', encoding='utf-8') as fp:
                json.dump(dict(median_quantile=median_quantile,
                               mean_quantile=mean_quantile,
                               n_edges=n_edges,
                               n_perturbations_per_edge=n_perturbations_per_edge,
                               precision_at_10=precision_at_10,
                               q10_of_excess=quantiles_of_excess_over_median[0],
                               q50_of_excess=quantiles_of_excess_over_median[1],
                               q90_of_excess=quantiles_of_excess_over_median[2]),
                          fp)

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
                    log_message(msg.format(name, difference))
                # Copy the param to the child (but first free up space).
                child_params[0].data = child_params[0].data.new_empty((0,))
                child_params[0].data = param_data_cpu.to(device)

def _model_copier(model, device):
    """
    Function to copy SemanticMatchingModels from one gpu to another, for use 
    with DataParallelizedModule.
    """
    model.cpu()
    frame = pd.DataFrame(model.term_vecs.weight.data.numpy(), index=model.index)
    new_model = SemanticMatchingModel(frame, use_cuda=False)
    new_model.load_state_dict(model.state_dict())
    new_model.device = device
    new_model.to(device)
    return new_model


def train_model(model, dataset, num_batch_workers=0, use_multiple_gpus=False,
                validation_dataset=None):
    """
    Incrementally train the model on the given dataset.

    If a positive number of batch worker processes is requested, generation
    of training data batches will be done in parallel across multiple CPU cores
    by spawning that number of child processes.

    If use of multiple GPUs is requested (and they are available), evaluation
    of batches will be parallelized across all available GPUs.

    If a validation dataset is given, every 1000 training batch iterations 
    the loss over 50 batches from this dataset (which should be distinct 
    from the training data) will be printed.

    As it is, this function will never return. It writes the results so far
    to `sme/sme.model` every 1000 iterations, and you can run it for as
    long as you want.
    """
    log_message("Starting model training.")
    model.train()

    log_message("Making parallelized model.")

    parallel_model = DataParallelizedModule(model, model.device, copy_fn=_model_copier, parallelize=use_multiple_gpus)

    log_message("Making optimizer.")
    optimizer = optim.SGD(parallel_model.parameters(), lr=0.1)

    # Note that you want drop_last=False with a CyclingSampler.
    log_message("Making data loader.")
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=num_batch_workers,
        collate_fn=dataset.collate_batch,
        sampler=CyclingSampler(dataset),
        pin_memory=True,
    )
    if validation_dataset is not None:
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            drop_last=False,
            num_workers=num_batch_workers,
            collate_fn=validation_dataset.collate_batch,
            sampler=CyclingSampler(validation_dataset),
            pin_memory=True
        )
        # Since we want to step through the (infinite, cyclic) sequence of
        # validation batches generated by this data loader, and take fixed-
        # size slices from that sequence, we convert the data loader (which
        # is iterable) into an iterator.  Note that (as the sequence is
        # infinite) this iterator will never raise StopIteration.
        validation_data_iterator = iter(validation_data_loader)

    losses = []
    steps = 0

    log_message("Entering training (batch) loop.")
    for batch in data_loader:
        if parallel_model.device != torch.device("cpu"):
            batch = tuple(
                x.cuda(device=parallel_model.device, non_blocking=True)
                for x in batch
            )

        parallel_model.zero_grad()
        rels, lefts, rights, weights, targets = batch

        energy = parallel_model(rels, lefts, rights)
        loss_function = nn.BCEWithLogitsLoss(weight=weights)
        loss = loss_function(energy, targets)
        loss.backward()

        parallel_model.broadcast_gradients()
        for model_copy in parallel_model.children:
            nn.utils.clip_grad_norm_(model_copy.parameters(), 1)

        optimizer.step()

        for model_copy in parallel_model.children:
            model_copy.reset_synonym_relation()

        losses.append(loss.data.cpu().item())
        steps += 1

        if steps in (1, 10, 20, 50, 100) or steps % 100 == 0:
            avg_loss = np.mean(losses)
            log_message(
                "%d steps, loss=%4.4f"
                % (steps, avg_loss)
            )
            losses.clear()

        if steps % 1000 == 0:
            incremental_accum = TimeAccumulator()
            log_message("Saving model.")
            with stopwatch(incremental_accum):
                model.cpu()
            incremental_accum.print("Moving model to cpu took", accumulated_time=0.0)
            with stopwatch(incremental_accum):
                torch.save(model.state_dict(), MODEL_FILENAME)
            incremental_accum.print("Saving to disk took", accumulated_time=0.0)

            with stopwatch(incremental_accum):
                model.to(model.device)
            incremental_accum.print("Moving model back to its device took", accumulated_time=0.0)
            log_message("saved")

            # With optim.SGD as the optimizer and when the model is
            # parallelized across multiple GPU's we see slight divergences
            # over time between the parallel models.  (This did not happen
            # with a custom optimizer that coalesces sparse gradients at
            # every training iteration.)  So we force agreement between the
            # children periodically.

            log_message("Synchronizing parallel model.")
            with stopwatch(incremental_accum):
                parallel_model.synchronize_children()
            incremental_accum.print("Synchonizing took", accumulated_time=0.0)

            # If a validation dataset was given, show the loss over (some of)
            # the validation data.
            if validation_dataset is not None:
                log_message("Evaluating loss over validation data.")
                model.eval()  # turn off training mode temporarily
                n_validation_batches = 50
                validation_loss = torch.tensor(0, dtype=torch.float32,
                                               device=parallel_model.device)
                
                # Take the specified number of batches from the validation
                # dataset via the data (loader) iterator.
                for _ in range(n_validation_batches):
                    batch = next(validation_data_iterator)
                    if parallel_model.device != torch.device("cpu"):
                        batch = tuple(
                            x.cuda(device=parallel_model.device, non_blocking=True)
                            for x in batch
                        )

                    rels, lefts, rights, weights, targets = batch

                    energy = parallel_model(rels, lefts, rights)
                    loss_function = nn.BCEWithLogitsLoss(weight=weights)
                    loss = loss_function(energy, targets)
                    validation_loss += loss
                
                validation_loss /= n_validation_batches  # mean over batches
                validation_loss = validation_loss.data.cpu().item()
                log_message("Validation loss (mean over {} batches) is {}.".format(
                    n_validation_batches,
                    validation_loss
                ))
                model.train()  # reset to train mode
                
    log_message()


def get_model():
    """
    Instantiate a model, either by loading it from the saved checkpoint, or
    by creating it from scratch if nothing is there.
    """
    if os.access(MODEL_FILENAME, os.F_OK):
        log_message("Loading previously saved model.")
        model = SemanticMatchingModel.load_model(MODEL_FILENAME)
    else:
        log_message("Creating a new model.")
        total_accum = TimeAccumulator()
        incremental_accum = TimeAccumulator()
        with stopwatch(total_accum):
            frame = SemanticMatchingModel.load_initial_frame()
        log_message("Normalizing initial frame.")
        with stopwatch(total_accum), stopwatch(incremental_accum):
            frame = l2_normalize_rows(frame)
        incremental_accum.print('Normalizing frame took', accumulated_time=0.0)
        log_message('Constructing model from frame.')
        with stopwatch(total_accum), stopwatch(incremental_accum):
            model = SemanticMatchingModel(frame)
        incremental_accum.print("Construction took", accumulated_time=0.0)
        total_accum.print('Total model creation time:')
    return model


if __name__ == "__main__":
    _message_writer = _MessageWriter()
    log_message("Starting semantic matching.")
    model = get_model()
    log_message("Initializing edge dataset ....")
    dataset_accumulator = TimeAccumulator()
    with stopwatch(dataset_accumulator):
        dataset = EdgeDataset(EDGES_FILENAME, model)
    dataset_accumulator.print("Edge dataset initialization took",
                              accumulated_time=0.0)
    log_message("Edge dataset contains {} edges.".format(len(dataset)))
    validation_dataset = None
    if os.path.isfile(VALIDATION_FILENAME):
        log_message("Initializing validation dataset ....")
        with stopwatch(dataset_accumulator):
            validation_dataset = EdgeDataset(VALIDATION_FILENAME, model)
        dataset_accumulator.print("Validation dataset initialization took")
        log_message("Validation dataset contains {} edges.".format(len(validation_dataset)))
    train_model(model, dataset, num_batch_workers=NUM_BATCH_WORKERS, use_multiple_gpus=USE_MULTIPLE_GPUS, validation_dataset=validation_dataset)
    # model.evaluate_conceptnet(EDGES_FILENAME)
    del _message_writer
