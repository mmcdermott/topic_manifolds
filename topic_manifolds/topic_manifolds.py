"""
these classes and utility produce manifolds corresponding to topic-labeled datasets.
"""

import copy, math, time, numpy as np
from collections import Counter, defaultdict, Sequence
from tqdm.auto import tqdm
from typing import Union, Any, List
from mixins.mixins import *

from .simplicial_manfiolds import *

def nCr(n, r): return int(math.factorial(n)/ (math.factorial(r) * math.factorial(n - r)))

class TopicCombinationSet(TimeableMixin, SwapcacheableMixin, TQDMableMixin, SeedableMixin):
    MAXIMAL = 'maximal'
    _TOPIC_ARGSORT = 'topic_argsort'
    _SIMPLEX_CNT = 'simplex_cnt'
    _GET_MAXIMAL_CLIQUE = 'get_maximal_clique'
    _NORMALIZATION_TIME = 'normalization'

    def __init__(
        self,
        probabilities: np.ndarray,
        num_settings_to_cache: int  = 5,
        cliques_to_check:      int  = 250,
        do_verbose:            bool = False,
        **mixin_kwargs,
    ):
        self.do_verbose = do_verbose
        self.probabilities = probabilities
        self.N_examples, self.N_topics = self.probabilities.shape
        self.cliques_to_check = cliques_to_check

        self._register_start(self._TOPIC_ARGSORT)
        # We need to negate it because we want to get the $k$-largest, not the $k$-smallest.
        self.topic_orderings = (-probabilities).argsort(axis=-1, )
        self._register_end(self._TOPIC_ARGSORT)

        self._cache_size = num_settings_to_cache

        super().__init__(**mixin_kwargs)

    def swap_to_params(self, new_params: dict) -> None: self._swap_to_key(key=new_params)

    def get_valid_simplices(self):
        attrs = (
            'valid_simplices', 'top_k', 'top_probs', 'valid_idxs', 'base_ts', 'containing_maximal_cliques'
        )
        if all(hasattr(self, attr) for attr in attrs): return
        elif self.do_verbose: print("Calculating valid simplices at parameter settings...")

        k = self.simplex_dim + 1

        top_k = self.topic_orderings[:, :k]
        top_probs = self.probabilities[np.arange(self.N_examples)[:, None], top_k]
        obs_probability_mass = top_probs.sum(axis=1)

        valid_mask = (obs_probability_mass > self.probability_thresh)

        valid_idxs      = np.arange(self.N_examples)[valid_mask]
        valid_top_k     = top_k[valid_mask]
        valid_top_probs = top_probs[valid_mask]

        N_valid_examples = valid_top_probs.shape[0]

        if self.do_verbose:
            print(
                f"Dropping {self.N_examples - N_valid_examples} examples as they lack sufficient "
                f"probability mass in their top-{k}.\n"
            )

        self._register_start(self._SIMPLEX_CNT)
        topic_simplices_cnts = Counter(frozenset(t) for t in valid_top_k)
        valid_simplices = set(
            t for t, cnt in topic_simplices_cnts.items() if cnt >= self.min_examples_per_simplex
        )

        sufficiently_dense_mask = np.array([frozenset(row) in valid_simplices for row in valid_top_k])

        valid_top_k     = valid_top_k[sufficiently_dense_mask]
        valid_top_probs = valid_top_probs[sufficiently_dense_mask]
        valid_idxs      = valid_idxs[sufficiently_dense_mask]

        self._register_end(self._SIMPLEX_CNT)

        if self.do_verbose:
            N_valid_simplices = len(valid_simplices)
            N_total_simplices = nCr(self.N_topics, k)
            print(
                f"Observe {N_valid_simplices} simplices (of {N_total_simplices} total possible) "
                f"in total across {self.N_examples} examples.\n"
            )

        self._update_current_swapcache_key(values_dict=dict(
            base_ts                    = set(itertools.chain.from_iterable(valid_simplices)),
            valid_simplices            = valid_simplices,
            valid_top_k                = valid_top_k,
            valid_top_probs            = valid_top_probs,
            valid_idxs                 = valid_idxs,
            containing_maximal_cliques = {},
        ))

    def _expand(
        self,
        topic_set: frozenset,
    ):
        ts = []
        for t in self.base_ts - topic_set:
            can_include = True
            for new_ts in itertools.combinations(topic_set, self.simplex_dim):
                if not frozenset((t,) + new_ts) in self.valid_simplices:
                    can_include = False
                    break

            if can_include: ts.append(t)

        return set(ts)

    def maximally_expand(
        self,
        topic_set: Sequence,
        depth:     int = 0,
    ):
        if not isinstance(topic_set, frozenset): topic_set = frozenset(topic_set)

        if topic_set in self.containing_maximal_cliques:
            return self.containing_maximal_cliques[topic_set]

        ts = self._expand(topic_set)
        if len(ts) == 0: return topic_set

        max_opt = topic_set
        ts_rng = ts if (depth > 2 or len(ts) < 10) else self._tqdm(ts, leave=False, desc="Expanding")
        for t in ts_rng:
            query_topic_set = frozenset([t, *topic_set])
            new_topic_set = self.maximally_expand(query_topic_set, depth=depth+1)

            if len(new_topic_set) > len(max_opt): max_opt = new_topic_set

        self.containing_maximal_cliques[topic_set] = max_opt
        return max_opt

    def get_maximal_clique(
        self, local_budget: Optional[int] = None, seed: Optional[int] = None
    ) -> List[frozenset]:
        self._seed(seed, "Find Maximal Clique")
        k = self.simplex_dim + 1
        budget = local_budget if local_budget is not None else self.cliques_to_check

        self._register_start(self._GET_MAXIMAL_CLIQUE)

        if self.containing_maximal_cliques:
            maximal_clique = max(self.containing_maximal_cliques.values(), key=len)
        else:
            random_simplex = random.choice(list(self.valid_simplices))
            maximal_clique = random_simplex

        options            = copy.deepcopy(list(self.valid_simplices))

        current_sample     = maximal_clique
        M                  = len(current_sample)
        current_subcliques = [list(itertools.combinations(current_sample, s)) for s in range(2, M-1)]

        samples_rng  = self._tqdm(np.arange(budget), desc=f"Maximal clique: {len(maximal_clique)} so far")
        maximal_cliques_by_sample = []
        for sample in samples_rng:
            do_random_subclique = random.choice([True, False])
            if not do_random_subclique:
                need_new_seed_attempts = 0
                new_seed = None

                while new_seed is None and need_new_seed_attempts < 10:
                    new_subclique_len = random.choice(range(2, M-1))
                    len_idx = new_subclique_len - 2

                    num_subcliques = len(current_subcliques[len_idx])
                    if num_subcliques == 0:
                        break
                    else:
                        new_subclique_idx = random.choice(range(num_subcliques))
                        new_seed = current_subcliques[len_idx].pop(new_subclique_idx)

                        # TODO(mmd): This actually breaks the MCMC algorithm a bit, by making the distribution
                        # depend on history.
                        if new_seed in self.containing_maximal_cliques: new_seed = None

                    need_new_seed_attempts += 1

                if new_seed is None: do_random_subclique = True

            # We do this in such a weird way as we may set it to True in the first condition above.
            if do_random_subclique:
                new_seed_idx = random.choice(np.arange(len(options)))
                new_seed = options.pop(new_seed_idx)

            max_expansion = self.maximally_expand(new_seed)

            u = random.random()
            thresh = (len(max_expansion) / len(current_sample)) ** 0.5

            if u < thresh:
                current_sample     = max_expansion
                M                  = len(current_sample)
                current_subcliques = [list(itertools.combinations(current_sample, s)) for s in range(2, M-1)]

            if len(max_expansion) > len(maximal_clique):
                maximal_clique = max_expansion
                samples_rng.set_description(f"Maximal clique: {len(maximal_clique)} so far")

            maximal_cliques_by_sample.append(maximal_clique)

        self._update_current_swapcache_key(values_dict=dict(maximal_clique = maximal_clique))
        self._register_end(self._GET_MAXIMAL_CLIQUE)

        return maximal_cliques_by_sample

    def filter_to_maximal_clique(self) -> None:
        k = self.simplex_dim + 1
        maximal_clique_valid_simplices = {
            t_set for t_set in self.valid_simplices if t_set.issubset(self.maximal_clique)
        }
        assert len(maximal_clique_valid_simplices) == nCr(len(self.maximal_clique), k)

        simplex_valid_mask = np.array(
            [frozenset(row) in maximal_clique_valid_simplices for row in self.valid_top_k]
        )

        maximal_clique_valid_top_k     = self.valid_top_k[simplex_valid_mask]
        maximal_clique_valid_top_probs = self.valid_top_probs[simplex_valid_mask]
        maximal_clique_valid_idxs      = self.valid_idxs[simplex_valid_mask]

        if self.do_verbose:
            print(
                "After filtering out simplices that are not universally compatible, we have "
                f"{len(maximal_clique_valid_simplices)}/{len(maximal_clique_valid_top_probs)} "
                "simplices / examples, respectively."
            )

        self._update_current_swapcache_key(values_dict=dict(
            maximal_clique_valid_simplices   = maximal_clique_valid_simplices,
            maximal_clique_valid_top_k       = maximal_clique_valid_top_k,
            maximal_clique_valid_idxs        = maximal_clique_valid_idxs,
            maximal_clique_valid_top_probs   = maximal_clique_valid_top_probs,
        ))

    def get_normalized_probs_and_entropy(self) -> None:
        self._register_start(self._NORMALIZATION_TIME)
        normalized = np.divide(
            self.maximal_clique_valid_top_probs,
            self.maximal_clique_valid_top_probs.sum(axis=1)[:, np.newaxis]
        )
        entropy    = -(normalized * np.log(normalized)).sum(axis=1)
        entropy_per_simplex = defaultdict(list)
        for i, (e, ts) in enumerate(zip(entropy, self.maximal_clique_valid_top_k)):
            entropy_per_simplex[frozenset(ts)].append(e)

        agg_entropy_per_simplex = {
            k: (np.min(es), np.max(es), np.histogram(es)) for k, es in entropy_per_simplex.items()
        }

        maximal_clique_valid_normalized_probs = normalized
        maximal_clique_valid_entropies = entropy
        maximal_clique_valid_entropy_per_simplex = entropy_per_simplex
        maximal_clique_valid_agg_entropy = agg_entropy_per_simplex

        self._register_end(self._NORMALIZATION_TIME)

        self._update_current_swapcache_key(values_dict=dict(
            maximal_clique_valid_normalized_probs    = maximal_clique_valid_normalized_probs,
            maximal_clique_valid_entropies           = maximal_clique_valid_entropies,
            maximal_clique_valid_entropy_per_simplex = maximal_clique_valid_entropy_per_simplex,
            maximal_clique_valid_agg_entropy         = maximal_clique_valid_agg_entropy,
        ))

    def get_optimal_topic_set(
        self,
        simplex_dim:              int               = 2,
        min_examples_per_simplex: int               = 25,
        probability_thresh:       Union[str, float] = MAXIMAL,
        local_budget:             Optional[int]     = None,
        seed:                     Optional[int]     = None,
    ) -> None:
        assert isinstance(simplex_dim, int) and simplex_dim > 1
        assert isinstance(min_examples_per_simplex, int) and min_examples_per_simplex > 0
        self._seed(seed, "Get Optimal Topic Set")

        if probability_thresh == self.MAXIMAL: probability_thresh = simplex_dim / (simplex_dim + 1)

        _cache_key = dict(
            simplex_dim              = simplex_dim,
            min_examples_per_simplex = min_examples_per_simplex,
            probability_thresh       = probability_thresh,
        )
        # We cache both the params and (eventually) derived values.
        self._update_swapcache_key_and_swap(key=_cache_key, values_dict=_cache_key)

        self.get_valid_simplices()
        self.get_maximal_clique(local_budget)
        self.filter_to_maximal_clique()
        self.get_normalized_probs_and_entropy()

class TopicSimplicialManifold(TimeableMixin, SeedableMixin, SaveableMixin, TQDMableMixin):
    _BUILD_MANIFOLD = 'build_manifold'
    _DEL_BEFORE_SAVING_ATTRS = ('manifold',)

    @classmethod
    def build_from_text(cls, X: np.ndarray):
        raise NotImplementedError
        probabilities = None
        return cls(probabilities)

    @classmethod
    def build_from_images(cls, X: np.ndarray):
        raise NotImplementedError
        probabilities = None
        return cls(probabilities)

    def __init__(
        self,
        dataset:                 np.ndarray,
        topic_combinations:      TopicCombinationSet,
        manifold_kwargs:         dict,
        num_samples_per_simplex: int              = 25,
        subsample_data:          Union[int, None] = None,
        seed:                    Optional[int]    = None,
        do_get_maximal_clique:   bool             = True,
        do_verbose:              bool             = False,
    ):
        assert dataset.shape[0] == topic_combinations.N_examples, (
            f"Dataset size mismatch! dataset.shape = {dataset.shape}, topic_combinations.N_examples = "
            f"{topic_combinations.N_examples}."
        )

        self._seed(seed, "Init")

        self.dataset            = dataset
        self.topic_combinations = topic_combinations
        self.manifold_kwargs    = manifold_kwargs
        self.num_samples_per_simplex = num_samples_per_simplex
        self.do_verbose         = do_verbose
        self.subsample_data     = subsample_data

        self.N_examples         = dataset.shape[0]

        self._build_manifold()

        if do_get_maximal_clique:
            self.topic_combinations.get_optimal_topic_set(
                simplex_dim              = self.manifold.d,
                num_examples_per_simplex = self.num_examples_per_simplex,
            )

        self.generate()

    def _post_load(self, kwargs: dict):
        if 'manifold' not in kwargs: self._buil_manifold()

    def _build_manifold(self):
        self._register_start(self._BUILD_MANIFOLD)
        self.manifold = LabeledSimplicialManifold(**self.manifold_kwargs)

        assert len(self.manifold.vocab) < len(self.topic_combinations.maximal_clique), (
            f"Can't produce a manifold with {len(self.manifold.vocab)} simplices using a topic clique of "
            f"size {len(self.topic_combinations.maximal_clique)}"
        )

        self._register_end(self._BUILD_MANIFOLD)

    def _set_topics_internal(self, seed=None):
        self._seed(seed, "Assign topics to simplex vertices")

        topic_clique = copy.deepcopy(list(self.topic_combinations.maximal_clique))
        random.shuffle(topic_clique)

        self.simplex_vertices_to_topics = {mv: t for mv, t in zip(self.manifold.vocab, topic_clique)}
        self.topics_to_simplex_vertices = {t: mv for mv, t in self.simplex_vertices_to_topics.items()}
        self.valid_topics = set(self.simplex_vertices_to_topics.values())

        self.valid_topic_simplices = set(
            frozenset(self.simplex_vertices_to_topics[v] for v in s) for s in self.manifold.simplices
        )
        self.example_on_manifold_idx = np.array([
            frozenset(t_row) in set(self.valid_topic_simplices) for t_row in \
                self.topic_combinations.maximal_clique_valid_top_k
        ])

        for attr in ('manifold_idxs', 'manifold_top_k', 'manifold_normalized_probs', 'manifold_entropies'):
            topic_comb_attr = getattr(
                self.topic_combinations, attr.replace('manifold', 'maximal_clique_valid')
            )
            try:
                new_val = topic_comb_attr[self.example_on_manifold_idx]
                setattr(self, attr, new_val)
            except:
                print(attr)
                print(self.example_on_manifold_idx.shape, self.example_on_manifold_idx.dtype)
                print(topic_comb_attr.shape, topic_comb_attr.dtype)
                raise

        self.manifold_agg_entropy = self.topic_combinations.maximal_clique_valid_agg_entropy

        self.N_on_manifold = len(self.manifold_idxs)
        self.manifold_topic_labels = self.manifold_top_k[
            np.arange(self.N_on_manifold), self.manifold_normalized_probs.argmax(axis=1)
        ]

        recreated_topics = self.topic_combinations.probabilities[self.manifold_idxs].argmax(axis=1)
        assert (self.manifold_topic_labels == recreated_topics).all(), "Topic check failed!"

        return self._assign_examples_to_simplices()

    def _assign_examples_to_simplices(self, seed=None):
        self._seed(seed, "Generate topic assignment")

        idxs = np.arange(self.N_examples)
        if self.subsample_data is not None and self.subsample_data > 0: idxs = idxs[:self.subsample_data]

        simplices_rng = self._tqdm(
            copy.deepcopy(self.manifold.simplices), desc='Mapping Simplices', leave=False
        )

        self.in_dataset = defaultdict(list)

        for simplex in simplices_rng:
            simplex_key = frozenset(simplex)
            topic_simplex = frozenset([self.simplex_vertices_to_topics[v] for v in simplex])

            mask = np.array([frozenset(t_row) == topic_simplex for t_row in self.manifold_top_k])
            on_simplex = {}
            for attr in (
                'idxs', 'top_k', 'normalized_probs', 'entropies', 'topic_labels'
            ):
                on_simplex[attr] = getattr(self, f"manifold_{attr}")[mask]

            on_simplex['agg_entropy'] = self.manifold_agg_entropy[topic_simplex]

            on_simplex['N_examples'] = len(on_simplex['idxs'])
            on_simplex['local_coordinates'] = np.array([
                {self.topics_to_simplex_vertices[t]: p for t, p in zip(t_row, p_row)}
                    for t_row, p_row in zip(on_simplex['top_k'], on_simplex['normalized_probs'])
            ])

            hist_counts, hist_bucket_endpoints = on_simplex['agg_entropy'][2]
            n_buckets = len(hist_bucket_endpoints) - 1

            sampling_P = []
            for e in on_simplex['entropies']:
                try:
                    bucket_idx = next(i for i in range(n_buckets) if hist_bucket_endpoints[i] > e)
                except StopIteration:
                    bucket_idx = n_buckets - 1

                sample_p = 1/max(1, hist_counts[bucket_idx])
                assert not np.isnan(sample_p) and sample_p > 0
                sampling_P.append(sample_p)

            sampling_P = np.array(sampling_P)
            sampling_P /= sampling_P.sum()

            num_samples = min(on_simplex['N_examples'], self.num_samples_per_simplex)
            sample_idxs = np.random.choice(
                np.arange(on_simplex['N_examples']), num_samples, replace = False, p = sampling_P,
            )

            for attr in ('idxs', 'topic_labels', 'local_coordinates'):
                self.in_dataset[attr].extend(on_simplex[attr][sample_idxs])

    def _gen_geodesic_distances(self) -> None:
        self.in_dataset['pairwise_geodesic_distances'] = self.manifold.efficient_pairwise_distances(
            self.in_dataset['local_coordinates']
        )

    def _gen_euclidean_distances(self, noise_rate: float = 0) -> None:
        self.in_dataset['pairwise_euclidean_distances'] = (self.manifold.pairwise_embedded_distances(
            self.in_dataset['local_coordinates'], noise_rate
        ), noise_rate)

    def generate(
        self,
        num_samples_per_simplex:  Optional[int]  = None,
        subsample_data:           Optional[int]  = None,
        re_gen_max_clique:        bool           = False,
        max_clique_re_gen_budget: Optional[int]  = None,
        seed:                     Optional[int]  = None,
    ) -> None:
        self._seed(seed, "Generate")
        if num_samples_per_simplex is not None: self.num_samples_per_simplex = num_samples_per_simplex
        if subsample_data is not None: self.subsample_data = subsample_data

        if re_gen_max_clique:
            self.topic_combinations.get_optimal_topic_set(
                simplex_dim              = self.manifold.d,
                num_examples_per_simplex = num_examples_per_simplex,
                local_budget             = max_clique_re_gen_budget
            )

        self._set_topics_internal()
        if self.do_verbose: print("finished setting topics. Assigning examples.")

        self._assign_examples_to_simplices()

    def get_geodesic_distances(self):
        if 'pairwise_geodesic_distances' not in self.in_dataset: self._gen_geodesic_distances()
        return self.in_dataset['pairwise_geodesic_distances']

    def get_euclidean_distances(self, noise_rate: float = 0):
        if (
            'pairwise_euclidean_distances' not in self.in_dataset or 
            self.in_dataset['pairwise_euclidean_distances'][1] != noise_rate
        ):
            self._gen_euclidean_distances(noise_rate)

        return self.in_dataset['pairwise_euclidean_distances'][0]
