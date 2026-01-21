import pandas as pd
import numpy as np
import textdistance
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


from collections import defaultdict
from typing import  Hashable, Set, FrozenSet, Union, Literal

import pandas as pd
import swifter
from abydos import phonetic


def compute_distance(args):
    """
    Compute a distance between two tokens/strings given their indices.

    Parameters
    ----------
    args : tuple
        (i, j, words, metric)
        - i, j: indices of the two words
        - words: list of strings (vocabulary)
        - metric: str, e.g., 'jw' or 'lev'

    Returns
    -------
    tuple
        (i, j, dist) where dist is a numeric distance in [0,1] for JW,
        or an integer distance for Levenshtein (depending on implementation).
    """
    i, j, words, metric = args

    w_i = words[i]
    w_j = words[j]
    # Select metric
    if metric == "jw":
        # Jaro–Winkler normalized_distance returns a value in [0,1]
        # (0 = identical, 1 = maximally different)
        dist = textdistance.jaro_winkler.normalized_distance(w_i, w_j)

    elif metric == "lev":
        # Levenshtein distance: integer number of edits
        # If you want normalized behavior, replace with:
        # textdistance.levenshtein.normalized_distance(...)
        dist = textdistance.levenshtein.distance(w_i, w_j)
    elif metric == "n_lev":
        dist = textdistance.levenshtein.normalized_distance(w_i, w_j)
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Supported: 'jw', 'lev', 'n_lev.")

    return (i, j, dist)


def distance_metrics_calculator(words, metric):
    """
    Build a full symmetric distance matrix between all pairs of strings in `words`
    using multiprocessing.

    Parameters
    ----------
    words : list[str]
        Vocabulary of unique tokens/words.
    metric : str
        Distance metric name: e.g., 'jw' (Jaro–Winkler normalized distance) or
        'lev' (Levenshtein edit distance).

    Returns
    -------
    pd.DataFrame
        Square (n x n) distance matrix with both index and columns = `words`.
    """
    n = len(words)

    # Generate index pairs for the upper triangle (including diagonal),
    # so we compute each pair only once, then mirror it for symmetry.
    index_pairs = [(i, j) for i in range(n) for j in range(i, n)]

    # Pack *all* worker inputs into a single iterable of tasks
    # (this is the correct pattern for Pool.imap_unordered).
    tasks = [(i, j, words, metric) for (i, j) in index_pairs]

    # Preallocate distance matrix
    dist_matrix = np.zeros((n, n), dtype=float)

    # Parallel computation
    with Pool(processes=cpu_count()) as pool:
        for idx_i, idx_j, dist in tqdm(
            pool.imap_unordered(compute_distance, tasks),
            total=len(tasks)
        ):
            dist_matrix[idx_i, idx_j] = dist
            dist_matrix[idx_j, idx_i] = dist  # Ensure symmetry

    # Return as DataFrame for convenience
    return pd.DataFrame(dist_matrix, index=words, columns=words)


def get_distance_frozensets(distance_df, max_distance):
    matched_df = distance_df <=max_distance
    matched_frozensets = set()
    for name in matched_df.columns:
        matched_names = matched_df[name][matched_df[name]].index
        for matched_name in matched_names:
            if name < matched_name:  # to avoid duplicates and self-matches
                matched_frozensets.add(frozenset([name, matched_name]))
    return matched_frozensets




def build_phonetic_matched_frozensets(
    words: Union[pd.Series, list[str]],
    metric: Literal["dm"] = "dm"
    ) -> Set[FrozenSet[Hashable]]:
    """
    Build a set of matched token pairs using a phonetic algorithm, returned as
    frozensets of indices (or labels), i.e. {frozenset({i, j}), ...}.

    Parameters
    ----------
    words:
        A list/Series of tokens (strings). If a pandas Series is given, its index is used
        as the identifier for returned pairs; otherwise integer positions are used.
    metric:
        Phonetic metric identifier. Currently supports:
        - "dm": Daitch–Mokotoff (Abydos phonetic.DaitchMokotoff)

    Returns
    -------
    set[frozenset[Hashable]]
        A set of undirected matched pairs. Each element is `frozenset({id_i, id_j})`
        where `id_i != id_j`.

        - If `words` is a Series: ids are `words.index` values.
        - If `words` is a list: ids are integer positions [0..n-1].`.
    """
    # Normalize input to a pandas Series so we can reliably keep stable identifiers.
    if isinstance(words, pd.Series):
        word_series = words.astype(str)
    else:
        word_series = pd.Series([str(w) for w in words])

    # -------------------------------------------------------------------------
    # 1) Choose phonetic encoder
    # -------------------------------------------------------------------------
    if metric == "daitch_mokotoff":
        encoder = phonetic.DaitchMokotoff()
    elif metric == "beider_morse_exact":
        encoder = phonetic.BeiderMorse(match_mode='exact')
        
    elif metric == "beider_morse_approx":
        encoder = phonetic.BeiderMorse(match_mode='approx')

    else:
        raise ValueError(f"Unsupported phonetic metric: {metric!r}")

    # -------------------------------------------------------------------------
    # 2) Encode each word -> set of phonetic codes
    # -------------------------------------------------------------------------
    # encoder.encode(word) returns a set of code strings in Abydos.
    codes_per_word = word_series.swifter.apply(encoder.encode)
    
    # if it is beidermorse, needs extra processing
    if metric.startswith("beider_morse"):
        codes_per_word = codes_per_word.map(lambda x: set(str.split(x)))

    # Ensure each value is a set (defensive; encoder already returns a set)
    codes_per_word = codes_per_word.apply(set)
    

    # -------------------------------------------------------------------------
    # 3) Build inverted index: code -> set(word_ids)
    # -------------------------------------------------------------------------
    variant_to_ids: dict[str, set[Hashable]] = defaultdict(set)
    for word_id, code_set in codes_per_word.items():
        for code in code_set:
            variant_to_ids[code].add(word_id)

    # -------------------------------------------------------------------------
    # 4) For each word_id, find other ids sharing at least one code
    # -------------------------------------------------------------------------
    def matching_ids(word_id: Hashable, code_set: set[str]) -> list[Hashable] | None:
        matched = set()
        for code in code_set:
            matched.update(variant_to_ids[code])
        matched.discard(word_id)
        return list(matched) if matched else None

    matches_series = pd.Series(
        {word_id: matching_ids(word_id, code_set) for word_id, code_set in codes_per_word.items()}
    )

    # -------------------------------------------------------------------------
    # 5) Convert to undirected frozenset pairs (id_i, id_j)
    # -------------------------------------------------------------------------
    # explode() gives rows: (word_id, matched_id)
    exploded = matches_series.explode().dropna().reset_index()
    exploded.columns = ["id_i", "id_j"]

    # Build undirected pairs
    matched_frozensets = set(
        frozenset({row["id_i"], row["id_j"]})
        for _, row in exploded.iterrows()
        if row["id_i"] != row["id_j"]
    )

    return matched_frozensets


def sorensen_dice_coverage(
    label_tokens,
    alias_tokens,
    matched_frozensets,
):
    """
    Compute a Dice–Sørensen *coverage* score between two token lists, using a
    precomputed set of allowable cross-token matches:

        DSC = 2 * |M| / (|A| + |B|)

    where:
    - A = tokens from the label 
    - B = tokens from the alias 
    - M = the set of matched token pairs chosen by a greedy matching procedure
          (each token can be used at most once)

    Dice–Sørensen ranges from 0 to 1:
    - 0.0 means no components match
    - 1.0 means perfect overlap (all components matched)

    Notes
    -----
    1) We treat a token as “matching” if either:
       - exact equality (token_a == token_b), OR
       - the pair is in `matched_frozensets` (e.g., JW/LEV/phonetic match list)
    2) We do a greedy 1-to-1 matching: once a token is matched, it is removed
       from further consideration. This mirrors your original implementation.

    Parameters
    ----------
    label_tokens : list[str]
        Tokenized components of the main label.
    alias_tokens : list[str]
        Tokenized components of the alias.
    matched_frozensets : set[frozenset[str]]
        Set of allowed token matches, represented as frozensets({token1, token2}).

    Returns
    -------
    float
        Dice–Sørensen coefficient computed on the matched components.
    """
    # Work on mutable copies (so we can remove matched tokens)
    remaining_label = list(label_tokens)
    remaining_alias = list(alias_tokens)

    matched_count = 0  # |M| in the Dice formula

    # Try to match each label token to at most one alias token
    for lt in list(label_tokens):
        for at in list(remaining_alias):
            # A match is either exact or allowed by the precomputed matcher
            if at == lt or frozenset({lt, at}) in matched_frozensets:
                # Remove both tokens to enforce 1-to-1 matching
                if lt in remaining_label:
                    remaining_label.remove(lt)
                if at in remaining_alias:
                    remaining_alias.remove(at)

                matched_count += 1
                break

    denom = len(label_tokens) + len(alias_tokens)
    if denom == 0:
        return 0.0

    # Dice–Sørensen coefficient: 2|M| / (|A| + |B|)
    return (2.0 * matched_count) / denom

def row_coverage_calculator(row, df_to_match, treshold, matched_frozensets, list_1_name, list_2_name):
    """
    Given one reference row (e.g., a label record), compare it against a candidate
    dataframe (e.g., alias records) by computing a Dice–Sørensen coverage score
    over words lists, then filter by a threshold.

    Conceptually, this does:
      1) Build a candidate set via a join on a shared blocking key (`key`).
      2) For each candidate pair, compute a coverage score between two token lists:
         - row[list_1_name]  (e.g., label_name_list)
         - row[list_2_name]  (e.g., alias_name_list)
         allowing approximate token matches via `matched_frozensets`.
      3) Keep only candidate pairs whose coverage >= `treshold`.

    Parameters
    ----------
    row : pd.Series
        One row from the “left” dataframe (often the label side). Must include
        the join column `key` and the list-valued column named by `list_1_name`.
    df_to_match : pd.DataFrame
        Candidate rows from the “right” dataframe (often the alias side). Must
        include `key` and the list-valued column named by `list_2_name`.
    treshold : float
        Minimum accepted coverage score. (used for reducing memory consumption)
    matched_frozensets : set[frozenset[str]]
        Set of allowed approximate token matches (e.g., from JW/LEV/DM/BM token
        pairing), represented as frozenset({token_a, token_b}).
    list_1_name : str
        Column name for the left token list, e.g. 'label_name_list'.
    list_2_name : str
        Column name for the right token list, e.g. 'alias_name_list'.

    Returns
    -------
    pd.DataFrame
        A dataframe containing only those joined candidate pairs whose computed
        `coverage_value` meets or exceeds the threshold.
    """
    # -------------------------------------------------------------------------
    # 1) Candidate generation (“blocking”):
    # Join this single row against all candidates that share the same `key`.
    # row.to_frame().T converts a Series into a one-row DataFrame so we can merge.
    # -------------------------------------------------------------------------
    cross = pd.merge(row.to_frame().T, df_to_match, on="key").drop("key", axis=1)
    # -------------------------------------------------------------------------
    # 2) Scoring:
    # Compute Dice–Sørensen coverage for each candidate pair.
    # -------------------------------------------------------------------------
    cross["coverage_value"] = cross.apply(
        lambda r: sorensen_dice_coverage(r[list_1_name], r[list_2_name], matched_frozensets),
        axis=1,
    )
    # -------------------------------------------------------------------------
    # 3) Filtering:
    # Keep only candidate pairs above the chosen coverage threshold.
    # -------------------------------------------------------------------------
    cross = cross[cross["coverage_value"] >= treshold]

    return cross
