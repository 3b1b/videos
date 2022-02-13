from manim_imports_ext import *
from tqdm import tqdm as ProgressDisplay
from scipy.stats import entropy


MISS = np.uint8(0)
MISPLACED = np.uint8(1)
EXACT = np.uint8(2)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)
SHORT_WORD_LIST_FILE = os.path.join(DATA_DIR, "possible_words.txt")
LONG_WORD_LIST_FILE = os.path.join(DATA_DIR, "allowed_words.txt")
WORD_FREQ_FILE = os.path.join(DATA_DIR, "wordle_words_freqs_full.txt")
WORD_FREQ_MAP_FILE = os.path.join(DATA_DIR, "freq_map.json")
SECOND_GUESS_MAP_FILE = os.path.join(DATA_DIR, "second_guess_map.json")
PATTERN_MATRIX_FILE = os.path.join(DATA_DIR, "pattern_matrix.npy")
ENT_SCORE_PAIRS_FILE = os.path.join(DATA_DIR, "ent_score_pairs.json")

# To store the large grid of patterns at run time
PATTERN_GRID_DATA = dict()


def safe_log2(x):
    return math.log2(x) if x > 0 else 0


# Reading from files

def get_word_list(short=False):
    result = []
    file = SHORT_WORD_LIST_FILE if short else LONG_WORD_LIST_FILE
    with open(file) as fp:
        result.extend([word.strip() for word in fp.readlines()])
    return result


def get_word_frequencies(regenerate=False):
    if os.path.exists(WORD_FREQ_MAP_FILE) or regenerate:
        with open(WORD_FREQ_MAP_FILE) as fp:
            result = json.load(fp)
        return result
    # Otherwise, regenerate
    freq_map = dict()
    with open(WORD_FREQ_FILE) as fp:
        for line in fp.readlines():
            pieces = line.split(' ')
            word = pieces[0]
            freqs = [
                float(piece.strip())
                for piece in pieces[1:]
            ]
            freq_map[word] = np.mean(freqs[-5:])
    with open(WORD_FREQ_MAP_FILE, 'w') as fp:
        json.dump(freq_map, fp)
    return freq_map


def get_frequency_based_priors(n_common=3000, width_under_sigmoid=10):
    """
    We know that that list of wordle answers was curated by some human
    based on whether they're sufficiently common. This function aims
    to associate each word with the likelihood that it would actually
    be selected for the final answer.

    Sort the words by frequency, then apply a sigmoid along it.
    """
    freq_map = get_word_frequencies()
    words = np.array(list(freq_map.keys()))
    freqs = np.array([freq_map[w] for w in words])
    arg_sort = freqs.argsort()
    sorted_words = words[arg_sort]

    # We want to imagine taking this sorted list, and putting it on a number
    # line so that it's length is 10, situating it so that the n_common most common
    # words are positive, then applying a sigmoid
    x_width = width_under_sigmoid
    c = x_width * (-0.5 + n_common / len(words))
    xs = np.linspace(c - x_width / 2, c + x_width / 2, len(words))
    priors = dict()
    for word, x in zip(sorted_words, xs):
        priors[word] = sigmoid(x)
    return priors


def get_true_wordle_prior():
    words = get_word_list()
    short_words = get_word_list(short=True)
    return dict(
        (w, int(w in short_words))
        for w in words
    )


# Generating color patterns between strings, etc.


def words_to_int_arrays(words):
    return np.array([[ord(c)for c in w] for w in words], dtype=np.uint8)


def generate_pattern_matrix(words1, words2):
    """
    A pattern for two words represents the wordle-similarity
    pattern (grey -> 0, yellow -> 1, green -> 2) but as an integer
    between 0 and 3^5. Reading this integer in ternary gives the
    associated pattern.

    This function computes the pairwise patterns between two lists
    of words, returning the result as a grid of hash values. Since
    this can be time-consuming, many operations that can be are vectorized
    (perhaps at the expense of easier readibility), and the the result
    is saved to file so that this only needs to be evaluated once, and
    all remaining pattern matching is a lookup.
    """

    # Number of letters/words
    nl = len(words1[0])
    nw1 = len(words1)  # Number of words
    nw2 = len(words2)  # Number of words

    # Convert word lists to integer arrays
    word_arr1, word_arr2 = map(words_to_int_arrays, (words1, words2))

    # equality_grid keeps track of all equalities between all pairs
    # of letters in words. Specifically, equality_grid[a, b, i, j]
    # is true when words[i][a] == words[b][j]
    equality_grid = np.zeros((nw1, nw2, nl, nl), dtype=bool)
    for i, j in it.product(range(nl), range(nl)):
        equality_grid[:, :, i, j] = np.equal.outer(word_arr1[:, i], word_arr2[:, j])

    # full_pattern_matrix[a, b] should represent the 5-color pattern
    # for guess a and answer b, with 0 -> grey, 1 -> yellow, 2 -> green
    full_pattern_matrix = np.zeros((nw1, nw2, nl), dtype=np.uint8)

    # Green pass
    for i in range(nl):
        matches = equality_grid[:, :, i, i].flatten()  # matches[a, b] is true when words[a][i] = words[b][i]
        full_pattern_matrix[:, :, i].flat[matches] = EXACT

        for k in range(nl):
            # If it's a match, mark all elements associated with
            # that letter, both from the guess and answer, as covered.
            # That way, it won't trigger the yellow pass.
            equality_grid[:, :, k, i].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False

    # Yellow pass
    for i, j in it.product(range(nl), range(nl)):
        matches = equality_grid[:, :, i, j].flatten()
        full_pattern_matrix[:, :, i].flat[matches] = MISPLACED
        for k in range(nl):
            # Similar to above, we want to mark this letter
            # as taken care of, both for answer and guess
            equality_grid[:, :, k, j].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False

    # Rather than representing a color pattern as a lists of integers,
    # store it as a single integer, whose ternary representations corresponds
    # to that list of integers.
    pattern_matrix = np.dot(
        full_pattern_matrix,
        (3**np.arange(nl)).astype(np.uint8)
    )

    return pattern_matrix


def generate_full_pattern_matrix():
    words = get_word_list()
    pattern_matrix = generate_pattern_matrix(words, words)
    # Save to file
    np.save(PATTERN_MATRIX_FILE, pattern_matrix)
    return pattern_matrix


def get_pattern_matrix(words1, words2):
    if not PATTERN_GRID_DATA:
        if not os.path.exists(PATTERN_MATRIX_FILE):
            log.info("\n".join([
                "Generating pattern matrix. This takes a minute, but",
                "the result will be saved to file so that it only",
                "needs to be computed once.",
            ]))
            generate_full_pattern_matrix()
        PATTERN_GRID_DATA['grid'] = np.load(PATTERN_MATRIX_FILE)
        PATTERN_GRID_DATA['words_to_index'] = dict(zip(
            get_word_list(), it.count()
        ))

    full_grid = PATTERN_GRID_DATA['grid']
    words_to_index = PATTERN_GRID_DATA['words_to_index']

    indices1 = [words_to_index[w] for w in words1]
    indices2 = [words_to_index[w] for w in words2]
    return full_grid[np.ix_(indices1, indices2)]


def get_pattern(guess, answer):
    if PATTERN_GRID_DATA:
        saved_words = PATTERN_GRID_DATA['words_to_index']
        if guess in saved_words and answer in saved_words:
            return get_pattern_matrix([guess], [answer])[0, 0]
    return generate_pattern_matrix([guess], [answer])[0, 0]


def pattern_from_string(pattern_string):
    return sum((3**i) * int(c) for i, c in enumerate(pattern_string))


def pattern_to_int_list(pattern):
    result = []
    curr = pattern
    for x in range(5):
        result.append(curr % 3)
        curr = curr // 3
    return result


def pattern_to_string(pattern):
    d = {MISS: "⬛", MISPLACED: "🟨", EXACT: "🟩"}
    return "".join(d[x] for x in pattern_to_int_list(pattern))


def patterns_to_string(patterns):
    return "\n".join(map(pattern_to_string, patterns))


def get_possible_words(guess, pattern, word_list):
    all_patterns = get_pattern_matrix([guess], word_list).flatten()
    return list(np.array(word_list)[all_patterns == pattern])


def get_word_buckets(guess, possible_words):
    buckets = [[] for x in range(3**5)]
    hashes = get_pattern_matrix([guess], possible_words).flatten()
    for index, word in zip(hashes, possible_words):
        buckets[index].append(word)
    return buckets


# Functions associated with entropy calculation


def get_weights(words, priors):
    frequencies = np.array([priors[word] for word in words])
    total = frequencies.sum()
    if total == 0:
        return np.zeros(frequencies.shape)
    return frequencies / total


def get_pattern_distributions(allowed_words, possible_words, weights):
    """
    For each possible guess in allowed_words, this finds the probability
    distribution across all of the 3^5 wordle patterns you could see, assuming
    the possible answers are in possible_words with associated probabilities
    in weights.

    It considers the pattern hash grid between the two lists of words, and uses
    that to bucket together words from possible_words which would produce
    the same pattern, adding together their corresponding probabilities.
    """
    pattern_matrix = get_pattern_matrix(allowed_words, possible_words)

    n = len(allowed_words)
    distributions = np.zeros((n, 3**5))
    n_range = np.arange(n)
    for j, prob in enumerate(weights):
        distributions[n_range, pattern_matrix[:, j]] += prob
    return distributions


def entropy_of_distributions(distributions, atol=1e-12):
    axis = len(distributions.shape) - 1
    return entropy(distributions, base=2, axis=axis)


def get_entropies(allowed_words, possible_words, weights):
    if weights.sum() == 0:
        return np.zeros(len(allowed_words))
    distributions = get_pattern_distributions(allowed_words, possible_words, weights)
    return entropy_of_distributions(distributions)


def max_bucket_size(guess, possible_words, weights):
    dist = get_pattern_distributions([guess], possible_words, weights)
    return dist.max()


def words_to_max_buckets(possible_words, weights):
    return dict(
        (word, max_bucket_size(word, possible_words, weights))
        for word in ProgressDisplay(possible_words)
    )

    words_and_maxes = list(w2m.items())
    words_and_maxes.sort(key=lambda t: t[1])
    words_and_maxes[:-20:-1]


def get_bucket_sizes(allowed_words, possible_words):
    """
    Returns a (len(allowed_words), 243) shape array reprenting the size of
    word buckets associated with each guess in allowed_words
    """
    weights = np.ones(len(possible_words))
    return get_pattern_distributions(allowed_words, possible_words, weights)


def get_bucket_counts(allowed_words, possible_words):
    """
    Returns the number of separate buckets that each guess in allowed_words
    would separate possible_words into
    """
    bucket_sizes = get_bucket_sizes(allowed_words, possible_words)
    return (bucket_sizes > 0).sum(1)


# Functions to analyze second guesses


def get_average_second_step_entropies(first_guesses, allowed_second_guesses, possible_words, priors):
    result = []
    weights = get_weights(possible_words, priors)
    if weights.sum() == 0:
        return np.zeros(len(first_guesses))

    distributions = get_pattern_distributions(first_guesses, possible_words, weights)
    for first_guess, dist in ProgressDisplay(list(zip(first_guesses, distributions)), leave=False, desc="Searching 2nd step entropies"):
        word_buckets = get_word_buckets(first_guess, possible_words)
        # List of maximum entropies you could achieve in
        # the second step for each pattern you might see
        # after this setp
        ents2 = np.array([
            get_entropies(
                allowed_words=allowed_second_guesses,
                possible_words=bucket,
                weights=get_weights(bucket, priors)
            ).max()
            for bucket in word_buckets
        ])
        # Multiply each such maximal entropy by the corresponding
        # probability of falling into that bucket
        result.append(np.dot(ents2, dist))
    return np.array(result)


# Solvers

def get_guess_values_array(allowed_words, possible_words, priors, look_two_ahead=False):
    weights = get_weights(possible_words, priors)
    ents1 = get_entropies(allowed_words, possible_words, weights)
    probs = np.array([
        0 if word not in possible_words else weights[possible_words.index(word)]
        for word in allowed_words
    ])

    if look_two_ahead:
        # Look two steps out, but restricted to where second guess is
        # amoung the remaining possible words
        ents2 = np.zeros(ents1.shape)
        top_indices = np.argsort(ents1)[-250:]
        ents2[top_indices] = get_average_second_step_entropies(
            first_guesses=np.array(allowed_words)[top_indices],
            allowed_second_guesses=allowed_words,
            possible_words=possible_words,
            priors=priors
        )
        return np.array([ents1, ents2, probs])
    else:
        return np.array([ents1, probs])


def entropy_to_expected_score(ent):
    """
    Based on a regression associating entropies with typical scores
    from that point forward in simulated games, this function returns
    what the expected number of guesses required will be in a game where
    there's a given amount of entropy in the remaining possibilities.
    """
    # Assuming you can definitely get it in the next guess,
    # this is the expected score
    min_score = 2**(-ent) + 2 * (1 - 2**(-ent))

    # To account for the likely uncertainty after the next guess,
    # and knowing that entropy of 11.5 bits seems to have average
    # score of 3.5, we add a line to account
    # we add a line which connects (0, 0) to (3.5, 11.5)
    return min_score + 1.5 * ent / 11.5


def get_expected_scores(allowed_words, possible_words, priors,
                        look_two_ahead=False,
                        n_top_candidates_for_two_step=25,
                        ):
    # Currenty entropy of distribution
    weights = get_weights(possible_words, priors)
    H0 = entropy_of_distributions(weights)
    H1s = get_entropies(allowed_words, possible_words, weights)

    word_to_weight = dict(zip(possible_words, weights))
    probs = np.array([word_to_weight.get(w, 0) for w in allowed_words])
    # If this guess is the true answer, score is 1. Otherwise, it's 1 plus
    # the expected number of guesses it will take after getting the corresponding
    # amount of information.
    expected_scores = probs + (1 - probs) * (1 + entropy_to_expected_score(H0 - H1s))

    if not look_two_ahead:
        return expected_scores

    # For the top candidates, refine the score by looking two steps out
    # This is currently quite slow, and could be optimized to be faster.
    # But why?
    sorted_indices = np.argsort(expected_scores)
    allowed_second_guesses = get_word_list()
    expected_scores += 1  # Push up the rest
    for i in ProgressDisplay(sorted_indices[:n_top_candidates_for_two_step], leave=False):
        guess = allowed_words[i]
        H1 = H1s[i]
        dist = get_pattern_distributions([guess], possible_words, weights)[0]
        buckets = get_word_buckets(guess, possible_words)
        second_guesses = [
            optimal_guess(allowed_second_guesses, bucket, priors, look_two_ahead=False)
            for bucket in buckets
        ]
        H2s = [
            get_entropies([guess2], bucket, get_weights(bucket, priors))[0]
            for guess2, bucket in zip(second_guesses, buckets)
        ]

        prob = word_to_weight.get(guess, 0)
        expected_scores[i] = sum((
            # 1 times Probability guess1 is correct
            1 * prob,
            # 2 times probability guess2 is correct
            2 * (1 - prob) * sum(
                p * word_to_weight.get(g2, 0)
                for p, g2 in zip(dist, second_guesses)
            ),
            # 2 plus expected score two steps from now
            (1 - prob) * (2 + sum(
                p * (1 - word_to_weight.get(g2, 0)) * entropy_to_expected_score(H0 - H1 - H2)
                for p, g2, H2 in zip(dist, second_guesses, H2s)
            ))
        ))
    return expected_scores


def get_score_lower_bounds(allowed_words, possible_words):
    """
    Assuming a uniform distribution on how likely each element
    of possible_words is, this gives the a lower boudn on the
    possible score for each word in allowed_words
    """
    bucket_counts = get_bucket_counts(allowed_words, possible_words)
    N = len(possible_words)
    # Probabilities of getting it in 1
    p1s = np.array([w in possible_words for w in allowed_words]) / N
    # Probabilities of getting it in 2
    p2s = bucket_counts / N - p1s
    # Otherwise, assume it's gotten in 3 (which is optimistics)
    p3s = 1 - bucket_counts / N
    return p1s + 2 * p2s + 3 * p3s


def optimal_guess(allowed_words, possible_words, priors,
                  look_two_ahead=False,
                  optimize_for_uniform_distribution=False,
                  purely_maximize_information=False,
                  ):
    if purely_maximize_information:
        if len(possible_words) == 1:
            return possible_words[0]
        weights = get_weights(possible_words, priors)
        ents = get_entropies(allowed_words, possible_words, weights)
        return allowed_words[np.argmax(ents)]

    # Just experimenting here...
    if optimize_for_uniform_distribution:
        expected_scores = get_score_lower_bounds(
            allowed_words, possible_words
        )
    else:
        expected_scores = get_expected_scores(
            allowed_words, possible_words, priors,
            look_two_ahead=look_two_ahead
        )
    return allowed_words[np.argmin(expected_scores)]


def brute_force_optimal_guess(all_words, possible_words, priors, n_top_picks=10, display_progress=False):
    if len(possible_words) == 0:
        # Doesn't matter what to return in this case, so just default to first word in list.
        return all_words[0]
    # For the suggestions with the top expected scores, just
    # actually play the game out from this point to see what
    # their actual scores are, and minimize.
    expected_scores = get_score_lower_bounds(all_words, possible_words)
    top_choices = [all_words[i] for i in np.argsort(expected_scores)[:n_top_picks]]
    true_average_scores = []
    if display_progress:
        iterable = ProgressDisplay(
            top_choices,
            desc=f"Possibilities: {len(possible_words)}",
            leave=False
        )
    else:
        iterable = top_choices

    for next_guess in iterable:
        scores = []
        for answer in possible_words:
            score = 1
            possibilities = list(possible_words)
            guess = next_guess
            while guess != answer:
                possibilities = get_possible_words(
                    guess, get_pattern(guess, answer),
                    possibilities,
                )
                # Make recursive? If so, we'd want to keep track of
                # the next_guess map and pass it down in the recursive
                # subcalls
                guess = optimal_guess(
                    all_words, possibilities, priors,
                    optimize_for_uniform_distribution=True
                )
                score += 1
            scores.append(score)
        true_average_scores.append(np.mean(scores))
    return top_choices[np.argmin(true_average_scores)]


# Run simulated wordle games


def get_two_step_score_lower_bound(first_guess, allowed_words, possible_words):
    """
    Useful to prove what the minimum possible average score could be
    for a given initial guess
    """
    N = len(possible_words)
    buckets = get_word_buckets(first_guess, possible_words)
    min_score = 0
    for bucket in buckets:
        if len(bucket) == 0:
            continue
        lower_bounds = get_score_lower_bounds(allowed_words, bucket)
        min_score += (len(bucket) / N) * lower_bounds.min()
    p = (1 / len(possible_words)) * (first_guess in possible_words)
    return p + (1 - p) * (1 + min_score)


def find_top_scorers(n_top_candidates=100, quiet=True, file_ext="", **kwargs):
    # Run find_best_two_step_entropy first
    file = os.path.join(get_directories()["data"], "wordle", "best_double_entropies.json")
    with open(file) as fp:
        double_ents = json.load(fp)

    answers = get_word_list(short=True)
    priors = get_true_wordle_prior()
    guess_to_score = {}
    guess_to_dist = {}

    for row in ProgressDisplay(double_ents[:n_top_candidates]):
        first_guess = row[0]
        result, decision_map = simulate_games(
            first_guess, priors=priors,
            optimize_for_uniform_distribution=True,
            quiet=quiet,
            **kwargs,
        )
        average = result["average_score"]
        total = int(np.round(average * len(answers)))
        guess_to_score[first_guess] = total
        guess_to_dist[first_guess] = result["score_distribution"]

    top_scorers = sorted(list(guess_to_score.keys()), key=lambda w: guess_to_score[w])
    result = [[w, guess_to_score[w], guess_to_dist[w]] for w in top_scorers]

    file = os.path.join(
        get_directories()["data"], "wordle",
        "best_scores" + file_ext + ".json",
    )
    with open(file, 'w') as fp:
        json.dump(result, fp)

    return result


def find_best_two_step_entropy():
    words = get_word_list()
    answers = get_word_list(short=True)
    priors = get_true_wordle_prior()

    ents = get_entropies(words, answers, get_weights(answers, priors))
    sorted_indices = np.argsort(ents)
    top_candidates = np.array(words)[sorted_indices[:-250:-1]]
    top_ents = ents[sorted_indices[:-250:-1]]

    ent_file = os.path.join(get_directories()["data"], "wordle", "best_entropies.json")
    with open(ent_file, 'w') as fp:
        json.dump([[tc, te] for tc, te in zip(top_candidates, top_ents)], fp)

    ents2 = get_average_second_step_entropies(
        top_candidates, words, answers, priors,
    )

    total_ents = top_ents + ents2
    sorted_indices2 = np.argsort(total_ents)

    double_ents = [
        [top_candidates[i], top_ents[i], ents2[i]]
        for i in sorted_indices2[::-1]
    ]

    ent2_file = os.path.join(get_directories()["data"], "wordle", "best_double_entropies.json")
    with open(ent2_file, 'w') as fp:
        json.dump(double_ents, fp)

    return double_ents


def find_smallest_second_guess_buckets(n_top_picks=100):
    all_words = get_word_list()
    possibilities = get_word_list(short=True)
    priors = get_true_wordle_prior()
    weights = get_weights(possibilities, priors)

    dists = get_pattern_distributions(all_words, possibilities, weights)
    sorted_indices = np.argsort((dists**2).sum(1))

    top_indices = sorted_indices[:n_top_picks]
    top_picks = np.array(all_words)[top_indices]
    top_dists = dists[top_indices]
    # Figure out the average number of matching words there will
    # be after two steps of game play
    avg_ts_buckets = []
    for first_guess, dist in ProgressDisplay(list(zip(top_picks, top_dists))):
        buckets = get_word_buckets(first_guess, possibilities)
        avg_ts_bucket = 0
        for p, bucket in zip(dist, buckets):
            weights = get_weights(bucket, priors)
            sub_dists = get_pattern_distributions(all_words, bucket, weights)
            min_ts_bucket = len(bucket) * (sub_dists**2).sum(1).min()
            avg_ts_bucket += p * min_ts_bucket
        avg_ts_buckets.append(avg_ts_bucket)

    result = []
    for j in np.argsort(avg_ts_buckets):
        i = top_indices[j]
        result.append((
            # Word
            all_words[i],
            # Average bucket size after first guess
            len(possibilities) * (dists[i]**2).sum(),
            # Average bucket size after second, with optimal
            # play.
            avg_ts_buckets[j],
        ))
    return result


def get_optimal_second_guess_map(first_guess, n_top_picks=10, regenerate=False):
    with open(SECOND_GUESS_MAP_FILE) as fp:
        all_sgms = json.load(fp)

    if first_guess in all_sgms and not regenerate:
        return all_sgms[first_guess]

    log.info("\n".join([
        f"Generating optimal second guess map for {first_guess}.",
        "This involves brute forcing many simulations",
        "so can take a little while."
    ]))

    sgm = [""] * 3**5
    all_words = get_word_list()
    wordle_answers = get_word_list(short=True)
    priors = get_true_wordle_prior()

    buckets = get_word_buckets(first_guess, wordle_answers)
    for pattern, bucket in ProgressDisplay(list(enumerate(buckets)), leave=False):
        sgm[pattern] = brute_force_optimal_guess(
            all_words, bucket, priors,
            n_top_picks=n_top_picks,
            display_progress=True
        )

    # Save to file
    with open(SECOND_GUESS_MAP_FILE) as fp:
        all_sgms = json.load(fp)
    all_sgms[first_guess] = sgm
    with open(SECOND_GUESS_MAP_FILE, 'w') as fp:
        json.dump(all_sgms, fp)

    return sgm


def gather_entropy_to_score_data(first_guess="crane", priors=None):
    words = get_word_list()
    answers = get_word_list(short=True)
    if priors is None:
        priors = get_true_wordle_prior()

    # List of entropy/score pairs
    ent_score_pairs = []

    for answer in ProgressDisplay(answers):
        score = 1
        possibilities = list(filter(lambda w: priors[w] > 0, words))
        guess = first_guess
        guesses = []
        entropies = []
        while True:
            guesses.append(guess)
            weights = get_weights(possibilities, priors)
            entropies.append(entropy_of_distributions(weights))
            if guess == answer:
                break
            possibilities = get_possible_words(
                guess, get_pattern(guess, answer), possibilities
            )
            guess = optimal_guess(words, possibilities, priors)
            score += 1

        for sc, ent in zip(it.count(1), reversed(entropies)):
            ent_score_pairs.append((ent, sc))

    with open(ENT_SCORE_PAIRS_FILE, 'w') as fp:
        json.dump(ent_score_pairs, fp)

    return ent_score_pairs


def simulate_games(first_guess=None,
                   priors=None,
                   look_two_ahead=False,
                   optimize_for_uniform_distribution=False,
                   second_guess_map=None,
                   exclude_seen_words=False,
                   test_set=None,
                   shuffle=False,
                   hard_mode=False,
                   purely_maximize_information=False,
                   brute_force_optimize=False,
                   brute_force_depth=10,
                   results_file=None,
                   next_guess_map_file=None,
                   quiet=False,
                   ):
    all_words = get_word_list(short=False)
    short_word_list = get_word_list(short=True)

    if first_guess is None:
        first_guess = optimal_guess(
            all_words, all_words, priors,
            **choice_config
        )

    if priors is None:
        priors = get_frequency_based_priors()

    if test_set is None:
        test_set = short_word_list

    if shuffle:
        random.shuffle(test_set)

    seen = set()

    # Function for choosing the next guess, with a dict to cache
    # and reuse results that are seen multiple times in the sim
    next_guess_map = {}

    def get_next_guess(guesses, patterns, possibilities):
        phash = "".join(
            str(g) + "".join(map(str, pattern_to_int_list(p)))
            for g, p in zip(guesses, patterns)
        )
        if second_guess_map is not None and len(patterns) == 1:
            next_guess_map[phash] = second_guess_map[patterns[0]]
        if phash not in next_guess_map:
            choices = all_words
            if hard_mode:
                for guess, pattern in zip(guesses, patterns):
                    choices = get_possible_words(guess, pattern, choices)
            if brute_force_optimize:
                next_guess_map[phash] = brute_force_optimal_guess(
                    choices, possibilities, priors,
                    n_top_picks=brute_force_depth,
                )
            else:
                next_guess_map[phash] = optimal_guess(
                    choices, possibilities, priors,
                    look_two_ahead=look_two_ahead,
                    purely_maximize_information=purely_maximize_information,
                    optimize_for_uniform_distribution=optimize_for_uniform_distribution,
                )
        return next_guess_map[phash]

    # Go through each answer in the test set, play the game,
    # and keep track of the stats.
    scores = np.zeros(0, dtype=int)
    game_results = []
    for answer in ProgressDisplay(test_set, leave=False, desc=" Trying all wordle answers"):
        guesses = []
        patterns = []
        possibility_counts = []
        possibilities = list(filter(lambda w: priors[w] > 0, all_words))

        if exclude_seen_words:
            possibilities = list(filter(lambda w: w not in seen, possibilities))

        score = 1
        guess = first_guess
        while guess != answer:
            pattern = get_pattern(guess, answer)
            guesses.append(guess)
            patterns.append(pattern)
            possibilities = get_possible_words(guess, pattern, possibilities)
            possibility_counts.append(len(possibilities))
            score += 1
            guess = get_next_guess(guesses, patterns, possibilities)

        # Accumulate stats
        scores = np.append(scores, [score])
        score_dist = [
            int((scores == i).sum())
            for i in range(1, scores.max() + 1)
        ]
        total_guesses = scores.sum()
        average = scores.mean()
        seen.add(answer)

        game_results.append(dict(
            score=int(score),
            answer=answer,
            guesses=guesses,
            patterns=list(map(int, patterns)),
            reductions=possibility_counts,
        ))
        # Print outcome
        if not quiet:
            message = "\n".join([
                "",
                f"Score: {score}",
                f"Answer: {answer}",
                f"Guesses: {guesses}",
                f"Reductions: {possibility_counts}",
                *patterns_to_string((*patterns, 3**5 - 1)).split("\n"),
                *" " * (6 - len(patterns)),
                f"Distribution: {score_dist}",
                f"Total guesses: {total_guesses}",
                f"Average: {average}",
                *" " * 2,
            ])
            if answer is not test_set[0]:
                # Move cursor back up to the top of the message
                n = len(message.split("\n")) + 1
                print(("\033[F\033[K") * n)
            else:
                print("\r\033[K\n")
            print(message)

    final_result = dict(
        score_distribution=score_dist,
        total_guesses=int(total_guesses),
        average_score=float(scores.mean()),
        game_results=game_results,
    )

    # Save results
    for obj, file in [(final_result, results_file), (next_guess_map, next_guess_map_file)]:
        if file:
            path = os.path.join(DATA_DIR, "simulation_results", file)
            with open(path, 'w') as fp:
                json.dump(obj, fp)

    return final_result, next_guess_map


if __name__ == "__main__":
    first_guess = "salet"
    results, decision_map = simulate_games(
        first_guess=first_guess,
        priors=get_true_wordle_prior(),
        optimize_for_uniform_distribution=True,
        # shuffle=True,
        # brute_force_optimize=True,
        # hard_mode=True,
    )
