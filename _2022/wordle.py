from manim_imports_ext import *
from tqdm import tqdm as ProgressDisplay
from IPython.terminal.embed import InteractiveShellEmbed as embed
from scipy.stats import entropy
from _2017.efvgt import get_confetti_animations


MISS = 0
MISPLACED = 1
EXACT = 2

DATA_DIR = os.path.join(get_directories()["data"], "wordle")
SHORT_WORD_LIST_FILE = os.path.join(DATA_DIR, "possible_words.txt")
LONG_WORD_LIST_FILE = os.path.join(DATA_DIR, "allowed_words.txt")
ENT_MAP_FILE = os.path.join(DATA_DIR, "entropies.json")
PATTERN_GRID_FILE = os.path.join(DATA_DIR, "pattern_grid.np")
WORD_FREQ_FILE = os.path.join(DATA_DIR, "wordle_words_freqs_full.txt")
WORD_FREQ_MAP_FILE = os.path.join(DATA_DIR, "freq_map.json")
PATTERN_HASH_GRID_FILE = os.path.join(DATA_DIR, "pattern_grid.npy")
SECOND_GUESS_MAP_FILE = os.path.join(DATA_DIR, "second_guess_map.json")
THIRD_GUESS_MAP_FILE = os.path.join(DATA_DIR, "third_guess_map.json")
BEST_DOUBLE_ENTROPIES = os.path.join(DATA_DIR, "best_double_entropies.json")

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


def get_entropy_map(regenerate=False, rewrite=True):
    if regenerate:
        words = get_word_list()
        priors = get_frequency_based_priors()
        weights = get_weights(words, priors)
        entropies = get_entropies(words, words, weights)
        ent_map = dict(zip(words, entropies))
        if rewrite:
            with open(ENT_MAP_FILE, 'w') as fp:
                json.dump(ent_map, fp)
        return ent_map
    with open(ENT_MAP_FILE) as fp:
        ent_map = json.load(fp)
    return ent_map


# String matching, etc.


def pattern_trit_generator(guess, true_word):
    for c1, c2 in zip(guess, true_word):
        if c1 == c2:
            yield EXACT
        elif c1 in true_word:
            yield MISPLACED
        else:
            yield MISS


def get_pattern(guess, true_word):
    """
    A unique integer id associated with the grey/yellow/green wordle
    pattern relatign a guess to the tue answer. In the ternary representation
    of this number, 0 -> grey, 1 -> yellow, 2 -> green.
    """
    return sum(
        value * (3**i)
        for i, value in enumerate(pattern_trit_generator(guess, true_word))
    )


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


def patterns_hash(patterns):
    """
    Unique id for a list of patterns
    """
    return hash("".join(map(str, patterns)))
    # return sum((3**(5 * i) + 1) * (p + 1) for i, p in enumerate(patterns))


def generate_pattern_grid(words1, words2):
    """
    A pattern for two words represents the worle-similarity
    pattern (grey -> 0, yellow -> 1, green -> 2) but as an integer
    between 0 and 3^5. Reading this integer in ternary gives the
    associated pattern.

    This function computes the pairwise patterns between two lists
    of words, returning the result as a grid of hash values. Since
    this is the most time consuming part of many computations, all
    operations that can be are vectorized, perhaps at the expense
    of easier readibility.
    """
    # Convert word lists to integer arrays
    w1, w2 = (
        np.array([[ord(c) for c in w] for w in words], dtype=np.uint8)
        for words in (words1, words2)
    )

    if len(w1) == 0 or len(w2) == 0:
        return np.zeros((len(w1), len(w2)), dtype=np.uint8)

    # equality_grid[a, b, i, j] represents whether the ith letter
    # of words1[a] equals the jth letter of words2[b]
    equality_grid = np.zeros((len(w1), len(w2), 5, 5), dtype=bool)
    for i, j in it.product(range(5), range(5)):
        equality_grid[:, :, i, j] = np.equal.outer(w1[:, i], w2[:, j])

    patterns = np.zeros((len(w1), len(w2)), dtype=np.uint8)
    three_pows = (3**np.arange(5)).astype(np.uint8)
    for i, tp in enumerate(three_pows):
        # This accounts for yellow squares
        patterns[:, :] += tp * equality_grid[:, :, i, :].any(2)
        # This accounts for green squares
        patterns[:, :] += tp * equality_grid[:, :, i, i]

    return patterns


def generate_full_pattern_grid():
    words = get_word_list()
    grid = generate_pattern_grid(words, words)
    np.save(PATTERN_HASH_GRID_FILE, grid)


def get_pattern_grid(words1, words2):
    if not PATTERN_GRID_DATA:
        PATTERN_GRID_DATA['grid'] = np.load(PATTERN_HASH_GRID_FILE)
        PATTERN_GRID_DATA['words_to_index'] = dict(zip(
            get_word_list(), it.count()
        ))

    full_grid = PATTERN_GRID_DATA['grid']
    words_to_index = PATTERN_GRID_DATA['words_to_index']

    indices1 = [words_to_index[w] for w in words1]
    indices2 = [words_to_index[w] for w in words2]
    return full_grid[np.ix_(indices1, indices2)]


def get_possible_words(guess, pattern, word_list):
    all_hashes = get_pattern_grid([guess], word_list).flatten()
    return list(np.array(word_list)[all_hashes == pattern])


def get_word_buckets(guess, possible_words):
    buckets = [[] for x in range(3**5)]
    hashes = get_pattern_grid([guess], possible_words).flatten()
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
    pattern_grid = get_pattern_grid(allowed_words, possible_words)

    n = len(allowed_words)
    distributions = np.zeros((n, 3**5))
    n_range = np.arange(n)
    for j, prob in enumerate(weights):
        distributions[n_range, pattern_grid[:, j]] += prob
    return distributions


def entropy_of_distributions(distributions, atol=1e-12):
    axis = len(distributions.shape) - 1
    return entropy(distributions, base=2, axis=axis)


def get_entropies(allowed_words, possible_words, weights):
    if weights.sum() == 0:
        return np.zeros(len(allowed_words))
    distributions = get_pattern_distributions(allowed_words, possible_words, weights)
    return entropy_of_distributions(distributions)


def generate_entropy_map():
    words = get_word_list()
    priors = get_frequency_based_priors()
    weights = get_weights(words, priors)
    ent_map = dict(zip(
        words, get_entropies(words, words, weights)
    ))
    with open(ENT_MAP_FILE, 'w') as fp:
        json.dump(ent_map, fp)
    return ent_map


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


def build_best_second_guess_map(guess, allowed_words, possible_words, priors, **kwargs):
    word_buckets = get_word_buckets(guess, possible_words)
    msg = f"Building second guess map for \"{guess}\""
    return [
        optimal_guess(allowed_words, bucket, priors, **kwargs)
        for bucket in ProgressDisplay(word_buckets, desc=msg, leave=False)
    ]


def get_second_guess_map(guess, regenerate=False, save_to_file=True, **kwargs):
    with open(SECOND_GUESS_MAP_FILE) as fp:
        saved_maps = json.load(fp)

    if guess not in saved_maps or regenerate:
        words = get_word_list()
        priors = get_frequency_based_priors()

        sg_map = build_best_second_guess_map(guess, words, words, priors, **kwargs)
        saved_maps[guess] = sg_map

        if save_to_file:
            with open(SECOND_GUESS_MAP_FILE, 'w') as fp:
                json.dump(saved_maps, fp)

    return saved_maps[guess]


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
    if not isinstance(ent, np.ndarray):
        ent = np.array(ent)
    log_part = np.log(ent + 1, out=np.zeros(ent.size), where=(ent > 0))
    return 1 + 0.56 * log_part + 0.1 * ent


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


def optimal_guess(allowed_words, possible_words, priors, look_two_ahead=False):
    expected_guess_values = get_expected_scores(
        allowed_words, possible_words, priors,
        look_two_ahead=look_two_ahead
    )
    return allowed_words[np.argmin(expected_guess_values)]


# Scene types


class WordleScene(Scene):
    grid_height = 6
    font_to_grid_height_ratio = 10
    grid_center = ORIGIN
    secret_word = None
    color_map = {
        0: "#797C7E",  # GREY
        1: "#C6B566",  # YELLOW
        2: GREEN_D,  # GREEN
    }
    uniform_prior = False
    wordle_based_prior = False
    freq_prior = True

    CONFIG = {"random_seed": None}

    def setup(self):
        self.all_words = self.get_word_list()
        self.priors = self.get_priors()
        if self.secret_word is None:
            s_words = get_word_list(short=True)
            self.secret_word = random.choice(s_words)
        self.guesses = []
        self.patterns = []
        self.possibilities = self.get_initial_possibilities()

        self.add_grid()

    def get_word_list(self):
        return get_word_list()

    def get_initial_possibilities(self):
        return get_word_list()

    def get_priors(self):
        words = self.all_words
        if self.uniform_prior:
            return dict((w, 1) for w in words)
        elif self.wordle_based_prior:
            return get_true_wordle_prior()
        else:
            return get_frequency_based_priors()

    def get_pattern(self, guess):
        return get_pattern(guess, self.secret_word)

    def get_current_entropy(self):
        weights = get_weights(self.possibilities, self.priors)
        return entropy_of_distributions(weights)

    ##

    def add_grid(self):
        buff = 0.1
        row = Square(side_length=1).get_grid(1, 5, buff=buff)
        grid = row.get_grid(6, 1, buff=buff)
        grid.set_height(self.grid_height)
        grid.move_to(self.grid_center)
        grid.set_stroke(WHITE, 2)
        grid.words = VGroup()
        grid.pending_word = VGroup()
        grid.add(grid.words, grid.pending_word)
        grid.pending_pattern = None
        grid.add_updater(lambda m: m)
        self.grid = grid
        self.add(grid)

    def get_curr_row(self):
        return self.grid[len(self.grid.words)]

    def get_curr_square(self):
        row = self.get_curr_row()
        return row[len(self.grid.pending_word)]

    def add_letter(self, letter):
        grid = self.grid
        if len(grid.pending_word) == len(grid[0]):
            return

        font_size = self.font_to_grid_height_ratio * self.grid_height
        letter_mob = Text(letter.upper(), font="Consolas", font_size=font_size)
        letter_mob.move_to(self.get_curr_square())

        grid.pending_word.add(letter_mob)

    def delete_letter(self):
        if len(self.grid.pending_word) == 0:
            return
        letter_mob = self.grid.pending_word[-1]
        self.grid.pending_word.remove(letter_mob)

    def add_word(self, word, wait_time_per_letter=0.1):
        for letter in word:
            self.add_letter(letter)
            self.wait(
                wait_time_per_letter,
                ignore_presenter_mode=True
            )

    def pending_word_as_string(self):
        return "".join(
            t.text.lower()
            for t in self.grid.pending_word
        )

    def is_valid_guess(self):
        guess = self.pending_word_as_string()
        return guess in self.all_words

    def reveal_pattern(self, pattern=None, animate=True):
        grid = self.grid
        guess = self.pending_word_as_string()

        if not self.is_valid_guess():
            self.shake_word_out()
            return False

        if pattern is None:
            pattern = self.get_pattern(guess)

        self.show_pattern(pattern, animate=animate)

        self.guesses.append(guess)
        self.patterns.append(pattern)
        grid.words.add(grid.pending_word.copy())
        grid.pending_word.set_submobjects([])
        grid.pending_pattern = None
        self.possibilities = get_possible_words(
            guess, pattern, self.possibilities
        )

        # Win condition
        if self.has_won():
            self.win_animation()

        return True

    def shake_word_out(self):
        row = self.get_curr_row()
        c = row.get_center().copy()
        func = bezier([0, 0, 1, 1, -1, -1, 0, 0])
        self.play(UpdateFromAlphaFunc(
            VGroup(row, self.grid.pending_word),
            lambda m, a: m.move_to(c + func(a) * RIGHT),
            run_time=0.5,
        ))
        self.grid.pending_word.set_submobjects([])

    def show_pattern(self, pattern, animate=False, added_anims=[]):
        row = self.get_curr_row()
        colors = self.get_colors(pattern)
        if animate:
            self.animate_color_change(row, self.grid.pending_word, colors, added_anims)
        else:
            self.set_row_colors(row, colors)

        self.grid.pending_pattern = pattern

    def set_row_colors(self, row, colors):
        for square, color in zip(row, colors):
            square.set_fill(color, 1)

    def animate_color_change(self, row, word, colors, added_anims=[]):
        for square, color in zip(row, colors):
            square.future_color = color

        def alpha_func(mob, alpha):
            if not hasattr(mob, 'initial_height'):
                mob.initial_height = mob.get_height()
            mob.set_height(
                mob.initial_height * max(abs(interpolate(1, -1, alpha)), 1e-6),
                stretch=True
            )
            if isinstance(mob, Square) and alpha > 0.5:
                mob.set_fill(mob.future_color, 1)

        self.play(
            *(
                LaggedStart(*(
                    UpdateFromAlphaFunc(sm, alpha_func)
                    for sm in mob
                ), lag_ratio=0.5, run_time=2)
                for mob in (row, word)
            ),
            *added_anims
        )
        self.remove(row, word)
        self.add(self.grid)

    def get_colors(self, pattern):
        return [self.color_map[key] for key in pattern_to_int_list(pattern)]

    def win_animation(self):
        grid = self.grid
        row = grid[len(grid.words) - 1]
        letters = grid.words[-1]
        mover = VGroup(*(
            VGroup(square, letter)
            for square, letter in zip(row, letters)
        ))
        y = row.get_y()
        bf = bezier([0, 0, 1, 1, -1, -1, 0, 0])

        self.play(
            LaggedStart(*(
                UpdateFromAlphaFunc(sm, lambda m, a: m.set_y(y + 0.2 * bf(a)))
                for sm in mover
            ), lag_ratio=0.1, run_time=1.5),
            LaggedStart(*(
                Flash(letter, line_length=0.1, flash_radius=0.4)
                for letter in letters
            ), lag_ratio=0.3, run_time=1.5),
        )

    def has_won(self):
        return len(self.patterns) > 0 and self.patterns[-1] == 3**5 - 1

    @staticmethod
    def get_grid_of_words(all_words, n_rows, n_cols, dots_index=-5, sort_key=None, font_size=24):
        if sort_key:
            all_words = list(sorted(all_words, key=sort_key))
        subset = all_words[:n_rows * n_cols]
        show_ellipsis = len(subset) < len(all_words)

        if show_ellipsis:
            subset[dots_index] = "..." if n_cols == 1 else "....."
            subset[dots_index + 1:] = all_words[dots_index + 1:]

        full_string = ""
        for i, word in zip(it.count(1), subset):
            full_string += str(word)
            if i % n_cols == 0:
                full_string += " \n"
            else:
                full_string += "  "

        full_text_mob = Text(full_string, font="Consolas", font_size=font_size)

        result = VGroup()
        for word in subset:
            part = full_text_mob.get_part_by_text(word)
            part.text = word
            result.add(part)

        if show_ellipsis and n_cols == 1:
            result[dots_index].rotate(PI / 2)
            result[dots_index].next_to(result[dots_index - 1], DOWN, SMALL_BUFF)
            result[dots_index + 1:].next_to(result[dots_index], DOWN, SMALL_BUFF)
        result.set_color(GREY_A)
        result.words = subset

        return result

    @staticmethod
    def patterns_to_squares(patterns, color_map=None):
        if color_map is None:
            color_map = WordleScene.color_map
        row = Square().get_grid(1, 5, buff=SMALL_BUFF)
        rows = row.get_grid(len(patterns), 1, buff=SMALL_BUFF)
        rows.set_stroke(WHITE, 1)
        for pattern, row in zip(patterns, rows):
            for square, key in zip(row, pattern_to_int_list(pattern)):
                square.set_fill(color_map[key], 1)
        return rows

    # Interactive parts
    def on_key_press(self, symbol, modifiers):
        try:
            char = chr(symbol)
        except OverflowError:
            log.warning("The value of the pressed key is too large.")
            return

        is_letter = (ord('a') <= ord(char) <= ord('z'))

        if is_letter:
            self.add_letter(char)
        elif symbol == 65288:  # Delete
            self.delete_letter()
        elif symbol == 65293:  # Enter
            self.reveal_pattern()

        if char == 'q' and modifiers == 1:
            self.delete_letter()
            self.quit_interaction = True

        if not is_letter:
            super().on_key_press(symbol, modifiers)


class WordleSceneWithAnalysis(WordleScene):
    grid_center = [-1.75, 1, 0]
    grid_height = 4.5
    look_two_ahead = False
    show_prior = True
    n_top_picks = 13
    entropy_color = TEAL_C
    prior_color = BLUE_C
    weight_to_prob = 3.0
    pre_computed_first_guesses = []

    def setup(self):
        self.wait_to_proceed = True
        super().setup()
        self.show_possible_words()
        self.add_count_title()
        self.add_guess_value_grid_title()
        self.init_guess_value_grid()
        self.info_labels = VGroup()
        self.add(self.info_labels)

    def construct(self):
        self.show_guess_values()

    def add_guess_value_grid_title(self):
        titles = VGroup(
            Text("Top picks"),
            Text("E[Info.]", color=self.entropy_color),
        )
        if self.look_two_ahead:
            titles.add(TexText("E[Info$_2$]", color=self.entropy_color)[0])
        if self.show_prior:
            titles.add(Tex("p(\\text{word})", color=self.prior_color))

        titles.scale(0.7)
        titles.arrange(RIGHT, buff=MED_LARGE_BUFF)
        titles.set_max_width(5)
        low_y = titles[0][0].get_bottom()[1]

        for title in titles:
            first = title.family_members_with_points()[0]
            title.shift((low_y - first.get_bottom()[1]) * UP)
            underline = Underline(title)
            underline.match_y(first.get_bottom() + 0.025 * DOWN)
            underline.set_stroke(WHITE, 2)
            underline.scale(1.1)
            title.add_to_back(underline)
            title.set_backstroke()
            underline.set_stroke(GREY_C, 2)

        titles.to_edge(UP, buff=MED_SMALL_BUFF)
        titles.to_edge(RIGHT, buff=MED_SMALL_BUFF)

        self.add(titles)
        self.guess_value_grid_titles = titles

    def add_count_title(self):
        title = VGroup(
            Text("# Possibilities"),
            Text("/"),
            Text("Uncertainty", color=self.entropy_color),
        )
        title.arrange(RIGHT, buff=SMALL_BUFF)
        title.match_width(self.count_label).scale(1.1)
        title.next_to(self.count_label, UP, buff=MED_LARGE_BUFF)
        self.count_title = title
        self.add(title)

    def init_guess_value_grid(self):
        titles = self.guess_value_grid_titles
        line = Line().match_width(titles)
        line.set_stroke(GREY_C, 1)
        lines = line.get_grid(self.n_top_picks, 1, buff=0.5)
        lines.next_to(titles, DOWN, buff=0.75)

        self.guess_value_grid_lines = lines
        self.guess_value_grid = VGroup()

    def get_count_label(self):
        score = len(self.grid.words)
        label = VGroup(
            Integer(len(self.possibilities), edge_to_fix=UR),
            Text("Pos,"),
            DecimalNumber(self.get_current_entropy(), edge_to_fix=UR, color=self.entropy_color),
            Text("Bits", color=self.entropy_color),
        )
        label.arrange(
            RIGHT,
            buff=MED_SMALL_BUFF,
            aligned_edge=UP,
        )
        label.scale(0.6)
        label.next_to(self.grid[score], LEFT)
        return label

    def reveal_pattern(self):
        is_valid_guess = self.is_valid_guess()
        if is_valid_guess:
            self.isolate_guessed_row()

        super().reveal_pattern()

        if self.presenter_mode:
            while self.wait_to_proceed:
                self.update_frame(1 / self.camera.frame_rate)
            self.wait_to_proceed = True

        if is_valid_guess and not self.has_won():
            self.show_possible_words()
            self.wait()
            self.show_guess_values()
        if self.has_won():
            self.play(
                FadeOut(self.guess_value_grid, RIGHT),
                FadeOut(self.guess_value_grid_titles, RIGHT),
            )

    def show_pattern(self, pattern, *args, **kwargs):
        guess = self.pending_word_as_string()
        new_possibilities = get_possible_words(
            guess, pattern, self.possibilities
        )
        for word_mob, word, bar in zip(self.shown_words, self.shown_words.words, self.prob_bars):
            if word not in new_possibilities and word != "...":
                word_mob.set_fill(RED, 0.5)
                bar.set_opacity(0.2)
        self.show_pattern_information(guess, pattern, new_possibilities)
        super().show_pattern(pattern, *args, **kwargs)

    def show_pattern_information(self, guess, pattern, new_possibilities):
        # Put bits label next to pattern
        weights = get_weights(self.possibilities, self.priors)
        prob = sum(
            weight for word, weight in zip(self.possibilities, weights)
            if word in new_possibilities
        )
        info = -math.log2(prob)

        ref = self.count_label[2:]
        info_label = VGroup(
            DecimalNumber(info),
            Text("Bits")
        )
        info_label.set_color(RED)
        info_label.arrange(RIGHT)
        info_label.match_height(ref)
        info_label.next_to(self.get_curr_row(), RIGHT, buff=MED_SMALL_BUFF)
        info_label.match_y(ref)
        self.info_labels.add(info_label)

    def isolate_guessed_row(self):
        guess = self.pending_word_as_string()
        rows = self.guess_value_grid
        row_words = [row[0].text for row in rows]

        if guess in row_words:
            row = rows[row_words.index(guess)]
            rows.set_opacity(0.2)
            row[:-1].set_fill(YELLOW, 1)
        else:
            new_row = self.get_guess_value_row(
                self.guess_value_grid_lines[0], guess,
            )
            rows.shift(DOWN)
            rows.add(new_row)

    def get_shown_words(self, font_size=24):
        return self.get_grid_of_words(
            self.possibilities,
            n_rows=20 - 2 * len(self.grid.words),
            n_cols=1
        )

    def get_probability_bars(self, shown_words, max_width=1.0):
        mobs = shown_words
        words = shown_words.words
        probs = [self.priors.get(w, 0) for w in words]  # Unnormalized
        height = mobs[0].get_height() * 0.7
        bars = VGroup(*(
            Rectangle(
                width=prob * max_width,
                height=height,
                fill_color=self.prior_color,
                fill_opacity=0.7,
                stroke_width=0.5 * (prob > 0),
                stroke_color=self.prior_color
            )
            for prob in probs
        ))
        for bar, mob in zip(bars, mobs):
            bar.next_to(mob, RIGHT, SMALL_BUFF)
            bar.align_to(bars[0], LEFT)

        # if not self.show_prior:
        #     bars.set_opacity(0)

        return bars

    def show_possible_words(self):
        shown_words = self.get_shown_words()
        count_label = self.get_count_label()
        shown_words.next_to(count_label[:2], DOWN, buff=0.35)
        prob_bars = self.get_probability_bars(shown_words)

        if len(self.grid.words) > 0:
            # Set up label transition
            prev_count_label = self.count_label
            count_label.shift(
                (prev_count_label[1].get_right() - count_label[1].get_right())[0] * RIGHT
            )

            num_rate_func = squish_rate_func(rush_into, 0.3, 1)

            def update_moving_count_label(label, alpha):
                for i in (0, 2):
                    label[i].set_value(interpolate(
                        prev_count_label[i].get_value(),
                        count_label[i].get_value(),
                        num_rate_func(alpha),
                    ))
                label.set_y(interpolate(
                    prev_count_label.get_y(),
                    count_label.get_y(),
                    alpha
                ))
                return label

            label_transition = UpdateFromAlphaFunc(
                prev_count_label.copy(),
                update_moving_count_label,
                remover=True
            )

            # Set up word transition
            prev_words = self.shown_words
            for shown_word, s_word in zip(shown_words, shown_words.words):
                shown_word.save_state()
                if s_word in prev_words.words:
                    index = prev_words.words.index(s_word)
                    shown_word.move_to(prev_words[index])
                    prev_words[index].set_opacity(0)
                    self.prob_bars[index].set_opacity(0)
                elif "..." in prev_words.words:
                    shown_word.move_to(prev_words[prev_words.words.index("...")])
                    shown_word.set_opacity(0)
                else:
                    shown_word.set_opacity(0)

            prev_words.generate_target()
            for i, word in enumerate(prev_words.words):
                if word not in shown_words.words:
                    fader = prev_words.target[i]
                    fader.set_opacity(0)
                    fader.shift(LEFT)

            # Set up bar transitions
            for bar, s_word, word in zip(prob_bars, shown_words, shown_words.words):
                bar.save_state()
                if word not in prev_words.words:
                    bar.set_opacity(0)
                bar.match_y(s_word)
                bar.align_to(bar.saved_state, LEFT)

            # Carry out animations
            self.play(
                FadeOut(self.prob_bars, run_time=0.25),
                FadeOut(self.guess_value_grid, RIGHT),
                label_transition,
                MoveToTarget(prev_words, run_time=0.5),
                LaggedStartMap(Restore, shown_words, run_time=1),
                LaggedStartMap(Restore, prob_bars, run_time=1),
                run_time=1,
            )
            self.add(count_label)
            self.remove(prev_words)
            shown_words.set_opacity(1)

        self.add(count_label)
        self.add(shown_words)
        self.add(prob_bars)
        self.count_label = count_label
        self.shown_words = shown_words
        self.prob_bars = prob_bars

    def show_guess_values(self):
        self.guess_value_grid = self.get_guess_value_grid()
        self.play(ShowIncreasingSubsets(self.guess_value_grid))

    def get_guess_value_grid(self, font_size=36):
        if self.pre_computed_first_guesses and len(self.grid.words) == 0:
            guesses = self.pre_computed_first_guesses
        else:
            guesses = self.all_words
        guess_values_array = get_guess_values_array(
            guesses,
            self.possibilities,
            self.priors,
            look_two_ahead=self.look_two_ahead,
        )
        expected_scores = get_expected_scores(
            guesses,
            self.possibilities,
            self.priors,
            look_two_ahead=self.look_two_ahead
        )
        top_indices = np.argsort(expected_scores)[:self.n_top_picks]
        top_words = np.array(guesses)[top_indices]
        top_guess_value_parts = guess_values_array[:, top_indices]

        lines = self.get_guess_value_grid_lines()
        guess_value_grid = VGroup(*(
            self.get_guess_value_row(line, word, *values)
            for line, word, values in zip(
                lines, top_words, top_guess_value_parts.T
            )
        ))
        for value, row in zip(guess_values_array.sum(0)[top_indices], guess_value_grid):
            if value == 0:
                row.set_opacity(0)

        guess_value_grid.set_stroke(background=True)
        return guess_value_grid

    def get_guess_value_row(self, line, word,
                            entropy=None,
                            entropy2=None,
                            probability=None,
                            font_size=36):
        titles = self.guess_value_grid_titles
        row = VGroup()

        # Word
        word_mob = Text(str(word), font="Consolas", font_size=font_size)
        index = np.argmin([c.get_height() for c in word_mob])
        aligner = word_mob[index]
        word_mob.shift(line.get_center() - aligner.get_bottom() + 0.5 * SMALL_BUFF * UP)
        word_mob.match_x(titles[0])
        row.add(word_mob)

        # Entropy
        if entropy is None:
            weights = get_weights(self.possibilities, self.priors)
            entropy = get_entropies([word], self.possibilities, weights)[0]
        dec_kw = dict(num_decimal_places=2, font_size=font_size)
        row.add(DecimalNumber(entropy, color=self.entropy_color, **dec_kw))

        # Second entropy
        if self.look_two_ahead:
            if entropy2 is None:
                entropy2 = get_average_second_step_entropies(
                    [word], self.all_words, self.possibilities, self.priors
                )
            row.add(DecimalNumber(entropy2, color=self.entropy_color, **dec_kw))

        # Prior
        if self.show_prior:
            if probability is None:
                if word in self.possibilities:
                    weights = get_weights(self.possibilities, self.priors)
                    probability = weights[self.possibilities.index(word)]
                else:
                    probability = 0
            dec_kw['num_decimal_places'] = 5
            # Dividing out by the weight given to prob in scores
            row.add(DecimalNumber(probability, color=self.prior_color, **dec_kw))

        for mob, title in zip(row, titles):
            if mob is not word_mob:
                mob.match_y(aligner, DOWN)
            mob.match_x(title)

        row.add(line)
        return row

    def get_guess_value_grid_lines(self):
        titles = self.guess_value_grid_titles
        line = Line().match_width(titles)
        line.set_stroke(GREY_C, 1)
        lines = line.get_grid(self.n_top_picks, 1, buff=0.5)
        lines.next_to(titles, DOWN, buff=0.75)
        return lines

    def get_column_of_numbers(self, values, row_refs, col_ref, num_decimal_places=2, font_size=36):
        mobs = VGroup(*(
            DecimalNumber(
                value,
                num_decimal_places=num_decimal_places,
                font_size=font_size
            )
            for value in values
        ))
        for row_ref, mob in zip(row_refs, mobs):
            mob.match_x(col_ref)
            mob.match_y(row_ref)
        return mobs

    def on_key_press(self, symbol, modifiers):
        if chr(symbol) == " ":
            self.wait_to_proceed = False
        super().on_key_press(symbol, modifiers)


class WordleDistributions(WordleScene):
    grid_center = [-4.5, -0.5, 0]
    grid_height = 5
    bar_style = dict(
        fill_color=TEAL_D,
        fill_opacity=0.8,
        stroke_color=WHITE,
        stroke_width=0.1,
    )
    show_fraction_in_p_label = True

    def get_axes(self,
                 x_max=3**5 / 2, y_max=0.1,
                 width=7.5,
                 height=6):
        axes = Axes(
            (0, x_max),
            (0, y_max, y_max / 5),
            height=height,
            width=width,
            x_axis_config={
                "tick_size": 0,
            }
        )
        axes.next_to(self.grid, RIGHT, LARGE_BUFF, aligned_edge=DOWN)
        # y_label = Tex("p(\\text{Pattern})", font_size=24)
        # y_label.next_to(axes.y_axis.get_top(), UR, buff=SMALL_BUFF)
        # axes.y_axis.add(y_label)
        axes.y_axis.add_numbers(num_decimal_places=2)

        x_label = Text("Pattern", font_size=24)
        x_label.next_to(axes.x_axis.get_right(), DR, MED_SMALL_BUFF)
        x_label.shift_onto_screen()
        axes.x_axis.add(x_label)

        self.axes = axes
        return axes

    def get_total_words_label(self, font_size=36):
        label = VGroup(
            Integer(len(self.all_words), font_size=font_size, edge_to_fix=UR),
            Text("Total words", font_size=font_size)
        )
        label.arrange(RIGHT, aligned_edge=UP)
        label.match_x(self.grid)
        label.to_edge(UP, buff=MED_SMALL_BUFF)
        return label

    def get_dynamic_match_label(self, font_size=36):
        label = VGroup(
            Integer(len(self.possibilities), font_size=font_size, edge_to_fix=UR),
            Text("Possible matches", font_size=font_size)
        )
        label.arrange(RIGHT, aligned_edge=DOWN)
        label.set_max_width(self.grid.get_width())
        label.next_to(self.grid, UP)

        def update_label(label):
            word = self.pending_word_as_string()
            if self.grid.pending_pattern is None or not self.is_valid_guess():
                label.set_opacity(0)
            else:
                buckets = get_word_buckets(word, self.possibilities)
                bucket_size = len(buckets[self.grid.pending_pattern])
                label[0].set_value(bucket_size)
                label[0].next_to(label[1], LEFT, submobject_to_align=label[0][-1])
                label.set_opacity(1)

        label.add_updater(update_label)

        return label

    def get_bars(self, axes, values):
        x_unit = axes.x_axis.unit_size
        y_unit = axes.y_axis.unit_size
        bars = Rectangle(width=x_unit, **self.bar_style).replicate(3**5)

        for x, bar, value in zip(it.count(), bars, values):
            bar.set_height(value * y_unit, stretch=True)
            bar.move_to(axes.c2p(x, 0), DL)
        return bars

    def get_distribution_bars(self, axes, guess):
        distribution = get_pattern_distributions(
            [guess], self.possibilities,
            get_weights(self.possibilities, self.priors)
        )[0]
        buckets = get_word_buckets(guess, self.possibilities)
        pattern_indices = np.argsort(distribution)[::-1]

        bars = self.get_bars(axes, distribution[pattern_indices])
        bars.patterns = pattern_indices

        for i, bar in enumerate(bars):
            bar.prob = distribution[pattern_indices[i]]
            bar.count = len(buckets[pattern_indices[i]])
        return bars

    def get_bar_indicator(self, bars, pattern_index):
        pattern_index_tracker = ValueTracker(pattern_index)

        def get_pattern_index():
            return int(pattern_index_tracker.get_value())

        def get_pattern():
            return bars.patterns[get_pattern_index()]

        tri = ArrowTip(angle=PI / 2)
        tri.set_height(0.1)
        tri.add_updater(lambda m: m.next_to(bars[get_pattern_index()], DOWN, buff=0))

        row = self.get_curr_row()
        row_copy = row.copy()
        row_copy.scale(0.25)
        row_copy.add_updater(lambda m: m.next_to(tri, DOWN, SMALL_BUFF))

        bars.add_updater(lambda m: m.set_opacity(0.35))
        bars.add_updater(lambda m: m[get_pattern_index()].set_opacity(1))
        self.grid.add_updater(lambda m: self.show_pattern(get_pattern()))
        self.add(self.grid)
        row_copy.add_updater(lambda m: m.match_style(row).set_stroke(width=0.1))

        self.mouse_drag_point.move_to(tri)
        pattern_index_tracker.add_updater(lambda m: m.set_value(
            clip(self.axes.x_axis.p2n(self.mouse_drag_point.get_center()), 0, 3**5)
        ))

        indicator = Group(tri, row_copy)

        def get_bar():
            value = pattern_index_tracker.get_value()
            index = int(clip(value, 0, 3**5 - 1))
            return bars[index]

        return indicator, pattern_index_tracker, get_bar

    def get_dynamic_bar_label(self, tex, font_size=36):
        row_copy = self.get_curr_row().copy()
        row_copy.scale(0.25)
        ndp = len(tex[-1].split(".")[1])
        dec = DecimalNumber(0, num_decimal_places=ndp, font_size=font_size)
        result = VGroup(*Tex(*tex, font_size=font_size))
        row_copy.replace(result[1], dim_to_match=0)
        dec.replace(result[-1])
        result.replace_submobject(1, row_copy)

        result.remove(result[-1])
        result.add(dec)

        result.add_updater(lambda m: m[1].match_style(self.get_curr_row()).set_stroke(WHITE, 0.1))
        return result

    def get_p_label(self, get_bar, max_y=1):
        poss_string = "{:,}".format(len(self.possibilities)).replace(",", "{,}")
        strs = ["p\\left(", "00000", "\\right)", "="]
        if self.show_fraction_in_p_label:
            strs.extend(["{" + poss_string, "\\over ", poss_string + "}", "=", ])
        strs.append("0.0000")
        p_label = self.get_dynamic_bar_label(strs)

        if self.show_fraction_in_p_label:
            num = Integer(edge_to_fix=DOWN, font_size=36)
            num.move_to(p_label[4], DOWN)
            p_label.replace_submobject(4, num)

        def update_label(label):
            label[4].set_value(int(get_bar().count))
            label[-1].set_value(get_bar().prob)
            label.next_to(get_bar(), UR, SMALL_BUFF)
            label.set_y(min(max_y, label.get_y()))
            label.shift_onto_screen(buff=1.0)

        p_label.add_updater(update_label)
        return p_label

    def get_information_label(self, p_label, get_bar):
        info_label = self.get_dynamic_bar_label(
            (
                "I\\left(", "00000", "\\right)", "=",
                "\\log_2\\left(1 / p)", "=",
                "0.00"
            ),
        )
        info_label.add_updater(lambda m: m[-1].set_value(-safe_log2(get_bar().prob)))
        info_label.add_updater(lambda m: m.next_to(p_label, UP, aligned_edge=LEFT))
        info_label.add_updater(lambda m: m.shift_onto_screen())
        return info_label

    def get_entropy_label(self, font_size=36):
        guess = self.pending_word_as_string()
        if self.is_valid_guess():
            entropy = get_entropies(
                [guess],
                self.possibilities,
                get_weights(self.possibilities, self.priors)
            )[0]
        else:
            entropy = 0

        lhs = Tex(
            "E[I] = \\sum_x ",
            "p(x) \\cdot \\log_2(1 / p(x))", "=",
            tex_to_color_map={"I": BLUE},
            font_size=font_size,
        )
        value = DecimalNumber(entropy, font_size=font_size)
        value.next_to(lhs[-1], RIGHT)
        result = VGroup(lhs, value)
        result.move_to(self.axes, UR)
        result.to_edge(UP)

        return result

    def get_grid_of_matches(self, n_rows=20, n_cols=9):
        if self.grid.pending_pattern is not None:
            buckets = get_word_buckets(
                self.pending_word_as_string(),
                self.possibilities
            )
            words = buckets[self.grid.pending_pattern]
        else:
            words = self.possibilities
        word_mobs = self.get_grid_of_words(words, n_rows, n_cols)
        word_mobs.move_to(midpoint(self.grid.get_right(), RIGHT_SIDE))
        return word_mobs

    # Animations

    def add_distribution(self, axes):
        pass


class ExternalPatternEntry(WordleSceneWithAnalysis):
    # uniform_prior = True
    # wordle_based_prior = True

    def setup(self):
        super().setup()

    def get_pattern(self, guess):
        return pattern_from_string(input("Pattern please:"))


# Scenes


class IntroduceGame(WordleScene):
    secret_word = "brown"
    grid_center = 3.5 * LEFT

    def construct(self):
        # Secret
        grid = self.grid
        row_copy = self.get_curr_row().copy()
        secret = VGroup(
            Text("Secret word"),
            Vector(0.5 * DOWN),
            row_copy,
        )
        secret[2].match_width(secret[0])
        secret.arrange(DOWN, buff=MED_SMALL_BUFF)
        secret.next_to(grid, RIGHT, buff=2.0, aligned_edge=UP)

        word_list = random.sample(get_word_list(short=True), 100)
        word_list.append("?????")
        words = VGroup(*(Text(word, font="Consolas") for word in word_list))
        words.set_height(row_copy.get_height() * 0.7)
        words[-1].set_color(RED)

        for word in words:
            for char, square in zip(word, row_copy):
                char.move_to(square)

        self.wait()
        self.wait()
        self.play(
            Write(secret[0]),
            ShowCreation(secret[1]),
            TransformFromCopy(grid[0], secret[2])
        )
        self.play(
            ShowSubmobjectsOneByOne(words, run_time=10, rate_func=linear),
        )
        self.wait()

        self.row_copy = row_copy
        self.q_marks = words[-1]

        # Guesses
        numbers = VGroup(*(Integer(i) for i in range(1, 7)))
        for number, row in zip(numbers, grid):
            number.next_to(row, LEFT)

        self.play(FadeIn(numbers, lag_ratio=0.1))

        guesses = ["three", "blues", "one", "onnne", "wonky", "brown"]
        if not self.presenter_mode:
            for guess in guesses:
                self.add_word(guess)
                self.reveal_pattern()
        else:
            self.wait(note=f"Type {guesses}")

        # Show word lists
        all_words = get_word_list()
        answers = get_word_list(short=True)

        titles = VGroup(
            VGroup(Text("Allowed guesses"), Integer(len(all_words))),
            VGroup(Text("Possible answers"), Integer(len(answers))),
        )
        for title in titles:
            title.arrange(DOWN)

        titles.scale(0.8)
        titles.arrange(RIGHT, buff=1.5, aligned_edge=UP)
        titles[1][1].match_y(titles[0][1])
        titles.to_corner(UR)
        titles.to_edge(RIGHT, buff=MED_LARGE_BUFF)

        word_columns = VGroup(
            self.get_grid_of_words(all_words, 20, 5),
            self.get_grid_of_words(answers, 20, 1),
        )
        word_columns[1].set_color(GREEN)
        for column, title in zip(word_columns, titles):
            column.next_to(title, DOWN, MED_LARGE_BUFF)

        grid.add(numbers)
        self.play(
            FadeOut(secret),
            FadeOut(self.q_marks),
            grid.animate.scale(0.7, about_edge=LEFT),
            Write(titles[0][0], run_time=1),
        )
        self.play(
            CountInFrom(titles[0][1], 0),
            ShowIncreasingSubsets(word_columns[0]),
        )
        self.wait()
        self.play(
            FadeIn(titles[1][0]),
            CountInFrom(titles[1][1], 0),
            ShowIncreasingSubsets(word_columns[1]),
        )
        self.wait()

        # Try not to use wordle_words
        frame = self.camera.frame
        answer_rect = SurroundingRectangle(VGroup(titles[1], word_columns[1]))
        answer_rect.set_stroke(TEAL, 3)
        avoid = TexText("Let's try to avoid\\\\useing this")
        avoid.next_to(answer_rect, RIGHT)
        morty = Mortimer(height=2)
        morty.next_to(avoid, DOWN, MED_LARGE_BUFF)
        morty.change("hesitant", answer_rect)
        morty.save_state()
        morty.change("plain").set_opacity(0)

        self.play(
            frame.animate.match_x(word_columns, LEFT).shift(LEFT),
            ShowCreation(answer_rect),
            run_time=2,
        )
        self.play(
            Write(avoid),
            Restore(morty),
        )
        self.wait()

        # Common but not in wordle list
        priors = get_frequency_based_priors()
        not_in_answers = set(all_words).difference(answers)
        sorted_by_freq = list(sorted(not_in_answers, key=lambda w: priors[w]))
        n = 15
        most_common = self.get_grid_of_words(sorted_by_freq[-n:], n, 1)
        most_common.set_color(BLUE)
        most_common.move_to(morty.get_corner(UR), DOWN).shift(MED_SMALL_BUFF * UP)

        non_s_most_common = self.get_grid_of_words(
            list(filter(lambda w: w[-1] != 's', sorted_by_freq))[-n:], n, 1
        )
        non_s_most_common.match_style(most_common)
        non_s_most_common.replace(most_common)

        label = Text("Not in wordle list", font_size=36)
        label.next_to(most_common, UP)

        self.play(
            FadeOut(avoid, DOWN),
            morty.animate.change("raise_LEFT_hand", most_common),
            ShowIncreasingSubsets(most_common),
        )
        self.play(FadeIn(label))
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.animate.change("pondering", most_common),
            LaggedStartMap(FadeOut, most_common, shift=RIGHT),
            LaggedStartMap(FadeIn, non_s_most_common, shift=RIGHT),
        )
        self.wait()

    def show_pattern(self, *args, **kwargs):
        guess = self.pending_word_as_string()
        letters = self.grid.pending_word.copy()
        for letter, q_mark in zip(letters, self.q_marks):
            letter.replace(q_mark, dim_to_match=1)

        added_anims = []
        if guess == self.secret_word:
            added_anims.append(LaggedStart(
                *(
                    square.animate.set_fill(GREEN, 1)
                    for square in self.row_copy
                ),
                lag_ratio=0.7,
                run_time=2
            ))
            added_anims.append(LaggedStart(
                *(
                    Transform(q_mark, letter)
                    for q_mark, letter in zip(self.q_marks, letters)
                ),
                lag_ratio=0.7,
                run_time=2
            ))
        super().show_pattern(*args, added_anims=added_anims, **kwargs)


class ChoosingBasedOnLetterFrequencies(IntroduceGame):
    def construct(self):
        # Reconfigure grid to be flat
        grid = self.grid
        grid.set_submobjects([VGroup(*it.chain(*grid))])
        grid.add(grid.pending_word)
        self.add(grid)

        # Data on right
        letters_and_frequencies = [
            ("E", 13),
            ("T", 9.1),
            ("A", 8.2),
            ("O", 7.5),
            ("I", 7),
            ("N", 6.7),
            ("S", 6.3),
            ("H", 6.1),
            ("R", 6),
            ("D", 4.3),
            ("L", 4),
            ("U", 2.8),
            ("C", 2.8),
            ("M", 2.5),
            ("W", 2.4),
            ("F", 2.2),
            ("G", 2.0),
            ("Y", 2.0),
        ]
        freq_data = VGroup(*(
            VGroup(
                Text(letter, font="Consolas"),
                Rectangle(
                    height=0.25, width=0.2 * freq,
                    stroke_width=0,
                    fill_color=(BLUE if letter in "AEIOUY" else GREY_B),
                    fill_opacity=1,
                ),
                DecimalNumber(freq, num_decimal_places=1, font_size=24, unit="\\%")
            ).arrange(RIGHT)
            for letter, freq in letters_and_frequencies
        ))
        freq_data.arrange(DOWN, aligned_edge=LEFT)
        freq_data.set_height(FRAME_HEIGHT - 1)
        freq_data.to_edge(RIGHT, buff=LARGE_BUFF)
        self.freq_data = freq_data
        for row, lf in zip(freq_data, letters_and_frequencies):
            letter = lf[0]
            row.letter = letter
            row.rect = SurroundingRectangle(row, buff=SMALL_BUFF)
            row.rect.set_stroke(YELLOW, 0)
            row.add(row.rect)
        freq_data.add_updater(lambda m: m)
        self.add(freq_data)

    def add_letter(self, letter):
        super().add_letter(letter)
        self.update_freq_data_highlights()

    def delete_letter(self):
        super().delete_letter()
        self.update_freq_data_highlights()

    def update_freq_data_highlights(self):
        word = self.pending_word_as_string()
        for row in self.freq_data:
            if row.letter.lower() in word.lower():
                row.set_opacity(1)
                row.rect.set_fill(opacity=0)
                row.rect.set_stroke(width=2)
            else:
                row.set_opacity(0.5)
                row.rect.set_fill(opacity=0)
                row.rect.set_stroke(width=0)


class ExampleGridColors(WordleScene):
    grid_center = ChoosingBasedOnLetterFrequencies.grid_center
    secret_word = "baker"

    def construct(self):
        self.wait(3)

        grid = self.grid
        for guess in ["other", "nails"]:
            self.add_word(guess, 0)
            self.reveal_pattern()

        self.wait()

        for color in [BLACK, self.color_map[0]]:
            grid.generate_target()
            grid.target[:2].set_fill(color, 1),
            self.play(
                MoveToTarget(grid),
                lag_ratio=0.5,
                run_time=2
            )
        self.wait()

        self.embed()


class PreviewGamePlay(WordleSceneWithAnalysis):
    n_games = 10
    pre_computed_first_guesses = [
        "tares", "lares", "rates", "rales", "tears",
        "tales", "salet", "teras", "arles", "nares",
        "soare", "saner", "reals"
    ]

    def construct(self):
        self.show_guess_values()
        self.initial_guess_value_grid = self.guess_value_grid
        for x in range(self.n_games):
            self.clear()
            self.setup()
            self.secret_word = random.choice(get_word_list(short=True))
            self.guess_value_grid = self.initial_guess_value_grid
            self.add(self.guess_value_grid)
            while not self.has_won():
                guess = self.guess_value_grid[0][0].text
                self.add_word(guess)
                self.wait(0.5)
                self.reveal_pattern()


class UlteriorMotiveWrapper(VideoWrapper):
    title = "Ulterior motive: Lesson on entropy"


class IntroduceDistribution(WordleDistributions):
    secret_word = "stark"
    uniform_prior = True
    n_word_rows = 20
    n_bars_to_analyze = 3

    def construct(self):
        # Total labels
        total_label = self.get_total_words_label()
        self.add(total_label)

        # Show an example guess
        guess = "weary"
        match_label = self.get_dynamic_match_label()
        word_grid = self.get_grid_of_matches(n_rows=self.n_word_rows)

        self.wait()
        self.wait()
        self.play(ShowIncreasingSubsets(word_grid, run_time=3))
        self.wait(note=f"Write {guess}, (but don't submit)")
        if not self.presenter_mode or self.skip_animations:
            self.add_word(guess)

        # Show several possible patterns, with corresponding matches
        pattern_strs = ["20100", "01000"]
        prob_label = VGroup()
        for i, pattern_str in enumerate(pattern_strs):
            pattern = pattern_from_string(pattern_str)
            self.remove(match_label)
            self.show_pattern(pattern, animate=True)
            self.play(FadeOut(word_grid), FadeOut(prob_label))
            word_grid = self.get_grid_of_matches(n_rows=self.n_word_rows)
            self.add(match_label)
            match_label.update()
            self.play(
                CountInFrom(match_label[0], 0),
                ShowIncreasingSubsets(word_grid, run_time=2)
            )
            self.wait(note=f"Pattern {i} / {len(pattern_strs)}")

            num = match_label[0].get_value()
            denom = total_label[0].get_value()
            prob_label = self.get_dynamic_bar_label((
                "p\\left(", "0000", "\\right)", "=",
                "{" + "{:,}".format(num).replace(",", "{,}"), "\\over ",
                "{:,}".format(denom).replace(",", "{,}") + "}", "=",
                "0.0000",
            ))
            prob_label[-1].set_value(num / denom)
            prob_label.next_to(word_grid, UP)
            prob_label.clear_updaters()
            self.play(
                LaggedStart(
                    FadeTransform(match_label[0].copy().clear_updaters(), prob_label[4], remover=True),
                    FadeTransform(total_label[0].copy().clear_updaters(), prob_label[6], remover=True),
                    lag_ratio=0.5,
                ),
                FadeIn(VGroup(*prob_label[:4], prob_label[5], prob_label[7:])),
            )
            self.add(prob_label)
            self.wait()

        # Show distribution
        axes = self.get_axes(y_max=0.15)
        bars = self.get_distribution_bars(axes, guess)

        self.play(
            FadeOut(word_grid),
            FadeOut(prob_label),
        )
        self.play(LaggedStart(
            Write(axes),
            ShowIncreasingSubsets(bars),
            lag_ratio=0.5
        ))
        self.add(bars)
        self.wait()

        index = 20
        bar_indicator, x_tracker, get_bar = self.get_bar_indicator(bars, index)
        p_label = self.get_p_label(get_bar)

        self.add(bar_indicator, x_tracker)
        self.add(p_label)
        self.add(match_label)
        for x in range(self.n_bars_to_analyze):
            self.wait(note=f"Play around with probability {x} / {self.n_bars_to_analyze}")
            word_grid = self.get_grid_of_matches(n_rows=12, n_cols=5)
            word_grid.next_to(p_label, UP, LARGE_BUFF)
            self.play(ShowIncreasingSubsets(word_grid))
            self.wait()
            self.remove(word_grid)

        # Describe aim for expected information
        want = Text("What we want:")
        standin = Tex(
            "E[\\text{Information}] = \\sum_{x} p(x) \\cdot (\\text{Something})",
            tex_to_color_map={
                "\\text{Something}": GREY_B,
                "\\text{Information}": BLUE,
            },
            font_size=36,
        )
        group = VGroup(want, standin)
        group.arrange(DOWN, buff=MED_LARGE_BUFF)
        group.to_corner(UR)

        self.play(FadeIn(want))
        self.play(Write(standin))
        self.wait(note="Discuss adding up over all patterns")

        # Define information
        info_label = self.get_information_label(p_label, get_bar)
        il_copy = info_label.copy().clear_updaters()
        self.play(FadeIn(il_copy, UP, remover=True))
        self.add(info_label)
        self.wait(note="Give intuitions on values of I")

        # Define entropy
        entropy_definition = self.get_entropy_label()
        brace = Brace(entropy_definition, DOWN, buff=SMALL_BUFF)
        ent_label = TexText("Entropy, $H$")
        ent_label.set_color(BLUE)
        ent_label.next_to(brace, DOWN, SMALL_BUFF)

        self.play(
            FadeIn(entropy_definition, UP),
            FadeOut(standin, UP),
            FadeOut(want, UP),
        )
        self.wait(note="Drag tracker through full distributinon")
        self.wait()
        self.play(x_tracker.animate.set_value(10), run_time=3)
        self.play(
            GrowFromCenter(brace),
            Write(ent_label)
        )
        self.wait()

        # Show an alternate word
        self.remove(bar_indicator, x_tracker, p_label, info_label, bars, match_label)
        entropy_definition[-1].set_opacity(0)
        self.grid.clear_updaters()
        self.grid.add_updater(lambda m: m)
        self.get_curr_row().set_fill(BLACK, 0)
        self.grid.pending_pattern = None
        self.wait(note="Delete word, write \"slate\", but don't enter!")

        if not self.presenter_mode or self.skip_animations:
            for x in range(5):
                self.delete_letter()
            self.add_word('slate')

        guess = self.pending_word_as_string()
        bars = self.get_distribution_bars(axes, guess)

        self.play(ShowIncreasingSubsets(bars, run_time=3))
        self.wait()

        trackers = self.add_trackers(bars, index=20)
        self.wait(note="Play around more with distribiution")

        self.recalculate_entropy(entropy_definition, guess, trackers[0])
        self.wait()

        # Examples of good entropy
        self.remove(*trackers)
        bars.set_opacity(0.7)
        self.grid.clear_updaters()
        self.grid.add_updater(lambda m: m)
        self.grid.pending_word.set_submobjects([])
        self.add_word("?????")
        self.get_curr_row().set_fill(BLACK, 0)

        eqs = VGroup(
            Tex("E[I] = \\log_2\\left({1 \\over 1 / 3^5}\\right) = \\log_2(3^5) \\approx 7.92"),
            Tex("E[I] = \\log_2\\left({1 \\over 1 / 16}\\right) = \\log_2(16) = 4.00"),
            Tex("E[I] = \\log_2\\left({1 \\over 1 / 64}\\right) = \\log_2(64) = 6.00"),
        )
        eqs.scale(0.7)
        for eq in eqs:
            eq.next_to(ent_label, DOWN, LARGE_BUFF)

        prev_values = [bar.prob for bar in bars]

        last_eq = VGroup()
        ent_rhs = entropy_definition[-1]
        for eq, x in zip(eqs, [3**5, 16, 64]):
            values = [1 / x] * x + [0] * (3**5 - x)
            self.set_bars_to_values(bars, values, ent_rhs, added_anims=[FadeOut(last_eq)])
            self.wait()
            self.play(FadeIn(eq, UP))
            last_eq = eq
            self.wait()

        self.grid.pending_word.set_submobjects([])
        self.add_word(guess, wait_time_per_letter=0)
        self.set_bars_to_values(bars, prev_values, ent_rhs, added_anims=[FadeOut(last_eq)])
        self.wait()

        # Show the second guess
        true_pattern = self.get_pattern(guess)
        self.show_pattern(true_pattern, animate=True)
        trackers[0].set_value(list(bars.patterns).index(true_pattern))
        match_label.update()
        self.play(FadeIn(match_label))
        self.wait()

        self.play(
            ApplyMethod(
                match_label[0].copy().move_to,
                total_label[0], UR,
                remover=True,
            ),
            FadeOut(total_label[0], UP)
        )
        total_label[0].set_value(match_label[0].get_value())
        self.add(total_label)
        self.wait()

        faders = [axes, bars, match_label]
        for fader in faders:
            fader.clear_updaters()
        self.play(
            LaggedStart(*map(FadeOut, faders)),
            FadeOut(entropy_definition[-1]),
        )
        self.grid.clear_updaters()
        self.reveal_pattern(animate=False)
        next_guess = "stars"
        self.wait(note=f"Type in \"{next_guess}\"")
        if not self.presenter_mode or self.skip_animations:
            self.add_word(next_guess)

        # Show new distribution
        guess = self.pending_word_as_string()
        axes = self.get_axes(y_max=1, x_max=50)
        bars = self.get_distribution_bars(axes, guess)
        self.play(
            FadeIn(axes),
            FadeIn(bars, lag_ratio=0.1, run_time=2)
        )
        self.wait()

        self.remove(match_label)
        trackers = self.add_trackers(bars)
        self.wait(note="Play around with distribution")
        self.recalculate_entropy(entropy_definition, guess)

        # Better second guess
        self.grid.clear_updaters()
        self.get_curr_row().set_fill(BLACK, 0)
        self.pending_pattern = None
        entropy_definition[1].set_opacity(0)
        self.remove(bars, *trackers)
        guess = "print"
        self.wait(note=f"Write \"{guess}\"")
        if not self.presenter_mode or self.skip_animations:
            for x in range(5):
                self.delete_letter()
            self.add_word(guess)

        self.wait()
        bars = self.get_distribution_bars(axes, guess)
        self.play(FadeIn(bars, lag_ratio=0.1, run_time=2))
        self.wait()
        self.recalculate_entropy(entropy_definition, guess)
        self.wait()
        trackers = self.add_trackers(bars)
        self.wait(note="Play around with distribution")

    def set_bars_to_values(self, bars, values, ent_rhs, run_time=3, added_anims=[]):
        y_unit = self.axes.y_axis.unit_size
        bars.generate_target()
        bar_template = bars[0].copy()
        entropy = entropy_of_distributions(np.array(values))
        for bar, value in zip(bars.target, values):
            target = bar_template.copy()
            target.set_height(
                y_unit * value,
                stretch=True,
                about_edge=DOWN
            )
            target.move_to(bar, DOWN)
            bar.become(target)

        self.play(
            MoveToTarget(bars, run_time=run_time),
            ChangeDecimalToValue(ent_rhs, entropy, run_time=run_time),
            *added_anims
        )

    def recalculate_entropy(self, entropy_definition, guess, x_tracker=None):
        dec = entropy_definition[-1]
        dec.set_value(
            get_entropies(
                [guess], self.possibilities,
                get_weights(self.possibilities, self.priors)
            )[0]
        )
        dec.set_opacity(1)
        dec.next_to(entropy_definition[-2][-1])
        anims = [CountInFrom(dec, 0)]
        run_time = 1
        if x_tracker is not None:
            x_tracker.suspend_updating()
            anims.append(UpdateFromAlphaFunc(
                x_tracker, lambda m, a: m.set_value(a * 200),
                run_time=5,
            ))
            run_time = 3
        self.play(*anims, run_time=run_time)
        self.wait()

    def add_trackers(self, bars, index=1):
        bar_indicator, x_tracker, get_bar = self.get_bar_indicator(bars, index)
        p_label = self.get_p_label(get_bar, max_y=1)
        info_label = self.get_information_label(p_label, get_bar)
        match_label = self.get_dynamic_match_label()
        trackers = [x_tracker, bar_indicator, p_label, info_label, match_label]
        self.add(*trackers)
        return trackers


class ExpectedMatchesInsert(Scene):
    def construct(self):
        tex = Tex(
            "\\sum_{x} p(x) \\big(\\text{\\# Matches}\\big)",
            tex_to_color_map={"\\text{\\# Matches}": GREEN}
        )
        self.play(Write(tex))
        self.wait()


class DefineInformation(Scene):
    def construct(self):
        # Spaces
        pre_space = Square(side_length=3)
        pre_space.set_stroke(WHITE, 2)
        pre_space.set_fill(BLUE, 0.7)

        post_space = self.get_post_space(pre_space, 1)
        arrow = Vector(2 * RIGHT)
        group = VGroup(pre_space, arrow, post_space)
        group.arrange(RIGHT)

        # Labels
        kw = dict(font_size=36)
        pre_label = Text("Space of possibilities", **kw)
        pre_label.next_to(pre_space, UP, SMALL_BUFF)
        obs_label = Text("Observation", **kw)
        obs_label.next_to(arrow, UP)
        post_labels = self.get_post_space_labels(post_space, **kw)

        # 1 bit (has an s)
        self.add(pre_space)
        self.add(pre_label)
        self.wait()
        self.wait()
        self.play(
            ShowCreation(arrow),
            FadeIn(obs_label, lag_ratio=0.1),
            FadeTransform(pre_space.copy().set_fill(opacity=0), post_space),
            FadeIn(post_labels[1], 3 * RIGHT),
        )
        self.wait()
        self.play(Write(post_labels[0], run_time=1))
        self.wait()

        # Show all words
        n_rows = 25
        n_cols = 8
        all_words = get_word_list()
        words_sample = random.sample(all_words, n_rows * n_cols)
        word_grid = WordleScene.get_grid_of_words(words_sample, n_rows, n_cols)
        word_grid.replace(pre_space, dim_to_match=1)
        word_grid.scale(0.95)
        word_grid.shuffle()
        for word in word_grid:
            word.save_state()
        word_grid.scale(2)
        word_grid.set_opacity(0)

        has_s = TexText("Has an `s'", font_size=24)
        has_s.next_to(arrow, DOWN)

        self.play(LaggedStartMap(Restore, word_grid, lag_ratio=0.02, run_time=2))
        self.wait()
        self.play(FadeIn(has_s, 0.5 * DOWN))
        self.wait()

        # 2 bits (has a t)
        frame = self.camera.frame
        mini_group1 = self.get_mini_group(pre_space, arrow, post_space, post_labels)
        mini_group1.target.to_edge(UP, buff=0.25)
        post_space2 = self.get_post_space(pre_space, 2).move_to(post_space)
        post_labels2 = self.get_post_space_labels(post_space2, **kw)
        has_t = TexText("Has a `t'", font_size=24)
        has_t.next_to(arrow, DOWN, SMALL_BUFF)

        self.play(
            MoveToTarget(mini_group1),
            FadeOut(has_s),
            FadeOut(post_space),
            FadeOut(post_labels),
            frame.animate.move_to(2 * RIGHT)
        )
        self.play(
            FadeTransform(pre_space.copy(), post_space2),
            FadeIn(post_labels2, shift=3 * RIGHT)
        )
        self.wait()
        self.play(FadeIn(has_t, 0.5 * DOWN))
        self.wait()
        self.remove(has_t)

        # 3 through 5 bits
        last_posts = VGroup(post_space2, post_labels2)
        mini_groups = VGroup(mini_group1)
        for n in range(3, 7):
            new_mini = self.get_mini_group(pre_space, arrow, *last_posts)
            new_mini.target.next_to(mini_groups, DOWN, buff=0.5)
            new_post_space = self.get_post_space(pre_space, n)
            new_post_space.move_to(post_space)
            new_post_labels = self.get_post_space_labels(new_post_space, **kw)

            self.play(LaggedStart(
                MoveToTarget(new_mini),
                AnimationGroup(
                    FadeOut(last_posts),
                    FadeIn(new_post_space),
                    FadeIn(new_post_labels),
                ),
                lag_ratio=0.5
            ))
            self.wait()

            mini_groups.add(new_mini)
            last_posts = VGroup(new_post_space, new_post_labels)

        # Show formula
        group = VGroup(pre_space, pre_label, word_grid, arrow, obs_label, *last_posts)

        kw = dict(tex_to_color_map={"I": YELLOW})
        formulas = VGroup(
            Tex("\\left( \\frac{1}{2} \\right)^I = p", **kw),
            Tex("2^I = \\frac{1}{p}", **kw),
            Tex("I = \\log_2\\left(\\frac{1}{p}\\right)", **kw),
        )
        formulas.arrange(RIGHT, buff=LARGE_BUFF)
        formulas.to_edge(UP)
        formulas[1:].match_y(formulas[0][-1][0])

        formulas[0].save_state()
        formulas[0].move_to(formulas[1])
        self.play(
            FadeIn(formulas[0]),
            group.animate.to_edge(DOWN, buff=MED_SMALL_BUFF)
        )
        self.wait()
        self.play(Restore(formulas[0]))
        for i in (0, 1):
            self.play(TransformMatchingShapes(formulas[i].copy(), formulas[i + 1]))
            self.wait()

        rhs_rect = SurroundingRectangle(formulas[2])
        rhs_rect.set_stroke(YELLOW, 2)
        self.play(ShowCreation(rhs_rect))
        self.wait()

        # Ask why?
        randy = Randolph("confused", height=1.5)
        randy.next_to(rhs_rect, DL, MED_LARGE_BUFF)
        randy.look_at(rhs_rect)
        randy.save_state()
        randy.change("plain")
        randy.set_opacity(0)

        self.play(Restore(randy))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.animate.change("maybe"))
        self.play(Blink(randy))
        self.wait()

        # Readibility
        expr = Tex(
            "20 \\text{ bits} \\Leftrightarrow p \\approx 0.00000095",
            tex_to_color_map={"\\text{bits}": YELLOW}
        )
        expr.next_to(group, UP, buff=0.75)

        self.play(
            FadeOut(randy),
            Write(expr),
        )
        self.wait()

        # Additive
        group = group[:-2]
        self.play(
            FadeOut(expr),
            FadeOut(last_posts),
            group.animate.scale(0.7, about_edge=DL),
            FadeOut(mini_groups, RIGHT),
            frame.animate.move_to(RIGHT),
        )

        ps1 = self.get_post_space(pre_space, 2)
        ps2 = self.get_post_space(pre_space, 5)
        ps2.set_stroke(width=1)
        ps2.add(ps1.copy().fade(0.5))
        arrow2 = arrow.copy()
        ps1.next_to(arrow, RIGHT)
        arrow2.next_to(ps1, RIGHT)
        ps2.next_to(arrow2, RIGHT)

        ps1.label = self.get_post_space_labels(ps1, font_size=24)
        ps2.label = self.get_post_space_labels(
            self.get_post_space(pre_space, 3).replace(ps2),
            font_size=24
        )

        self.play(
            FadeTransform(pre_space.copy().set_opacity(0), ps1),
            FadeIn(ps1.label, 2 * RIGHT),
        )
        self.wait()
        self.play(
            FadeTransform(ps1.copy().set_opacity(0), ps2),
            FadeIn(ps2.label, 2 * RIGHT),
            ShowCreation(arrow2),
        )
        self.wait()

        brace = Brace(VGroup(ps1.label, ps2.label), UP)
        b_label = brace.get_text("5 bits").set_color(YELLOW)

        self.play(
            GrowFromCenter(brace),
            FadeIn(b_label, 0.2 * UP),
        )
        self.wait()

    def get_post_space(self, pre_space, n_bits):
        n_rows = 2**((n_bits // 2))
        n_cols = 2**((n_bits // 2) + n_bits % 2)
        result = pre_space.get_grid(n_rows, n_cols, buff=0)
        result.replace(pre_space, stretch=True)
        result[:-1].set_fill(opacity=0)
        return result

    def get_post_space_labels(self, post_space, **kw):
        n_bits = int(math.log2(len(post_space)))
        top_label = TexText("Information = ", f"${n_bits}$ bits", **kw)
        if n_bits == 1:
            top_label[-1][-1].set_opacity(0)
        top_label.next_to(post_space, UP, buff=0.15)
        top_label.set_color(YELLOW)
        bottom_label = TexText(f"$p = {{1 \\over {2**n_bits}}}$", **kw)
        bottom_label.next_to(post_space, DOWN, SMALL_BUFF)
        return VGroup(top_label, bottom_label)

    def get_mini_group(self, pre_space, arrow, post_space, post_labels):
        mini_group = VGroup(pre_space, arrow, post_space, post_labels[0]).copy()
        mini_group.generate_target()
        mini_group.target.scale(0.25)
        mini_group.target[-1][0].set_opacity(0)
        mini_group.target[-1][1].scale(3, about_edge=DOWN)
        mini_group.target[-1][1].match_x(mini_group.target[2])
        mini_group.target.next_to(post_space, RIGHT, buff=2.0)
        mini_group[::2].set_fill(opacity=0)
        mini_group.target[::2].set_stroke(width=1)
        return mini_group


class AskForFormulaForI(Scene):
    def construct(self):
        tex = Tex(
            "I = ???",
            tex_to_color_map={"I": YELLOW},
            font_size=72,
        )
        tex.to_edge(UP)
        self.play(Write(tex))
        self.wait()


class MinusLogExpression(Scene):
    def construct(self):
        tex = Tex(
            "I = -\\log_2(p)",
            tex_to_color_map={"I": YELLOW},
            font_size=60,
        )
        self.play(FadeIn(tex, DOWN))
        self.wait()


class LookTwoAheadWrapp(VideoWrapper):
    title = "Later..."


class ShowEntropyCalculations(IntroduceDistribution):
    grid_height = 3.5
    grid_center = [-5.0, -1.0, 0]
    CONFIG = {"random_seed": 0}
    n_words = 100

    def construct(self):
        # Axes
        grid = self.grid

        kw = dict(x_max=220, width=8.5, height=2.75)
        low_axes = self.get_axes(y_max=0.2, **kw)
        low_axes.to_edge(RIGHT, buff=0.1)
        low_axes.to_edge(UP, buff=0.5)
        y_label = Tex("p", font_size=24)
        y_label.next_to(low_axes.y_axis.n2p(0.2), RIGHT)
        low_axes.y_axis.add(y_label)
        self.add(low_axes)

        high_axes = self.get_axes(y_max=0.4, **kw)
        high_axes.next_to(low_axes, DOWN, buff=0.8)
        y_label = Tex("p \\cdot \\log_2(1/p)", font_size=24)
        y_label.next_to(high_axes.y_axis.n2p(0.4), RIGHT, MED_SMALL_BUFF)
        high_axes.y_axis.add(y_label)
        self.add(high_axes)

        # Formula
        ent_formula = self.get_entropy_label()
        ent_formula.scale(0.7)
        ent_formula.move_to(high_axes, UP)
        ent_formula.shift(DOWN)
        ent_rhs = ent_formula[1]
        self.add(ent_formula)
        n = 3**5

        # Bang on through
        words = list(random.sample(self.all_words, self.n_words))
        words = ["weary", "other", "tares", "kayak"] + words
        # words.sort()

        for word in words:
            low_bars = self.get_distribution_bars(low_axes, word)
            self.add(low_bars)

            dist = np.array([bar.prob for bar in low_bars])
            ent_summands = -np.log2(dist, where=dist > 1e-10) * dist
            high_bars = self.get_bars(high_axes, ent_summands)
            high_bars.add_updater(lambda m: m.match_style(low_bars))
            self.add(high_bars)

            self.add_word(word, wait_time_per_letter=0)
            trackers = self.add_trackers(low_bars, index=0)
            x_tracker, bar_indicator, p_label, info_label, match_label = trackers
            p_label.add_updater(lambda m: m.move_to(low_axes, DL).shift([2, 1, 0]))
            self.remove(info_label)
            self.remove(match_label)

            # Highlight answer
            arrow = Tex("\\rightarrow")
            pw = grid.pending_word.copy()
            pw.generate_target()
            pw.arrange(RIGHT, buff=0.05)
            rhs = ent_rhs.copy()
            rhs.set_value(sum(ent_summands))
            group = VGroup(pw, arrow, rhs)
            group.set_color(BLUE)
            group.arrange(RIGHT)
            group.match_width(grid)
            group.next_to(grid, UP, LARGE_BUFF)

            self.add(group)

            # Show calculation
            x_tracker.suspend_updating()
            n = list(dist).index(0) + 1
            self.play(
                UpdateFromAlphaFunc(
                    x_tracker, lambda m, a: m.set_value(int(a * (n - 1))),
                ),
                UpdateFromAlphaFunc(
                    ent_rhs, lambda m, a: m.set_value(sum(ent_summands[:int(a * n)]))
                ),
                rate_func=linear,
                run_time=4 * n / 3**5,
            )
            self.wait()
            # x_tracker.resume_updating()
            # self.embed()
            self.remove(group)

            # Clear
            self.remove(*trackers, low_bars, high_bars, pw, arrow, rhs)
            ent_rhs.set_value(0)
            grid.pending_word.set_submobjects([])
            grid.clear_updaters()
            grid.add_updater(lambda m: m)
            self.get_curr_row().set_fill(BLACK, 0)


class WrapperForEntropyCalculation(VideoWrapper):
    title = "Search for maximum entropy"


class UniformPriorExample(WordleSceneWithAnalysis):
    uniform_prior = True
    show_prior = False
    weight_to_prob = 0  # TODO
    pre_computed_first_guesses = [
        "tares", "lares", "rales", "rates", "teras",
        "nares", "soare", "tales", "reais", "tears",
        "arles", "tores", "salet",
    ]


class HowThePriorWorks(Scene):
    def construct(self):
        # Prepare columns
        all_words = get_word_list()
        freq_map = get_word_frequencies()
        sorted_words = list(sorted(all_words, key=lambda w: -freq_map[w]))

        col1, col2 = cols = [
            WordleScene.get_grid_of_words(
                word_list, 25, 1, dots_index=-12
            )
            for word_list in (random.sample(all_words, 100), sorted_words)
        ]
        for col in cols:
            col.set_height(6)
            col.set_x(-1)
            col.to_edge(DOWN, buff=MED_SMALL_BUFF)

        bars1, bars2 = [
            self.get_freq_bars(col, freq_map, max_width=width, exp=exp)
            for col, width, exp in zip(cols, (1, 2), (0.3, 1))
        ]

        group1 = VGroup(col1, bars1)
        group2 = VGroup(col2, bars2)

        col1_title = VGroup(
            Text("Relative frequencies of all words"),
            Text("From the Google Books English n-gram public dataset", font_size=24, color=GREY_B),
        )
        col1_title.arrange(DOWN)
        col1_title.set_height(1)
        col1_title.next_to(col1, UP, MED_LARGE_BUFF)

        # Introduce frequencies
        for bar in bars1:
            bar.save_state()
            bar.stretch(0, 0, about_edge=LEFT)
            bar.set_stroke(width=0)

        self.wait()
        self.add(col1)
        self.play(
            LaggedStartMap(Restore, bars1),
            FadeIn(col1_title, 0.5 * UP)
        )
        self.wait()

        arrow = Vector(2 * RIGHT, stroke_width=5)
        arrow.set_x(0).match_y(col1)
        arrow_label = Text("Sort", font_size=36)
        arrow_label.next_to(arrow, UP, SMALL_BUFF)

        self.play(
            ShowCreation(arrow),
            Write(arrow_label),
            group1.animate.next_to(arrow, LEFT)
        )
        group2.next_to(arrow, RIGHT, buff=LARGE_BUFF)
        self.play(LaggedStart(*(
            FadeInFromPoint(VGroup(word, bar), col1.get_center())
            for word, bar in zip(col2, bars2)
        ), lag_ratio=0.1, run_time=3))
        self.wait()

        # Word play
        numbers = VGroup(
            *(Integer(i + 1) for i in range(13))
        )
        numbers.match_height(col2[0])
        for number, word in zip(numbers, col2):
            number.next_to(word, LEFT, SMALL_BUFF, aligned_edge=UP)
            number.word = word
            number.add_updater(lambda m: m.match_style(m.word))

        rect = SurroundingRectangle(col2[:13], buff=0.05)
        rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            rect.animate.replace(col2[7], stretch=True).set_opacity(0),
            col2[7].animate.set_color(YELLOW),
            ShowIncreasingSubsets(numbers),
            run_time=0.5
        )
        self.wait()
        self.remove(rect)
        for i in [0, 1, 2, 8, 7, 6, 3, 4, (9, 10, 11, 12)]:
            col2.set_color(GREY_A)
            for j in listify(i):
                col2[j].set_color(YELLOW)
            self.wait(0.5)
        self.play(col2.animate.set_color(GREY_A))

        # Don't care about relative frequencies
        comp_words = ["which", "braid"]
        which_group, braid_group = comp = VGroup(*(
            VGroup(
                Text(word, font="Consolas"),
                Vector(RIGHT),
                DecimalNumber(freq_map[word], num_decimal_places=6)
            ).arrange(RIGHT)
            for word in comp_words
        ))
        comp.arrange(DOWN, buff=2.0)
        comp.to_edge(LEFT)

        percentages = DecimalNumber(99.9, num_decimal_places=1, unit="\\%").replicate(2)
        rhss = VGroup()
        for per, group in zip(percentages, comp):
            rhs = group[2]
            rhss.add(rhs)
            per.move_to(rhs, LEFT)
            rhs.generate_target()
            rhs.target.scale(0.8)
            rhs.target.set_color(GREY_B)
            rhs.target.next_to(per, DOWN, aligned_edge=LEFT)

        self.play(
            FadeOut(arrow),
            FadeOut(arrow_label),
            FadeOut(group1),
            FadeTransform(col2[0].copy(), which_group[0]),
        )
        self.play(
            ShowCreation(which_group[1]),
            CountInFrom(which_group[2], 0),
        )
        self.wait()
        self.play(FadeTransform(which_group[:2].copy(), braid_group[:2]))
        self.play(CountInFrom(braid_group[2], 0, run_time=0.5))
        self.wait()
        self.play(
            FadeIn(percentages, 0.75 * DOWN),
            *map(MoveToTarget, rhss),
        )
        self.wait()

        # Sigmoid
        axes = Axes((-10, 10), (0, 2, 0.25), width=12, height=6)
        axes.y_axis.add_numbers(np.arange(0.25, 2.25, 0.25), num_decimal_places=2, font_size=18)
        axes.center()

        col3 = WordleScene.get_grid_of_words(sorted_words, 25, 4, dots_index=-50)
        col3.arrange(DOWN, buff=SMALL_BUFF)
        col3.generate_target()
        col3.target.rotate(-90 * DEGREES)
        col3.target.match_width(axes.x_axis)
        col3.target.next_to(axes.x_axis, DOWN, buff=0)

        col2_words = [w.text for w in col2]
        col3.match_width(col2)
        for word in col3:
            if word.text in col2_words:
                word.move_to(col2[col2_words.index(word.text)])
            else:
                word.rotate(-90 * DEGREES)
                word.move_to(col2[col2_words.index('...')])
                word.scale(0)
                word.set_opacity(0)

        self.remove(col2),
        self.play(LaggedStart(
            FadeOut(VGroup(comp, percentages), 2 * LEFT),
            FadeOut(numbers),
            FadeOut(bars2),
            FadeOut(col1_title, UP),
            MoveToTarget(col3),
            Write(axes),
            FadeOut(col2[col2_words.index("...")])
        ))
        self.wait()

        graph = axes.get_graph(sigmoid)
        graph.set_stroke(BLUE, 3)
        graph_label = Tex("\\sigma(x) = {1 \\over 1 + e^{-x} }")
        graph_label.next_to(graph.get_end(), UL)

        self.play(ShowCreation(graph))
        self.play(Write(graph_label))
        self.wait()

        # Lines to graph
        lines = Line().replicate(len(col3))
        lines.set_stroke(BLUE_B, 1.0)

        def update_lines(lines):
            for line, word in zip(lines, col3):
                line.put_start_and_end_on(
                    word.get_top(),
                    axes.input_to_graph_point(axes.x_axis.p2n(word.get_center()), graph),
                )

        lines.add_updater(update_lines)
        self.play(ShowCreation(lines, lag_ratio=0.05, run_time=5))
        self.wait()

        self.play(col3.animate.scale(0.5, about_edge=UP), run_time=3)
        self.play(col3.animate.scale(2.0, about_edge=UP), run_time=3)
        self.wait()
        for vect in [RIGHT, 2 * LEFT, RIGHT]:
            self.play(col3.animate.shift(vect), run_time=2)
        lines.clear_updaters()
        self.wait()

        # Show window of words
        n_shown = 15
        col4 = WordleScene.get_grid_of_words(
            sorted_words[3000:3000 + n_shown], 20, 1
        )
        dots = Text("...", font="Consolas", font_size=24).rotate(90 * DEGREES)
        col4.add_to_back(dots.copy().next_to(col4, UP))
        col4.add(dots.copy().next_to(col4, DOWN))
        col4.set_height(6)
        col4.to_corner(UL)
        col4.shift(RIGHT)

        numbers = VGroup(*(Integer(n) for n in range(3000, 3000 + n_shown)))
        numbers.set_height(col4[1].get_height())
        for number, word in zip(numbers, col4[1:]):
            number.next_to(word, LEFT, MED_SMALL_BUFF, aligned_edge=UP)
            number.match_style(word)
            number.align_to(numbers[0], LEFT)
            word.add(number)

        self.play(ShowIncreasingSubsets(col4))
        self.wait()

    def get_freq_bars(self, words, freq_map, max_width=2, exp=1):
        freqs = [freq_map.get(w.text, 0)**exp for w in words]  # Smoothed out a bit
        max_freq = max(freqs)
        bars = VGroup()
        height = np.mean([w.get_height() for w in words]) * 0.8
        for word, freq in zip(words, freqs):
            bar = Rectangle(
                height=height,
                width=max_width * freq / max_freq,
                stroke_color=WHITE,
                stroke_width=1,
                fill_color=BLUE,
                fill_opacity=1,
            )
            bar.next_to(word, RIGHT, SMALL_BUFF)
            if word.text not in freq_map:
                bar.set_opacity(0)
            bars.add(bar)
        return bars


class EntropyOfWordDistributionExample(WordleScene):
    grid_height = 4
    grid_center = 4.5 * LEFT
    secret_word = "graph"
    wordle_based_prior = True

    def construct(self):
        # Try first two guesses
        grid = self.grid
        guesses = ["other", "nails"]
        if not self.presenter_mode or self.skip_animations:
            self.add_word("other")
            self.reveal_pattern()
            self.add_word("nails")
            self.reveal_pattern()
        else:
            self.wait()
            self.wait("Enter \"{}\" then \"{}\"".format(*guesses))

        # Add match label
        match_label = VGroup(Integer(4, edge_to_fix=RIGHT), Text("Matches"))
        match_label.scale(0.75)
        match_label.arrange(RIGHT, buff=MED_SMALL_BUFF)
        match_label.next_to(grid, UP)
        self.add(match_label)

        # Show words
        s_words = get_word_list(short=True)
        col1 = self.get_grid_of_words(
            sorted(list(set(self.possibilities).intersection(s_words))),
            4, 1
        )
        col1.scale(1.5)
        col1.next_to(grid, RIGHT, buff=1)
        bars1 = VGroup(*(
            self.get_prob_bar(word, 0.25)
            for word in col1
        ))
        for bar in bars1:
            bar.save_state()
            bar.stretch(0, 0, about_edge=LEFT)
            bar.set_opacity(0)

        self.play(
            ShowIncreasingSubsets(col1),
            CountInFrom(match_label[0], 0),
        )
        self.wait()
        self.play(LaggedStartMap(Restore, bars1))
        self.wait()

        # Ask about entropy
        brace = Brace(bars1, RIGHT)
        question = Text("What is the\nentropy?", font_size=36)
        question.next_to(brace, RIGHT)

        formula = Tex(
            "H &=",
            "\\sum_x p(x) \\cdot", "\\log_2\\big(1 / p(x) \\big)\\\\",
            font_size=36,
        )
        formula.next_to(brace, RIGHT, submobject_to_align=formula[0])

        info_box = SurroundingRectangle(formula[2], buff=SMALL_BUFF)
        info_box.set_stroke(TEAL, 2)
        info_label = Text("Information", font_size=36)
        info_label.next_to(info_box, UP)
        info_label.match_color(info_box)
        info_value = Tex("\\log_2(4)", "=", "2", font_size=36)
        info_value[1].rotate(PI / 2)
        info_value.arrange(DOWN, SMALL_BUFF)
        info_value.next_to(info_box, DOWN)
        alt_lhs = formula[0].copy().next_to(info_value[-1], LEFT)

        self.play(
            GrowFromCenter(brace),
            Write(question),
        )
        self.wait()
        self.play(
            FadeIn(formula, lag_ratio=0.1),
            question.animate.shift(2 * UP)
        )
        self.wait()

        self.play(
            ShowCreation(info_box),
            Write(info_label)
        )
        self.wait()

        self.play(FadeIn(info_value[0]))
        self.wait()
        self.play(Write(info_value[1:]))
        self.wait()
        self.play(TransformFromCopy(formula[0], alt_lhs))
        self.wait()

        # Introduce remaining words
        col2 = self.get_grid_of_words(
            sorted(self.possibilities), 16, 1
        )
        col2.match_width(col1)
        col2.move_to(col1, LEFT)
        col2.save_state()
        col1_words = [w.text for w in col1]
        for word in col2:
            if word.text in col1_words:
                word.move_to(col1[col1_words.index(word.text)])
            else:
                word.move_to(col1)
                word.set_opacity(0)

        pre_bars2, bars2 = [
            VGroup(*(
                self.get_prob_bar(
                    word,
                    0.246 * self.priors[word.text] + 0.001,
                    num_decimal_places=3,
                )
                for word in group
            ))
            for group in (col2, col2.saved_state)
        ]

        new_brace = Brace(bars2, RIGHT)

        self.play(
            FadeTransform(col1, col2),
            FadeTransform(bars1, pre_bars2),
            LaggedStart(*map(FadeOut, [alt_lhs, info_value, info_label, info_box]))
        )
        self.play(
            ChangeDecimalToValue(match_label[0], 16, run_time=1),
            Restore(col2, run_time=2),
            ReplacementTransform(pre_bars2, bars2, run_time=2),
            Transform(brace, new_brace),
        )
        self.wait()

        # Proposed answer
        rhs1 = Tex("= \\log_2(16) = 4?", font_size=36)
        rhs1.next_to(formula[1], DOWN, aligned_edge=LEFT)
        cross = Cross(rhs1).set_stroke(RED, 6)
        rhs2 = Tex(
            "= &4 \\big(0.247 \\cdot \\log_2(1/0.247)\\big) \\\\",
            "+ &12 \\big(0.001 \\cdot \\log_2(1/0.001)\\big)\\\\ ",
            "= &2.11",
            font_size=30,
        )
        rhs2.next_to(rhs1, DOWN, aligned_edge=LEFT)

        self.play(Write(rhs1))
        self.wait()
        self.play(ShowCreation(cross))
        self.wait()

        self.play(
            Write(rhs2),
            run_time=3
        )
        self.wait()

        rect = SurroundingRectangle(rhs2[-1])
        self.play(ShowCreation(rect))
        self.wait()

    def get_prob_bar(self, word, prob, num_decimal_places=2, height=0.15, width_mult=8.0):
        bar = Rectangle(
            height=height,
            width=width_mult * prob,
            stroke_color=WHITE,
            stroke_width=1,
            fill_color=BLUE,
        )
        bar.next_to(word, RIGHT, MED_SMALL_BUFF)
        label = DecimalNumber(prob, font_size=24, num_decimal_places=num_decimal_places)
        label.next_to(bar, RIGHT, SMALL_BUFF)
        bar.add(label)
        bar.label = label
        bar.set_opacity(word[0].get_fill_opacity())
        return bar

    def seek_good_examples(self):
        words = get_word_list()
        swords = get_word_list(short=True)
        for answer in swords:
            poss = list(words)
            for guess in ["other", "nails"]:
                poss = get_possible_words(
                    guess,
                    get_pattern(guess, answer),
                    poss,
                )
            n = len(set(poss).intersection(swords))
            m = len(poss)
            if n == 4 and m in (16, 32, 64):
                print(answer, n, len(poss))


class TwoInterpretationsWrapper(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        screens = ScreenRectangle().get_grid(1, 2, buff=MED_LARGE_BUFF)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 1)
        screens.set_width(FRAME_WIDTH - 1)
        screens.move_to(DOWN)

        title = Text("Two applications of entropy", font_size=60)
        title.to_edge(UP)

        screen_titles = VGroup(
            Text("Expected information from guess"),
            Text("Remaining uncertainty"),
        )
        screen_titles.scale(0.8)
        for screen, word in zip(screens, screen_titles):
            word.next_to(screen, UP)

        screen_titles[0].set_color(BLUE)
        screen_titles[1].set_color(TEAL)

        self.add(title)
        self.add(screens)
        self.wait()
        for word in screen_titles:
            self.play(Write(word, run_time=1))
            self.wait()


class IntroduceDistributionFreqPrior(IntroduceDistribution):
    n_word_rows = 1
    uniform_prior = False
    show_fraction_in_p_label = False


class FreqPriorExample(WordleSceneWithAnalysis):
    pre_computed_first_guesses = [
        "tares", "lares", "rates", "rales", "tears",
        "tales", "salet", "teras", "arles", "nares",
        "soare", "saner", "reals"
    ]


class ConstrastResultsWrapper(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        screens = Rectangle(4, 3).replicate(2)
        screens.arrange(RIGHT, buff=SMALL_BUFF)
        screens.set_width(FRAME_WIDTH - 1)
        screens.set_stroke(WHITE, 1)
        screens.set_fill(BLACK, 1)
        screens.move_to(DOWN)
        self.add(screens)


class WordlePriorExample(WordleSceneWithAnalysis):
    secret_word = "thump"
    wordle_based_prior = True
    pre_computed_first_guesses = [
        "soare", "raise", "roate", "raile", "reast",
        "slate", "crate", "irate", "trace", "salet",
        "arise", "orate", "stare"
    ]


class HowToCombineEntropyAndProbability(WordleSceneWithAnalysis):
    secret_word = None

    def construct(self):
        self.embed()

        grid_cover = SurroundingRectangle(VGroup(
            self.guess_value_grid_titles,
            self.guess_value_grid
        ))
        grid_cover.set_fill(BLACK, 1)
        grid_cover.set_stroke(BLACK)
        self.add(grid_cover)

        guesses = ["study", "ample"]
        self.wait()
        self.wait(note="Type in {guesses}")
        if not self.presenter_mode or self.skip_animations:
            for guess in guesses:
                self.add_word(guess)
                self.reveal_pattern()


        self.add_word("maths")
        # 

        # Embed
        self.embed()


class LookTwoStepsAhead(WordleSceneWithAnalysis):
    look_two_ahead = True
    wordle_based_prior = True
    pre_computed_first_guesses = [
        "slate", "salet", "slane", "reast", "trace",
        "carse", "crate", "torse", "carle", "carte",
        "toile", "crane", "least", "saint", "crine",
        "roast",
    ]


class HowLookTwoAheadWorks(Scene):
    prob_color = BLUE_D
    entropy_color = TEAL
    first_guess = "tares"
    n_shown_trials = 240

    def get_priors(self):
        return get_frequency_based_priors()

    def construct(self):
        # Setup
        all_words = get_word_list()
        possibilities = get_word_list()
        priors = self.get_priors()

        # Show first guess
        guess1 = self.get_word_mob(self.first_guess)
        guess1.to_edge(LEFT)
        pattern_array1 = self.get_pattern_array(guess1, possibilities, priors)
        prob_bars1 = self.get_prob_bars(pattern_array1.pattern_mobs)

        self.add(guess1)
        self.play(
            ShowCreation(
                pattern_array1.connecting_lines,
                lag_ratio=0.1
            ),
            LaggedStartMap(
                FadeIn, pattern_array1.pattern_mobs,
                shift=0.2 * RIGHT,
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        self.play(Write(pattern_array1.dot_parts))
        self.wait()
        for bar in prob_bars1:
            bar.save_state()
            bar.stretch(0, 0, about_edge=LEFT)
            bar.set_opacity(0)
        self.play(LaggedStartMap(Restore, prob_bars1))
        self.wait()

        # Reminder on entropy
        H_eq = Tex(
            "H = E[I] = \\sum_{x} p(x) \\cdot \\log_2\\big((1 / p(x)\\big)",
            font_size=36
        )
        H_eq.next_to(prob_bars1, RIGHT)

        info_labels = VGroup(*(
            self.get_info_label(bar)
            for bar in prob_bars1
        ))

        self.play(Write(H_eq))
        self.wait()
        self.play(FadeIn(info_labels[0], lag_ratio=0.1))
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, info_labels[1:], lag_ratio=0.5),
            H_eq.animate.scale(0.7).to_edge(DOWN),
            run_time=2,
        )
        self.wait()

        H_label = self.get_entropy_label(guess1, pattern_array1.distribution)
        self.play(FadeTransform(H_eq, H_label))
        self.wait()
        self.play(LaggedStartMap(FadeOut, info_labels), run_time=1)

        # Show example second guess
        word_buckets = get_word_buckets(guess1.text, possibilities)

        arrows = VGroup()
        second_guesses = VGroup()
        second_ents = VGroup()
        for i, bar in enumerate(prob_bars1):
            pattern = pattern_array1.pattern_mobs[i].pattern
            bucket = word_buckets[pattern]
            optimal_word = optimal_guess(all_words, bucket, priors)
            shown_words = random.sample(all_words, self.n_shown_trials)
            shown_words.append(optimal_word)
            for j, shown_word in enumerate(shown_words):
                guess2 = self.get_word_mob(shown_word)
                guess2.set_width(0.7)
                guess2.match_y(bar)
                guess2.set_x(0, LEFT)
                arrow = Arrow(bar.label, guess2, stroke_width=3, buff=SMALL_BUFF)
                pattern_array2 = self.get_pattern_array(guess2, bucket, priors, n_shown=25)
                prob_bars2 = self.get_prob_bars(pattern_array2.pattern_mobs, width_scalar=5)
                h2_label = self.get_entropy_label(guess2, pattern_array2.distribution)

                group = VGroup(
                    arrow, guess2, h2_label, pattern_array2, prob_bars2,
                )
                self.add(group, second_ents)
                self.wait(1 / self.camera.frame_rate, ignore_presenter_mode=True)
                if i in (0, 1) and j == 0:
                    self.wait()
                self.remove(group)
            self.add(*group, second_ents)
            self.wait()

            # Consolidate
            arrow, guess2, h2_label, pattern_array2, prob_bars2 = group
            for line in pattern_array2.connecting_lines:
                line.reverse_points()
            h2_label.generate_target()
            h2_label.target.scale(0.8)
            h2_label.target.next_to(guess2, RIGHT)
            guess2.set_color(YELLOW)
            self.add(pattern_array2.connecting_lines, second_ents)
            self.play(
                MoveToTarget(h2_label),
                Uncreate(pattern_array2.connecting_lines, lag_ratio=0.01),
                LaggedStartMap(FadeOut, pattern_array2.pattern_mobs),
                LaggedStartMap(FadeOut, pattern_array2.dot_parts),
                LaggedStartMap(FadeOut, prob_bars2, scale=0.25),
                run_time=1
            )
            arrows.add(arrow)
            second_guesses.add(guess2)
            second_ents.add(h2_label)

        # Show weighted sum
        brace = Brace(VGroup(second_ents, pattern_array1), RIGHT)
        label = brace.get_text("Compute a\nweighted average", buff=MED_SMALL_BUFF)

        sum_parts = VGroup()
        for bar, h_label in zip(prob_bars1, second_ents):
            d0 = DecimalNumber(bar.prob, num_decimal_places=3)
            d1 = h_label[1].copy()
            d0.match_height(d1)
            group = VGroup(d0, Tex("\\cdot", font_size=24), d1)
            group.generate_target()
            group.target.arrange(RIGHT, buff=SMALL_BUFF)
            group.target.next_to(brace, RIGHT)
            group.target.match_y(bar)
            sum_parts.add(group)
            for part in group[:2]:
                part.move_to(bar.label)
                part.set_opacity(0)

        self.play(
            GrowFromCenter(brace),
            Write(label)
        )
        self.wait()
        self.play(
            LaggedStartMap(MoveToTarget, sum_parts, run_time=2),
            label.animate.scale(0.7).to_edge(DOWN),
        )
        self.wait()

    def get_word_mob(self, word):
        return Text(word, font="Consolas", font_size=36)

    def get_pattern_array(self, word, possibilities, priors, n_shown=15):
        weights = get_weights(possibilities, priors)
        dist = get_pattern_distributions([word.text], possibilities, weights)[0]
        indices = np.argsort(dist)[::-1]
        patterns = np.arange(3**5)[indices]
        patterns = patterns[:n_shown]  # Only show non-zero possibilities

        top_parts = VGroup(*(self.get_pattern_mob(p) for p in patterns[:n_shown]))
        dot_parts = Tex("\\vdots\\\\", "3^5 \\text{ patterns}\\\\", "\\vdots")

        for prob, row in zip(dist[indices][:n_shown], top_parts):
            row.prob = prob

        stack = VGroup(*top_parts, *dot_parts)
        dot_parts.match_width(stack[0])
        stack.arrange(DOWN, buff=SMALL_BUFF)
        stack.set_max_height(FRAME_HEIGHT - 1)
        stack.next_to(word, RIGHT, buff=1.5)
        # stack.set_y(0)
        stack.shift_onto_screen(buff=MED_LARGE_BUFF)

        pattern_mobs = top_parts
        connecting_lines = VGroup(*(
            self.get_connecting_line(word, row)
            for row in pattern_mobs
        ))

        result = VGroup(pattern_mobs, dot_parts, connecting_lines)
        result.pattern_mobs = pattern_mobs
        result.dot_parts = dot_parts
        result.connecting_lines = connecting_lines
        result.distribution = dist

        return result

    def get_pattern_mob(self, pattern, width=1.5):
        result = Square().replicate(5)
        result.arrange(RIGHT, buff=SMALL_BUFF)
        result.set_stroke(WHITE, width=0.5)
        for square, n in zip(result, pattern_to_int_list(pattern)):
            square.set_fill(WordleScene.color_map[n], 1)
        result.set_width(width)
        result.pattern = pattern
        return result

    def get_connecting_line(self, mob1, mob2):
        diff = mob2.get_left()[0] - mob1.get_right()[0]
        return CubicBezier(
            mob1.get_right() + SMALL_BUFF * RIGHT,
            mob1.get_right() + RIGHT * diff / 2,
            mob2.get_left() + LEFT * diff / 2,
            mob2.get_left() + SMALL_BUFF * LEFT,
            stroke_color=WHITE,
            stroke_width=1,
        )

    def get_prob_bars(self, pattern_mobs, width_scalar=10):
        result = VGroup()
        for pattern_mob in pattern_mobs:
            bar = Rectangle(
                width=width_scalar * pattern_mob.prob,
                height=pattern_mob.get_height(),
                fill_color=self.prob_color,
                fill_opacity=1,
                stroke_width=0.5,
                stroke_color=WHITE,
            )
            bar.next_to(pattern_mob, RIGHT, buff=SMALL_BUFF)
            label = DecimalNumber(100 * pattern_mob.prob, num_decimal_places=1, unit="\\%")
            # label = DecimalNumber(pattern_mob.prob, num_decimal_places=3)
            label.set_height(bar.get_height() * 0.6)
            label.next_to(bar, RIGHT, SMALL_BUFF)
            bar.label = label
            bar.add(label)
            bar.prob = pattern_mob.prob
            result.add(bar)
        return result

    def get_entropy_label(self, word_mob, distribution):
        ent2 = entropy_of_distributions(distribution)
        kw = dict(font_size=24)
        h_label = VGroup(Tex(f"H = ", **kw), DecimalNumber(ent2, **kw))
        h_label.set_color(self.entropy_color)
        h_label.arrange(RIGHT, buff=SMALL_BUFF, aligned_edge=UP)
        h_label.move_to(word_mob)
        h_label.shift(0.5 * DOWN)
        h_label.set_backstroke(width=8)
        return h_label

    def get_info_label(self, bar):
        result = VGroup(
            DecimalNumber(bar.prob, num_decimal_places=3),
            Tex("\\cdot \\log_2\\big( 1 / "),
            DecimalNumber(bar.prob, num_decimal_places=3),
            Tex("\\big) = "),
            DecimalNumber(-bar.prob * math.log2(bar.prob), num_decimal_places=3)
        )
        result.arrange(RIGHT, buff=SMALL_BUFF)
        result.set_height(bar.get_height())
        result.match_y(bar)
        result.set_x(0, LEFT)
        arrow = Arrow(bar.label.get_right(), result, stroke_width=2, buff=SMALL_BUFF)
        result.add_to_back(arrow)
        return result


class TwoStepLookAheadWithCrane(HowLookTwoAheadWorks):
    first_guess = "crane"
    n_shown_trials = 60

    def get_priors(self):
        return get_true_wordle_prior()


class BestDoubleEntropies(Scene):
    def construct(self):
        pass
        # Facts on theoretical possibilities:
        # Best two-step entropy is slane: 5.7702 + 4.2435 = 10.014
        # Given the start, with log2(2315) = 11.177 bits of entropy,
        # this means an average uncertainty of 1.163.
        # This is akin to being down to 2.239 words
        # In that case, there's a 1/2.239 = 0.4466 chance of getting it in 3
        # Otherwise, 0.5534 chance of requiring at least 4
        #
        # Assuming best case scenarios, that out of the 2315 answers, you get:
        # - 1 in 1
        # - 273 in 2 with your encoded second guesses
        # - Of the remaining 2041, you get 0.4466 * 2041 = 912 in 3
        # - Of the remaining 1,129, all are in 4
        # Average: (1 + 2 * 273 + 3 * 912 + 4 * 1129) / 2315 = 3.368
        #
        # But actually, number of 2's is (at most) 150, so we could update to:
        # Average: (1 + 2 * 150 + 3 * 967 + 4 * 1197) / 2315 = 3.451
        # More general formula
        # (1 + 2 * n + 3 * 0.4466 * (2315 - n - 1) + 4 * 0.5534 * (2315 - n - 1)) / 2315
        #
        # Analyzing crane games, it looks like indeed, the average uncertainty
        # at the third step is 1.2229, just slightly higher than the 1.163 above.
        # In fact, for 'crane' the average shoudl be 11.177 - 9.9685 = 1.208
        #
        # game_data = json.load(open("/Users/grant/Dropbox/3Blue1Brown/data/wordle/crane_with_wordle_prior.json"))
        # games = game_data["game_results"]
        # reductions = [g['reductions'] for g in games]
        # step3_state = [red[1] if len(red) > 1 else 1 for red in reductions]
        # step3_bits = [math.log2(x) for x in step3_state]
        # np.mean(step3_bits)
        # Out: 1.2229


# Distribution animations

class ShowScoreDistribution(Scene):
    data_file = "crane_with_wordle_prior.json"
    axes_config = dict(
        x_range=(0, 9),
        y_range=(0, 1, 0.1),
        width=8,
        height=6,
    )
    weighted_sample = False

    def construct(self):
        axes = self.get_axes()
        self.add(axes)

        with open(os.path.join(DATA_DIR, self.data_file)) as fp:
            game_data = json.load(fp)
        games = game_data["game_results"]
        scores = [game["score"] for game in games]

        bars = self.get_bars(axes, scores[:0])

        mean_label = VGroup(
            Text("Average score: "),
            DecimalNumber(np.mean(scores), num_decimal_places=3)
        )
        mean_label.arrange(RIGHT, aligned_edge=UP)
        mean_label.move_to(axes, UP)
        self.add(mean_label)

        grid = WordleScene.patterns_to_squares(6 * [0])
        grid.set_fill(BLACK, 0)
        grid.set_width(2)
        grid.move_to(axes, RIGHT)
        grid.words = VGroup()
        grid.add(grid.words)
        self.add(grid)

        score_label = VGroup(
            Text("Score: "),
            Integer(0, edge_to_fix=LEFT),
        )
        score_label.scale(0.75)
        score_label.arrange(RIGHT, aligned_edge=DOWN)
        score_label.next_to(grid, UP)
        self.add(score_label)

        def a2n(alpha):
            return integer_interpolate(0, len(scores), alpha)[0]

        def update_bars(bars, alpha):
            bars.set_submobjects(self.get_bars(axes, scores[:a2n(alpha)]))

        def update_mean_label(label, alpha):
            label[1].set_value(np.mean(scores[:max(1, a2n(alpha))]))

        def update_grid(grid, alpha):
            game = games[a2n(alpha)]
            patterns = game["patterns"]
            patterns.append(3**5 - 1)
            grid.set_fill(BLACK, 0)
            for pattern, row in zip(patterns, grid):
                for square, key in zip(row, pattern_to_int_list(pattern)):
                    square.set_fill(WordleScene.color_map[key], 1)
            grid.words.set_submobjects([
                Text(guess.upper(), font="Consolas")
                for guess in (*game["guesses"], game["answer"])
            ])
            for word, row in zip(grid.words, grid):
                word.set_height(row.get_height() * 0.6)
                for char, square in zip(word, row):
                    char.move_to(square)

        def update_score_label(score_label, alpha):
            score = games[a2n(alpha)]["score"]
            score_label[1].set_value(score)

        self.play(
            UpdateFromAlphaFunc(bars, update_bars),
            UpdateFromAlphaFunc(mean_label, update_mean_label),
            UpdateFromAlphaFunc(grid, update_grid),
            UpdateFromAlphaFunc(score_label, update_score_label),
            run_time=20,
            rate_func=linear,
        )
        self.wait()

    def get_axes(self):
        axes = Axes(**self.axes_config)
        x_axis, y_axis = axes.x_axis, axes.y_axis
        y_axis.add_numbers(num_decimal_places=1)
        x_axis.add_numbers()
        x_axis.numbers.shift(x_axis.unit_size * LEFT / 2)
        x_label = Text("Score", font_size=24)
        x_label.next_to(x_axis.get_end(), RIGHT)
        x_axis.add(x_label)
        return axes

    def get_bars(self, axes, scores):
        scores = np.array(scores)
        buckets = np.array([
            (scores == n + 1).sum()
            for n in np.arange(*axes.x_range)
        ])
        props = buckets / buckets.sum()
        bars = VGroup(*(
            self.get_bar(axes, n + 1, prop)
            for n, prop in enumerate(props)
        ))
        bars.set_submobject_colors_by_gradient(BLUE, YELLOW, RED)
        bars.set_stroke(WHITE, 1)
        for bar, count in zip(bars, buckets):
            bar.add(self.get_bar_count(bar, count))
        return VGroup(bars)

    def get_bar(self, axes, score, proportion):
        bar = Rectangle(
            width=axes.x_axis.unit_size,
            height=axes.y_axis.unit_size * proportion,
        )
        bar.set_fill(BLUE, 1)
        bar.set_stroke(WHITE, 1)
        bar.move_to(axes.c2p(score, 0), DR)
        return bar

    def get_bar_count(self, bar, count):
        result = Integer(count, font_size=30)
        result.next_to(bar, UP, SMALL_BUFF)
        if count == 0:
            result.set_opacity(0)
        return result


class SimulatedGamesUniformPriorDist(ShowScoreDistribution):
    data_file = "tares_with_uniform_prior.json"


class SimulatedGamesFreqBasedPriorDist(ShowScoreDistribution):
    data_file = "tares_with_freq_prior.json"


class SimulatedGamesWordleBasedPriorDist(ShowScoreDistribution):
    data_file = "soare_with_wordle_prior.json"


class SimulatedGamesWordleBasedPriorCraneStartDist(ShowScoreDistribution):
    data_file = "crane_with_wordle_prior.json"


class SimulatedGamesWordleBasedPriorExcludeSeenWordsDist(ShowScoreDistribution):
    data_file = "crane_with_wordle_prior_exclude_seen.json"


class SimulatedGamesFreqBasedPriorExcludeSeenWordsDist(ShowScoreDistribution):
    data_file = "tares_with_freq_prior_exclude_seen.json"


# Thumbnail

class Thumbnail(Scene):
    def construct(self):
        answer = "aging"
        guesses = ["crane", "tousy", answer]
        patterns = [get_pattern(guess, answer) for guess in guesses]

        rows = WordleScene.patterns_to_squares(
            patterns, color_map=[GREY_C, YELLOW, GREEN]
        )

        rows.set_width(0.6 * FRAME_WIDTH)
        rows.center()
        self.add(rows)

        words = VGroup()
        for guess, row in zip(guesses, rows):
            word = Text(guess.upper(), font="Consolas")
            word.set_height(0.6 * row.get_height())
            for char, square in zip(word, row):
                char.move_to(square)
            words.add(word)

        # self.add(words)

        self.embed()


# Run simulated wordle games


def simulated_games(first_guess=None,
                    priors=None,
                    look_two_ahead=False,
                    regenerate_second_guess_map=True,
                    save_second_guess_map_to_file=True,
                    exclude_seen_words=False,
                    n_samples=None,
                    shuffle=False,
                    quiet=False,
                    results_file=None,
                    **kw
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

    if n_samples is None:
        samples = short_word_list
    else:
        samples = random.sample(short_word_list, n_samples)

    if shuffle:
        random.shuffle(samples)

    seen = set()

    # Keep track of the best next guess for a given set of possibilities
    next_guess_map = {}

    def get_next_guess(possibilities):
        phash = hash("".join(possibilities))
        if phash not in next_guess_map:
            next_guess_map[phash] = optimal_guess(
                all_words, possibilities, priors,
                look_two_ahead=look_two_ahead,
            )
        return next_guess_map[phash]

    scores = np.array([], dtype=int)
    game_results = []
    for answer in ProgressDisplay(samples, leave=False, desc=" Trying all wordle answers"):
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
            guess = get_next_guess(possibilities)

        scores = np.append(scores, [score])
        score_dist = [
            int((scores == i).sum())
            for i in range(1, scores.max() + 1)
        ]
        average = scores.mean()
        seen.add(answer)

        game_results.append(dict(
            score=int(score),
            answer=answer,
            guesses=guesses,
            patterns=patterns,
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
                f"Average: {average}",
                *" " * 2,
            ])
            if answer is not samples[0]:
                # Move cursor back up to the top of the message
                n = len(message.split("\n")) + 1
                print(("\033[F\033[K") * n)
            else:
                print("\r\033[K\n")
            print(message)
    final_result = dict(
        average_score=float(scores.mean()),
        score_distribution=score_dist,
        game_results=game_results,
    )
    if results_file:
        with open(os.path.join(DATA_DIR, results_file), 'w') as fp:
            json.dump(final_result, fp)
    return final_result


if __name__ == "__main__":
    words = get_word_list()
    wordle_words = get_word_list(short=True)
    simulated_games(
        first_guess="crane",
        # priors=get_frequency_based_priors(),
        priors=get_true_wordle_prior(),
        # exclude_seen_words=True,
        # shuffle=True,
    )
