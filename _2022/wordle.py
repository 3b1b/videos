from manim_imports_ext import *
from tqdm import tqdm as ProgressDisplay
from IPython.terminal.embed import InteractiveShellEmbed as embed
from scipy.stats import entropy


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
    # for first_guess in ProgressDisplay(first_guesses, leave=False, desc="Searching 2nd step entropies"):
    for first_guess, dist in zip(first_guesses, distributions):
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


def build_best_second_guess_map(guess, allowed_words, possible_words, priors, look_two_ahead=False):
    word_buckets = get_word_buckets(guess, possible_words)
    msg = f"Building second guess map for \"{guess}\""
    return [
        optimal_guess(allowed_words, bucket, priors, look_two_ahead)
        for bucket in ProgressDisplay(word_buckets, desc=msg, leave=False)
    ]


def get_second_guess_map(guess, regenerate=False, save_to_file=True, look_two_ahead=False):
    with open(SECOND_GUESS_MAP_FILE) as fp:
        saved_maps = json.load(fp)

    if guess not in saved_maps or regenerate:
        words = get_word_list()
        priors = get_frequency_based_priors()

        sg_map = build_best_second_guess_map(guess, words, words, priors, look_two_ahead)
        saved_maps[guess] = sg_map

        if save_to_file:
            with open(SECOND_GUESS_MAP_FILE, 'w') as fp:
                json.dump(saved_maps, fp)

    return saved_maps[guess]


# Solvers

def entropy_to_expected_score(ent):
    """
    Based on a regression associating entropies with typical scores
    from that point forward in simulated games, this function returns
    what the expected number of guesses required will be in a game where
    there's a given amount of entropy in the remaining possibilities.
    """
    return 1 + 0.45 * ent - 0.02 * ent**2


def get_expected_scores(allowed_words, possible_words, priors):
    # Currenty entropy of distribution
    weights = get_weights(possible_words, priors)
    H0 = entropy_of_distributions(weights)
    entropies = get_entropies(allowed_words, possible_words, weights)
    expected_further_guesses = entropy_to_expected_score(H0 - entropies)

    word_to_weight = dict(zip(possible_words, weights))
    probs = np.array([word_to_weight.get(w, 0) for w in allowed_words])

    # If this guess is the true answer, score is 1. Otherwise, it's 1 plus
    # the expected number of guesses it will take after getting the corresponding
    # amount of information.
    return probs + (1 - probs) * (1 + expected_further_guesses)


# def get_two_step_expected_scores(guess, allowed_second_guesses, possible_words, priors):
#     buckets = get_word_buckets(guess, possible_words)
#     dist = get_pattern_distributions([guess], possible_words, get_weights(possible_words, priors))[0]
#     return sum(
#         prob * get_expected_scores(allowed_second_guesses, bucket, priors).min()
#         for bucket, prob in zip(buckets, dist)
#     )


def get_guess_values(allowed_words, possible_words, priors, weight_to_prob=3.0, look_two_ahead=False, as_array=False, quiet=True):
    weights = get_weights(possible_words, priors)
    ents1 = get_entropies(allowed_words, possible_words, weights)
    probs = np.array([
        0 if word not in possible_words else weights[possible_words.index(word)]
        for word in allowed_words
    ])

    guess_values = ents1 + weight_to_prob * probs

    if look_two_ahead:
        # Look two steps out, but restricted to where second guess is
        # amoung the remaining possible words
        ents2 = np.zeros(ents1.shape)
        top_indices = np.argsort(ents1)[-100:]
        ents2[top_indices] = get_average_second_step_entropies(
            first_guesses=np.array(allowed_words)[top_indices],
            allowed_second_guesses=random.sample(allowed_words, 2000),
            # allowed_second_guesses=allowed_words,
            possible_words=possible_words,
            priors=priors
        )
        guess_values += ents2

    if not quiet:
        guess_values_argsort = np.argsort(guess_values)
        print("\nTop 10 picks")
        for index in guess_values_argsort[:-11:-1]:
            print("{}: {:.2f} + {:.2f} + {} * {:.2f}".format(
                allowed_words[index],
                ents1[index],
                ents2[index] if look_two_ahead else 0,
                weight_to_prob,
                probs[index],
            ))
        print("\n")

    if as_array:
        if look_two_ahead:
            return np.array([ents1, ents2, weight_to_prob * probs])
        else:
            return np.array([ents1, probs])
    else:
        return guess_values


def optimal_guess(allowed_words, possible_words, priors, look_two_ahead=False, quiet=True):
    expected_guess_values = get_expected_scores(allowed_words, possible_words, priors)
    return allowed_words[np.argmin(expected_guess_values)]
    # Old
    guess_values = get_guess_values(
        allowed_words, possible_words, priors,
        as_array=False,
        look_two_ahead=look_two_ahead,
        quiet=quiet,
    )
    return allowed_words[np.argmax(guess_values)]


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
        self.possibilities = list(self.all_words)

        self.add_grid()

    def get_word_list(self):
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
        if len(grid.pending_word) == 5:
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
        # TODO
        pass

    def has_won(self):
        return self.patterns[-1] == 3**5 - 1

    @staticmethod
    def get_grid_of_words(all_words, n_rows, n_cols, dots_index=-5, sort_key=None, font_size=24):
        sorted_words = list(sorted(all_words, key=sort_key))
        subset = sorted_words[:n_rows * n_cols]
        show_ellipsis = len(subset) < len(all_words)

        if show_ellipsis:
            subset[dots_index] = "..." if n_cols == 1 else "....."
            subset[dots_index + 1:] = sorted_words[dots_index + 1:]

        full_string = ""
        for i, word in zip(it.count(1), subset):
            full_string += str(word)
            if i % n_cols == 0:
                full_string += " \n"
            else:
                full_string += "  "

        full_text_mob = Text(full_string, font="Consolas", font_size=font_size)

        result = VGroup(*(
            full_text_mob.get_part_by_text(word)
            for word in subset
        ))
        if show_ellipsis and n_cols == 1:
            result[dots_index].rotate(PI / 2)
            result[dots_index].next_to(result[dots_index - 1], DOWN, SMALL_BUFF)
            result[dots_index + 1:].next_to(result[dots_index], DOWN, SMALL_BUFF)
        result.set_color(GREY_A)
        result.words = subset

        return result

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

    def setup(self):
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
            Text("Entropy", color=self.entropy_color),
        )
        if self.look_two_ahead:
            titles.add(TexText("Entropy$_2$", color=self.entropy_color)[0])
        if self.show_prior:
            titles.add(TexText("3$\\times$Prior", color=self.prior_color))

        titles.scale(0.7)
        titles.arrange(RIGHT, buff=MED_LARGE_BUFF)
        titles.set_max_width(5)
        low_y = titles[0][0].get_bottom()[1]

        for title in titles:
            title.shift((low_y - title[0].get_bottom()[1]) * UP)
            underline = Underline(title)
            underline.match_y(title[0].get_bottom() + 0.025 * DOWN)
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
        guess_values_array = get_guess_values(
            self.all_words,
            self.possibilities,
            self.priors,
            look_two_ahead=self.look_two_ahead,
            as_array=True,
        )
        guess_values = guess_values_array.sum(0)
        top_indices = np.argsort(guess_values)[:-self.n_top_picks - 1:-1]
        top_words = np.array(self.all_words)[top_indices]
        top_guess_value_parts = guess_values_array[:, top_indices]

        lines = self.get_guess_value_grid_lines()
        guess_value_grid = VGroup(*(
            self.get_guess_value_row(line, word, *values)
            for line, word, values in zip(
                lines, top_words, top_guess_value_parts.T
            )
        ))
        for value, row in zip(guess_values[top_indices], guess_value_grid):
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


class WordleDistributions(WordleScene):
    grid_center = [-4.5, -0.5, 0]
    grid_height = 5
    bar_style = dict(
        fill_color=TEAL_D,
        fill_opacity=0.8,
        stroke_color=WHITE,
        stroke_width=0.1,
    )

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
        p_label = self.get_dynamic_bar_label((
            "p\\left(", "00000", "\\right)", "=",
            "{" + poss_string, "\\over ", poss_string + "}",
            "=", "0.0000",
        ))
        num = Integer(edge_to_fix=DOWN, font_size=36)
        num.move_to(p_label[4], DOWN)
        p_label.replace_submobject(4, num)

        def update_label(label):
            label[4].set_value(int(get_bar().count))
            label[-1].set_value(get_bar().prob)
            label.next_to(get_bar(), UR, SMALL_BUFF)
            label.set_y(min(max_y, label.get_y()))
            label.shift_onto_screen(buff=2.0)

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
    # TODO, if you want to play with friends
    uniform_prior = True
    # wordle_based_prior = True

    def setup(self):
        super().setup()

    def get_pattern(self, guess):
        return pattern_from_string(input("Pattern please:"))


# Scenes


class IntroduceGame(WordleScene):
    secret_word = "browns"
    grid_center = 3.5 * LEFT

    def construct(self):
        grid = self.grid
        secret = VGroup(
            Text("Secret word"),
            Vector(0.5 * DOWN),
            Text("?????", font="Consolas"),
        )
        secret.arrange(DOWN, buff=MED_SMALL_BUFF)
        secret.next_to(grid, RIGHT, buff=1.5, aligned_edge=UP)

        self.play(
            Write(secret),
        )

        self.embed()


class IntroduceDistribution(WordleDistributions):
    secret_word = "stark"
    uniform_prior = True
    n_word_rows = 20

    def construct(self):
        # Total labels
        total_label = self.get_total_words_label()
        self.add(total_label)

        # Show an example guess
        guess = "weary"
        match_label = self.get_dynamic_match_label()
        word_grid = self.get_grid_of_matches(n_rows=self.n_word_rows)

        self.wait()
        self.play(ShowIncreasingSubsets(word_grid, run_time=3))
        self.wait(note=f"Write {guess}, (but don't submit)")
        if not self.presenter_mode:
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
        for x in range(3):
            self.wait(note="Play around with probability")
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
        ent_label = Text("Entropy")
        ent_label.set_color(BLUE)
        ent_label.next_to(brace, DOWN, SMALL_BUFF)

        self.play(
            FadeIn(entropy_definition, UP),
            FadeOut(standin, UP),
            FadeOut(want, UP),
        )
        self.wait()
        x_tracker.suspend_updating()
        self.play(UpdateFromAlphaFunc(
            x_tracker, lambda m, a: m.set_value(a * 150),
            run_time=10,
            rate_func=linear,
        ))
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

        if not self.presenter_mode:
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
        if not self.presenter_mode:
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
        if not self.presenter_mode:
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

            # Highlight answer
            arrow = Tex("\\rightarrow")
            pw = grid.pending_word.copy()
            pw.generate_target()
            pw.arrange(RIGHT, buff=0.05)
            rhs = ent_rhs.copy()
            group = VGroup(pw, arrow, rhs)
            group.set_color(BLUE)
            group.arrange(RIGHT)
            group.match_width(grid)
            group.next_to(grid, UP, LARGE_BUFF)

            self.add(group)
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


class UniformPriorExample(WordleSceneWithAnalysis):
    uniform_prior = True
    show_prior = False


class HowThePriorWorks(Scene):
    def construct(self):
        pass


class FreqPriorExample(WordleSceneWithAnalysis):
    pass


class WordlePriorExample(WordleSceneWithAnalysis):
    wordle_based_prior = True


class LookTwoStepsAhead(WordleSceneWithAnalysis):
    look_two_ahead = True
    # wordle_based_prior = True


# TODO
class ShowBestFollowOnWords(Scene):
    first_guess = "slane"

    def construct(self):
        pass


class FakeEntropyExample(Scene):
    def construct(self):
        # Functions
        def H(p):
            return sum(-x * math.log2(x) for x in (p, 1 - p))

        def H0(p):
            return -math.log2(sum(x * x for x in (p, 1 - p)))

        axes = Axes()
        H_graph = axes.get_graph(H, (0, 1))
        H0_graph = axes.get_graph(H0, (0, 1))


# Run simulated wordle games


def assisted_solver():
    all_words = get_word_list(short=False)
    priors = get_frequency_based_priors()

    guesses = []
    patterns = []
    possibility_counts = []
    possibilities = list(all_words)

    score = 1
    while len(possibilities) > 1:
        guess = input("Guess: ")
        pre_pattern = input(f"Pattern for \'{guess}\': ")
        pattern = pattern_from_string(pre_pattern)
        guesses.append(guess)
        patterns.append(pattern)
        possibilities = get_possible_words(guess, pattern, possibilities)
        possibility_counts.append(len(possibilities))
        entropy = entropy_of_distributions(get_weights(possibilities, priors))
        print(patterns_to_string(patterns))
        print(f"{len(possibilities)} possibilities")
        print(f"Entropy: {entropy}")
        print(possibilities[:10])
        optimal_guess(all_words, possibilities, priors, quiet=False)
        score += 1


def simulated_games(first_guess=None,
                    priors=None,
                    look_two_ahead=False,
                    regenerate_second_guess_map=True,
                    save_second_guess_map_to_file=False,
                    exclude_seen_words=False,
                    n_samples=None,
                    shuffle=True,
                    quiet=False,
                    ):
    score_dist = []
    all_words = get_word_list(short=False)
    short_word_list = get_word_list(short=True)

    if priors is None:
        priors = get_frequency_based_priors()

    if first_guess is None:
        first_guess = optimal_guess(
            all_words, all_words, priors,
            look_two_ahead=look_two_ahead,
        )

    second_guess_map = get_second_guess_map(
        first_guess,
        look_two_ahead=look_two_ahead,
        regenerate=regenerate_second_guess_map,
        save_to_file=save_second_guess_map_to_file,
    )

    if n_samples is None:
        samples = short_word_list
    else:
        samples = random.sample(short_word_list, n_samples)

    if shuffle:
        random.shuffle(samples)

    seen = set()

    for answer in ProgressDisplay(samples, leave=False, desc=" Trying all wordle answers"):
        guesses = []
        patterns = []
        possibility_counts = []
        possibilities = list(filter(lambda w: priors[w] > 0, all_words))
        guess = first_guess

        if exclude_seen_words:
            possibilities = list(set(possibilities).difference(seen))
            if len(possibilities) < 100:
                guess = optimal_guess(all_words, possibilities, priors)
            elif len(possibilities) % 100 == 0:
                guess = optimal_guess(all_words, possibilities, priors, look_two_ahead=False)

        score = 1
        while guess != answer:
            pattern = get_pattern(guess, answer)
            guesses.append(guess)
            patterns.append(pattern)
            possibilities = get_possible_words(guess, pattern, possibilities)
            possibility_counts.append(len(possibilities))

            if score > 1 or exclude_seen_words:
                guess = optimal_guess(
                    all_words, possibilities, priors,
                    look_two_ahead=look_two_ahead,
                )
            else:
                guess = second_guess_map[pattern]

            score += 1
        if score >= len(score_dist):
            score_dist.extend([0] * (score - len(score_dist)))
        score_dist[score - 1] += 1
        average = sum((i + 1) * s for i, s in enumerate(score_dist)) / sum(score_dist)

        seen.add(answer)

        # Print outcome
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
            n = len(message.split("\n")) + 1
            print(("\033[F\033[K") * n)
        else:
            print("\r\033[K\n")
        print(message)


if __name__ == "__main__":
    words = get_word_list()
    wordle_answers = get_word_list(short=True)
    simulated_games(
        # first_guess="tares",
        first_guess="slane",
        # priors=get_frequency_based_priors(),
        priors=get_true_wordle_prior(),
        regenerate_second_guess_map=True,
    )
