from manim_imports_ext import *
from tqdm import tqdm as ProgressDisplay
from IPython.terminal.embed import InteractiveShellEmbed


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


# Reading from files

def get_word_list(short=False):
    result = []
    file = SHORT_WORD_LIST_FILE if short else LONG_WORD_LIST_FILE
    with open(file) as fp:
        result.extend([word.strip() for word in fp.readlines()])
    return result


def get_global_word_frequencies():
    with open(WORD_FREQ_MAP_FILE) as fp:
        result = json.load(fp)
    return result


def write_freq_map():
    freq_map = dict()
    with open(WORD_FREQ_FILE) as fp:
        for line in fp.readlines():
            pieces = line.split(' ')
            word = pieces[0]
            freqs = [
                float(piece.strip())
                for piece in pieces[1:]
            ]
            freq_map[word] = np.mean(freqs)
    with open(WORD_FREQ_MAP_FILE, 'w') as fp:
        json.dump(freq_map, fp)


def get_word_priors(n_common=4000):
    """
    The prior that a given word is the answer should not
    simply be proportional to its relative frequency in English.
    We know that that list of answers was curated by some human
    based on whether they're sufficiently common. So for instance,
    even though a word like 'other' is much more frequent than
    a word like 'merit', it shouldn't be given a grossly higher
    prior; both are almost certainly in the 'common' list, so should
    be given similar priors.

    What we really want is to assign some probability to each word
    based on how likely it is to be selected as one of the
    'common' ones.

    Sort the words by frequency, then apply a sigmoid along it.
    """
    freq_map = get_global_word_frequencies()
    words = np.array(list(freq_map.keys()))
    freqs = np.array([freq_map[w] for w in words])
    arg_sort = freqs.argsort()
    sorted_words = words[arg_sort]

    # We want to imagine taking this sorted list, and putting it on a number
    # line so that it's length is 10, situating it so that the n_common most common
    # words are positive, then applying a sigmoid
    x_width = 20
    c = x_width * (-0.5 + n_common / len(words))
    xs = np.linspace(c - x_width / 2, c + x_width / 2, len(words))
    priors = dict()
    for word, x in zip(sorted_words, xs):
        priors[word] = sigmoid(x)
    return priors


def get_entropy_map(regenerate=False, rewrite=True):
    if regenerate:
        words = get_word_list()
        priors = get_word_priors()
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
    return frequencies / frequencies.sum()


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
    logs = np.log2(distributions, where=(distributions > atol))
    axis = len(distributions.shape) - 1
    return -(logs * distributions).sum(axis)


def get_entropies(allowed_words, possible_words, weights):
    distributions = get_pattern_distributions(allowed_words, possible_words, weights)
    return entropy_of_distributions(distributions)


def generate_entropy_map():
    words = get_word_list()
    priors = get_word_priors()
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
    for first_guess in ProgressDisplay(first_guesses, leave=False):
        dist = get_pattern_distributions([first_guess], possible_words, weights).flatten()
        word_buckets = get_word_buckets(first_guess, possible_words)
        # List of maximum entropies you could achieve in
        # the second step for each pattern you might see
        # after this setp
        ss_ents = np.array([
            get_entropies(
                allowed_words=allowed_second_guesses,
                possible_words=bucket,
                weights=get_weights(bucket, priors)
            ).max()
            for bucket in word_buckets
        ])
        # Multiply each such maximal entropy by the corresponding
        # probability of falling into that bucket
        result.append(np.dot(ss_ents, dist))
    return np.array(result)


def build_best_second_guess_map(guess, allowed_words, possible_words, priors, look_two_ahead=False):
    word_buckets = get_word_buckets(guess, possible_words)
    msg = f"Building second guess map for \"{guess}\""
    return [
        optimal_guess(allowed_words, bucket, priors, look_two_ahead)
        for bucket in ProgressDisplay(word_buckets, desc=msg)
    ]


def get_second_guess_map(guess, regenerate=False, save_to_file=True, look_two_ahead=False):
    with open(SECOND_GUESS_MAP_FILE) as fp:
        saved_maps = json.load(fp)

    if guess not in saved_maps or regenerate:
        words = get_word_list()
        priors = get_word_priors()

        sg_map = build_best_second_guess_map(guess, words, words, priors, look_two_ahead)
        saved_maps[guess] = sg_map

        if save_to_file:
            with open(SECOND_GUESS_MAP_FILE, 'w') as fp:
                json.dump(saved_maps, fp)

    return saved_maps[guess]


# Solvers


def get_scores(allowed_words, possible_words, priors, look_two_ahead=False, as_array=False, quiet=True):
    weights = get_weights(possible_words, priors)
    if len(possible_words) == len(allowed_words) == 12972:
        ent_map = get_entropy_map()
        ents1 = np.array([ent_map[w] for w in allowed_words])
    else:
        ents1 = get_entropies(allowed_words, possible_words, weights)
    # Look two steps out, but restricted to where second guess is
    # amoung the remaining possible words
    ents2 = np.zeros(ents1.shape)
    if look_two_ahead:
        top_indices = np.argsort(ents1)[-50:]
        ents2[top_indices] = get_average_second_step_entropies(
            first_guesses=np.array(allowed_words)[top_indices],
            allowed_second_guesses=random.sample(allowed_words, 1000),
            # allowed_second_guesses=allowed_words,
            possible_words=possible_words,
            priors=priors
        )
    probs = np.array([
        0 if word not in possible_words else weights[possible_words.index(word)]
        for word in allowed_words
    ])

    # TODO, be systematic about this weight
    weight_to_prob = 3.0
    scores = ents1 + ents2 + weight_to_prob * probs

    if not quiet:
        scores_argsort = np.argsort(scores)
        print("\nTop 10 picks")
        for index in scores_argsort[:-11:-1]:
            print("{}: {:.2f} + {:.2f} + {} * {:.2f}".format(
                allowed_words[index],
                ents1[index],
                ents2[index],
                weight_to_prob,
                probs[index],
            ))
        print("\n")

    if as_array:
        if look_two_ahead:
            return np.array([ents1, ents2, probs])
        else:
            return np.array([ents1, probs])
    else:
        return scores


def optimal_guess(allowed_words, possible_words, priors, look_two_ahead=False, quiet=True):
    scores = get_scores(allowed_words, possible_words, priors, look_two_ahead, quiet)
    return allowed_words[np.argmax(scores)]


def assisted_solver():
    all_words = get_word_list(short=False)
    priors = get_word_priors()

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


def simulated_games(first_guess='tares', quiet=False, n_samples=None):
    score_dist = []
    all_words = get_word_list(short=False)
    short_word_list = get_word_list(short=True)
    priors = get_word_priors()

    # priors = dict([
    #     (w, int(w in short_word_list))
    #     for w in all_words
    # ])  # Cheating

    second_guess_map = get_second_guess_map(
        first_guess,
        # regenerate=True,
        # save_to_file=False,
        look_two_ahead=True,
    )

    if n_samples is None:
        samples = set(short_word_list)
    else:
        samples = random.sample(short_word_list, n_samples)

    seen = set()

    for answer in samples:
        guesses = []
        patterns = []
        possibility_counts = []
        # possibilities = list(all_words)
        possibilities = list(set(all_words).difference(seen))

        guess = first_guess
        score = 1
        while guess != answer:
            pattern = get_pattern(guess, answer)
            guesses.append(guess)
            patterns.append(pattern)
            possibilities = get_possible_words(guess, pattern, possibilities)
            possibility_counts.append(len(possibilities))
            if score == 1:
                guess = second_guess_map[pattern]
            else:
                guess = optimal_guess(
                    all_words, possibilities, priors,
                    # look_two_ahead=True,
                    quiet=True
                )
            score += 1
        if score >= len(score_dist):
            score_dist.extend([0] * (score - len(score_dist) + 1))
        score_dist[score] += 1
        average = sum(i * s for i, s in enumerate(score_dist)) / sum(score_dist)

        seen.add(answer)

        if not quiet:
            print(f"Score: {score}")
            print(f"Answer: {answer}")
            print(f"Guesses: {guesses}")
            print(f"Reductions: {possibility_counts}")
            print(patterns_to_string((*patterns, 3**5 - 1)))
            print("\n" * (8 - len(patterns)))
            print(f"Distribution: {score_dist}")
            print(f"Average: {average}")
            print("\n" * 4)

    print(f"Distribution: {score_dist}")
    print(f"Average: {average}")


# Scenes


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

    def setup(self):
        self.all_words = self.get_word_list()
        self.priors = self.get_priors()
        if self.secret_word is None:
            self.secret_word = random.choices(
                self.all_words,
                weights=get_weights(self.all_words, self.priors),
            )[0]
        self.guesses = []
        self.patterns = []
        self.possibilities = list(self.all_words)

        self.add_grid()

    def get_word_list(self):
        return get_word_list()

    def get_priors(self):
        return get_word_priors()

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
        self.grid = grid
        self.add(grid)

    def add_letter(self, letter):
        grid = self.grid
        if len(grid.pending_word) == 5:
            return
        row = grid[len(grid.words)]
        square = row[len(grid.pending_word)]

        font_size = self.font_to_grid_height_ratio * self.grid_height
        letter_mob = Text(letter.upper(), font="Consolas", font_size=font_size)
        letter_mob.move_to(square)

        grid.pending_word.add(letter_mob)
        self.add(letter_mob)

    def delete_letter(self):
        if len(self.grid.pending_word) == 0:
            return
        letter_mob = self.grid.pending_word[-1]
        self.grid.pending_word.remove(letter_mob)
        self.remove(letter_mob)

    def add_word(self, word, pattern=None):
        for letter in word:
            self.add_letter(letter)
            self.wait(0.1, ignore_presenter_mode=True)

    def pending_word_as_string(self):
        return "".join(
            t.text.lower()
            for t in self.grid.pending_word
        )

    def reveal_pattern(self, pattern=None):
        grid = self.grid
        row = grid[len(grid.words)]
        word_mob = grid.pending_word
        guess = self.pending_word_as_string()

        if len(guess) != 5 or guess not in self.all_words:
            # Invalid guess
            c = row.get_center().copy()
            func = bezier([0, 0, 1, 1, -1, -1, 0, 0])
            self.play(UpdateFromAlphaFunc(
                VGroup(row, word_mob),
                lambda m, a: m.move_to(c + func(a) * RIGHT),
                run_time=0.5,
            ))
            return False

        if pattern is None:
            pattern = self.get_pattern(guess)

        self.possibilities = get_possible_words(
            guess, pattern, self.possibilities
        )
        self.guesses.append(guess)
        self.patterns.append(pattern)

        self.animate_pattern(row, word_mob, pattern)
        grid.words.add(grid.pending_word)
        grid.pending_word = VGroup()

        # Win condition
        if self.has_won():
            self.win_animation()

        return True

    def animate_pattern(self, row, word_mob, pattern):
        colors = [
            self.color_map[key]
            for key in pattern_to_int_list(pattern)
        ]
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
            LaggedStart(*(
                UpdateFromAlphaFunc(square, alpha_func)
                for square in row
            ), lag_ratio=0.5),
            LaggedStart(*(
                UpdateFromAlphaFunc(letter, alpha_func)
                for letter in word_mob
            ), lag_ratio=0.5),
            run_time=2,
        )

    def win_animation(self):
        pass

    def has_won(self):
        return self.patterns[-1] == 3**5 - 1

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
        elif symbol == 65288:
            self.delete_letter()
        elif symbol == 65293:  # Enter
            self.reveal_pattern()

        if char == 'q' and modifiers == 1:
            self.delete_letter()
            self.quit_interaction = True

        if not is_letter:
            super().on_key_press(symbol, modifiers)


class WordleSceneWithAnalysis(WordleScene):
    grid_center = [-2.25, 1, 0]
    grid_height = 4
    look_two_ahead = False
    show_prior = True
    n_top_picks = 13
    entropy_color = TEAL_C
    prior_color = BLUE_C

    def setup(self):
        super().setup()
        self.show_possible_words()
        self.add_top_picks_title()
        self.score_grid = VGroup()
        self.init_score_grid()

    def construct(self):
        self.show_scores()

    def init_score_grid(self):
        titles = self.top_picks_titles
        line = Line().match_width(titles)
        line.set_stroke(GREY_C, 1)
        lines = line.get_grid(self.n_top_picks, 1, buff=0.5)
        lines.next_to(titles, DOWN, buff=0.75)

        self.score_grid_lines = lines
        self.score_grid = VGroup()

    def get_count_label(self):
        score = len(self.grid.words)
        label = VGroup(
            Integer(len(self.possibilities), edge_to_fix=UR),
            Text("Pos.,"),
            DecimalNumber(self.get_current_entropy(), edge_to_fix=UR, color=self.entropy_color),
            Text("Bits", color=self.entropy_color),
        )
        label.arrange(
            RIGHT,
            buff=MED_SMALL_BUFF,
            aligned_edge=UP,
        )
        label.scale(0.6)
        if score == 0:
            label.next_to(self.grid[0], LEFT, buff=SMALL_BUFF)
            label.shift(self.grid[0].get_height() * UP)
        else:
            label.next_to(self.grid[score - 1], LEFT)
        return label

    def add_top_picks_title(self):
        titles = VGroup(
            Text("Top picks"),
            Text("Entropy", color=self.entropy_color),
        )
        if self.look_two_ahead:
            titles.add(TexText("Entropy$_2$", color=self.entropy_color)[0])
        if self.show_prior:
            titles.add(Text("Prior", color=self.prior_color))

        titles.scale(0.8)
        titles.arrange(RIGHT, buff=LARGE_BUFF)
        titles.set_max_width(7)
        low_y = titles[0][0].get_bottom()[1]

        for title in titles:
            title.shift((low_y - title[0].get_bottom()[1]) * UP)
            underline = Underline(title)
            underline.match_y(title[0].get_bottom() + 0.025 * DOWN)
            underline.set_stroke(WHITE, 2)
            underline.scale(1.2)
            title.add_to_back(underline)
            title.set_backstroke()
            underline.set_stroke(GREY_C, 2)

        titles.move_to(midpoint(self.grid.get_right(), RIGHT_SIDE))
        titles.to_edge(UP, buff=MED_SMALL_BUFF)

        self.add(titles)
        self.top_picks_titles = titles

    def reveal_pattern(self):
        self.isolate_guessed_row()
        is_valid_word = super().reveal_pattern()  # TODO, account for invalid guesses
        if not is_valid_word:
            return False

        self.show_possible_words()
        self.wait()
        if not self.has_won():
            self.show_scores()

    def animate_pattern(self, *args, **kwargs):
        for word_mob, word, bar in zip(self.shown_words, self.shown_words.words, self.prob_bars):
            if word not in self.possibilities and word != "...":
                word_mob.set_fill(RED, 0.5)
                bar.set_opacity(0.2)
        super().animate_pattern(*args, **kwargs)

    def isolate_guessed_row(self):
        guess = self.pending_word_as_string()
        rows = self.score_grid
        row_words = [row[0].text for row in rows]

        if guess in row_words:
            row = rows[row_words.index(guess)]
            rows.set_opacity(0.2)
            row[:-1].set_fill(YELLOW, 1)
        else:
            new_row = self.get_score_row(
                self.score_grid_lines[0], guess,
            )
            rows.shift(DOWN)
            rows.add(new_row)

    def get_shown_words(self, font_size=24):
        n_rows = 22 - 2 * len(self.grid.words)
        n_cols = 1
        dots_index = -5

        sorted_words = list(sorted(self.possibilities))
        words = sorted_words[:n_rows * n_cols]
        show_ellipsis = len(words) < len(self.possibilities)

        if show_ellipsis:
            words[dots_index] = "..."
            words[dots_index + 1:] = sorted_words[dots_index + 1:]

        full_string = ""
        for i, word in zip(it.count(1), words):
            full_string += str(word)
            if i % n_cols == 0:
                full_string += " \n"
            else:
                full_string += "  "

        full_text_mob = Text(full_string, font="Consolas", font_size=font_size)
        shown_words = VGroup(*(
            full_text_mob.get_part_by_text(word)
            for word in words
        ))
        if show_ellipsis:
            shown_words[dots_index].rotate(PI / 2)
            shown_words[dots_index].next_to(shown_words[dots_index - 1], DOWN, SMALL_BUFF)
            shown_words[dots_index + 1:].next_to(shown_words[dots_index], DOWN, SMALL_BUFF)
        shown_words.set_color(GREY_A)
        shown_words.words = words
        return shown_words

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
        return bars

    def show_possible_words(self):
        shown_words = self.get_shown_words()
        count_label = self.get_count_label()
        shown_words.next_to(count_label[:2], DOWN, buff=0.35)
        prob_bars = self.get_probability_bars(shown_words)

        if len(self.grid.words) > 0:
            # Set up label transition
            prev_count_label = self.count_label
            count_label.match_x(prev_count_label)

            num_rate_func = squish_rate_func(rush_into, 0.3, 1)

            def update_moving_count_label(label, alpha):
                for i in (0, 2):
                    label[i].set_value(interpolate(
                        prev_count_label[i].get_value(),
                        count_label[i].get_value(),
                        num_rate_func(alpha),
                    ))
                label.move_to(interpolate(
                    prev_count_label.get_center(),
                    count_label.get_center(),
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
                FadeOut(self.score_grid, RIGHT),
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

    def show_scores(self):
        self.score_grid = self.get_score_grid()
        self.play(ShowIncreasingSubsets(self.score_grid))

    def get_score_grid(self, font_size=36):
        scores_array = get_scores(
            self.all_words,
            self.possibilities,
            self.priors,
            as_array=True,
        )
        scores = scores_array.sum(0)
        top_indices = np.argsort(scores)[:-self.n_top_picks - 1:-1]
        top_words = np.array(self.all_words)[top_indices]
        top_score_parts = scores_array[:, top_indices]

        lines = self.get_score_grid_lines()
        score_grid = VGroup(*(
            self.get_score_row(line, word, *values)
            for line, word, values in zip(
                lines, top_words, top_score_parts.T
            )
        ))
        return score_grid

    def get_score_row(self, line, word,
                      entropy=None,
                      entropy2=None,
                      probability=None,
                      font_size=36):
        titles = self.top_picks_titles
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

    def get_score_grid_lines(self):
        titles = self.top_picks_titles
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
    def construct(self):
        pass


class ExternalPatternEntry(WordleSceneWithAnalysis):
    # TODO, if you want to play with friends
    def setup(self):
        super().setup()

    def get_pattern(self, guess):
        return pattern_from_string(input("Pattern please:"))


class Test(WordleSceneWithAnalysis):
    CONFIG = {
        "random_seed": None
    }

    # def get_priors(self):
    #     short_list = get_word_list(short=True)
    #     # return dict(zip(self.all_words, it.repeat(1)))
    #     return dict(
    #         (w, 1 if w in short_list else 0)
    #         for w in self.all_words
    #     )

    def construct(self):
        super().construct()
        # Embed
        # self.embed()


# Script

if __name__ == "__main__":
    words = get_word_list()
    priors = get_word_priors()
    # weights = get_weights(words, priors)
    # get_entropies(words, words, weights)
    # get_average_second_step_entropies(
    #     words, words, words, priors
    # )

    # assisted_solver()
    # simulated_games(quiet=False)

    allowed_words = words
    possible_words = random.sample(allowed_words, 1000)
    weights = get_weights(possible_words, priors)

    shell = InteractiveShellEmbed()
    shell()
