/-
Copyright (c) 2024 Joseph Myers. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Joseph Myers
-/
module

public import Mathlib.Data.Fin.VecNotation
public import Mathlib.Data.List.ChainOfFn
public import Mathlib.Data.List.TakeWhile
public import Mathlib.Data.Nat.Dist
public import Mathlib.Data.Fintype.Fin
public import Mathlib.Tactic.IntervalCases
public import Mathlib.Tactic.FinCases

/-!
# IMO 2024 Q5

Turbo the snail plays a game on a board with $2024$ rows and $2023$ columns. There are hidden
monsters in $2022$ of the cells. Initially, Turbo does not know where any of the monsters are,
but he knows that there is exactly one monster in each row except the first row and the last
row, and that each column contains at most one monster.

Turbo makes a series of attempts to go from the first row to the last row. On each attempt,
he chooses to start on any cell in the first row, then repeatedly moves to an adjacent cell
sharing a common side. (He is allowed to return to a previously visited cell.) If he reaches a
cell with a monster, his attempt ends and he is transported back to the first row to start a
new attempt. The monsters do not move, and Turbo remembers whether or not each cell he has
visited contains a monster. If he reaches any cell in the last row, his attempt ends and the
game is over.

Determine the minimum value of $n$ for which Turbo has a strategy that guarantees reaching
the last row on the $n$th attempt or earlier, regardless of the locations of the monsters.

We follow the solution from the
[official solutions](https://www.imo2024.uk/s/IMO2024-solutions-updated.pdf). To show that $n$
is at least $3$, it is possible that the first cell Turbo encounters in the second row on his
first attempt contains a monster, and also possible that the first cell Turbo encounters in the
third row on his second attempt contains a monster. To show that $3$ attempts suffice, the first
attempt can be used to locate the monster in the second row; if this is not at either side of
the board, two more attempts suffice to pass behind that monster and from there go along its
column to the last row, while if it is at one side of the board, the second attempt follows a
zigzag path such that if it encounters a monster the third attempt can avoid all monsters.
-/

@[expose] public section

namespace Imo2024Q5

/-! ### Definitions for setting up the problem -/

-- There are N monsters, N+1 columns and N+2 rows.
variable {N : ℕ}

/-- A cell on the board for the game. -/
abbrev Cell (N : ℕ) : Type := Fin (N + 2) × Fin (N + 1)

/-- A row that is neither the first nor the last (and thus contains a monster). -/
abbrev InteriorRow (N : ℕ) : Type := (Set.Icc 1 ⟨N, by lia⟩ : Set (Fin (N + 2)))

/-- Data for valid positions of the monsters. -/
abbrev MonsterData (N : ℕ) : Type := InteriorRow N ↪ Fin (N + 1)

/-- The cells with monsters as a Set, given an injection from rows to columns. -/
def MonsterData.monsterCells (m : MonsterData N) :
    Set (Cell N) :=
  Set.range (fun x : InteriorRow N ↦ ((x : Fin (N + 2)), m x))

/-- Whether two cells are adjacent. -/
def Adjacent (x y : Cell N) : Prop :=
  Nat.dist x.1 y.1 + Nat.dist x.2 y.2 = 1

/-- A valid path from the first to the last row. -/
structure Path (N : ℕ) where
  /-- The cells on the path. -/
  cells : List (Cell N)
  nonempty : cells ≠ []
  head_first_row : (cells.head nonempty).1 = 0
  last_last_row : (cells.getLast nonempty).1 = Fin.last (N + 1)
  valid_move_seq : cells.IsChain Adjacent

/-- The first monster on a path, or `none`. -/
noncomputable def Path.firstMonster (p : Path N) (m : MonsterData N) : Option (Cell N) :=
  letI := Classical.propDecidable
  p.cells.find? (fun x ↦ (x ∈ m.monsterCells : Bool))

/-- A strategy, given the results of initial attempts, returns a path for the next attempt. -/
abbrev Strategy (N : ℕ) : Type := ⦃k : ℕ⦄ → (Fin k → Option (Cell N)) → Path N

/-- Playing a strategy, k attempts. -/
noncomputable def Strategy.play (s : Strategy N) (m : MonsterData N) :
    (k : ℕ) → Fin k → Option (Cell N)
| 0 => Fin.elim0
| k + 1 => Fin.snoc (s.play m k) ((s (s.play m k)).firstMonster m)

/-- The predicate for a strategy winning within the given number of attempts. -/
def Strategy.WinsIn (s : Strategy N) (m : MonsterData N) (k : ℕ) : Prop :=
  none ∈ Set.range (s.play m k)

/-- Whether a strategy forces a win within the given number of attempts. -/
def Strategy.ForcesWinIn (s : Strategy N) (k : ℕ) : Prop :=
  ∀ m, s.WinsIn m k

/-! ### API definitions and lemmas about `Cell` -/

/-- Reflecting a cell of the board (swapping left and right sides of the board). -/
def Cell.reflect (c : Cell N) : Cell N := (c.1, c.2.rev)

/-! ### API definitions and lemmas about `MonsterData` -/

/-- The row 1, in the form required for MonsterData. -/
def row1 (hN : 2 ≤ N) : InteriorRow N :=
  ⟨1, ⟨by lia, by
    rw [Fin.le_def]
    simp
    lia⟩⟩

lemma coe_coe_row1 (hN : 2 ≤ N) : (((row1 hN) : Fin (N + 2)) : ℕ) = 1 :=
  rfl

/-- The row 2, in the form required for MonsterData. -/
def row2 (hN : 2 ≤ N) : InteriorRow N :=
  ⟨⟨2, by lia⟩, ⟨by
    simp only [Fin.le_def, Fin.val_one]
    lia, by
    simp only [Fin.le_def]
    lia⟩⟩

/-- Reflecting monster data. -/
def MonsterData.reflect (m : MonsterData N) : MonsterData N where
  toFun := Fin.rev ∘ m
  inj' := fun i j hij ↦ by simpa using hij

set_option backward.isDefEq.respectTransparency false in
lemma MonsterData.reflect_reflect (m : MonsterData N) : m.reflect.reflect = m := by
  ext i
  simp [MonsterData.reflect]

lemma MonsterData.notMem_monsterCells_of_fst_eq_zero (m : MonsterData N)
    {c : Cell N} (hc : c.1 = 0) : c ∉ m.monsterCells := by
  simp [monsterCells, Prod.ext_iff, hc]

lemma MonsterData.le_N_of_mem_monsterCells {m : MonsterData N} {c : Cell N}
    (hc : c ∈ m.monsterCells) : (c.1 : ℕ) ≤ N := by
  simp only [monsterCells, Set.mem_range, Subtype.exists, Set.mem_Icc] at hc
  rcases hc with ⟨r, ⟨h1, hN⟩, rfl⟩
  rw [Fin.le_def] at hN
  exact hN

lemma MonsterData.mk_mem_monsterCells_iff_of_le {m : MonsterData N} {r : Fin (N + 2)}
    (hr1 : 1 ≤ r) (hrN : r ≤ ⟨N, by lia⟩) {c : Fin (N + 1)} :
    (r, c) ∈ m.monsterCells ↔ m ⟨r, hr1, hrN⟩ = c := by
  simp only [monsterCells, Set.mem_range, Prod.mk.injEq]
  refine ⟨?_, ?_⟩
  · rintro ⟨r', rfl, rfl⟩
    simp only
  · rintro rfl
    exact ⟨⟨r, hr1, hrN⟩, rfl, rfl⟩

lemma MonsterData.mem_monsterCells_iff_of_le {m : MonsterData N} {x : Cell N}
    (hr1 : 1 ≤ x.1) (hrN : x.1 ≤ ⟨N, by lia⟩) :
    x ∈ m.monsterCells ↔ m ⟨x.1, hr1, hrN⟩ = x.2 :=
  MonsterData.mk_mem_monsterCells_iff_of_le hr1 hrN

lemma MonsterData.mk_mem_monsterCells_iff {m : MonsterData N} {r : Fin (N + 2)}
    {c : Fin (N + 1)} :
    (r, c) ∈ m.monsterCells ↔ ∃ (hr1 : 1 ≤ r) (hrN : r ≤ ⟨N, by lia⟩), m ⟨r, hr1, hrN⟩ = c := by
  refine ⟨fun h ↦ ?_, fun ⟨hr1, hrN, h⟩ ↦ (mem_monsterCells_iff_of_le hr1 hrN).2 h⟩
  rcases h with ⟨⟨mr, hr1, hrN⟩, h⟩
  simp only [Prod.mk.injEq] at h
  rcases h with ⟨rfl, rfl⟩
  exact ⟨hr1, hrN, rfl⟩

/-! ### API definitions and lemmas about `Adjacent` -/

lemma Adjacent.le_of_lt {x y : Cell N} (ha : Adjacent x y) {r : Fin (N + 2)} (hr : r < y.1) :
    r ≤ x.1 := by
  rw [Adjacent, Nat.dist] at ha
  lia

/-! ### API definitions and lemmas about `Path` -/

lemma Path.exists_mem_fst_eq (p : Path N) (r : Fin (N + 2)) : ∃ c ∈ p.cells, c.1 = r := by
  let i : ℕ := p.cells.findIdx fun c ↦ r ≤ c.1
  have hi : i < p.cells.length := by
    refine List.findIdx_lt_length_of_exists ⟨p.cells.getLast p.nonempty, ?_⟩
    simp only [List.getLast_mem, p.last_last_row, decide_eq_true_eq, true_and, Fin.last]
    rw [Fin.le_def]
    have h := r.isLt
    rw [Nat.lt_succ_iff] at h
    convert! h
  have hig : r ≤ (p.cells[i]).1 := of_decide_eq_true (List.findIdx_getElem (w := hi))
  refine ⟨p.cells[i], List.getElem_mem _, ?_⟩
  refine (hig.lt_or_eq.resolve_left fun h => ?_).symm
  rcases Nat.eq_zero_or_pos i with hi | hi
  · simp [hi, List.getElem_zero, p.head_first_row] at h
  · suffices r ≤ p.cells[i - 1].1 by
      have hi' : i - 1 < i := by lia
      exact of_decide_eq_false (List.not_of_lt_findIdx hi') this
    have ha : Adjacent p.cells[i - 1] p.cells[i] := by
      convert! List.isChain_iff_getElem.1 p.valid_move_seq (i - 1) ?_
      · simp [Nat.sub_add_cancel hi]
      · lia
    exact ha.le_of_lt h

lemma Path.exists_mem_le_fst (p : Path N) (r : Fin (N + 2)) : ∃ c ∈ p.cells, r ≤ c.1 := by
  rcases p.exists_mem_fst_eq r with ⟨c, hc, he⟩
  exact ⟨c, hc, he.symm.le⟩

/-- The first path element on a row. -/
def Path.findFstEq (p : Path N) (r : Fin (N + 2)) : Cell N :=
  (p.cells.find? (fun c ↦ c.1 = r)).get (List.find?_isSome.2 (by simpa using p.exists_mem_fst_eq r))

lemma Path.find_eq_some_findFstEq (p : Path N) (r : Fin (N + 2)) :
    p.cells.find? (fun c ↦ c.1 = r) = some (p.findFstEq r) := by
  rw [Option.eq_some_iff_get_eq]
  exact ⟨_, rfl⟩

lemma Path.findFstEq_mem_cells (p : Path N) (r : Fin (N + 2)) : p.findFstEq r ∈ p.cells :=
  List.mem_of_find?_eq_some (p.find_eq_some_findFstEq r)

lemma Path.findFstEq_fst (p : Path N) (r : Fin (N + 2)) : (p.findFstEq r).1 = r := by
  have h := List.find?_some (p.find_eq_some_findFstEq r)
  simpa using h

lemma find?_eq_eq_find?_le {l : List (Cell N)} {r : Fin (N + 2)} (hne : l ≠ [])
    (hf : (l.head hne).1 ≤ r) (ha : l.IsChain Adjacent) :
    l.find? (fun c ↦ c.1 = r) = l.find? (fun c ↦ r ≤ c.1) := by
  induction l
  case nil => simp at hne
  case cons head tail hi =>
    by_cases h : head.1 = r
    · simp [h]
    · have h' : ¬(r ≤ head.1) := fun hr' ↦ h (le_antisymm hf hr')
      simp only [h, decide_false, Bool.false_eq_true, not_false_eq_true, List.find?_cons_of_neg, h']
      rcases tail with ⟨⟩ | ⟨htail, ttail⟩
      · simp
      · simp only [List.isChain_cons_cons] at ha
        rcases ha with ⟨ha1, ha2⟩
        simp only [List.head_cons] at hf
        simp only [ne_eq, reduceCtorEq, not_false_eq_true, List.head_cons, ha2, true_implies] at hi
        refine hi ?_
        simp only [Adjacent, Nat.dist] at ha1
        lia

lemma Path.findFstEq_eq_find?_le (p : Path N) (r : Fin (N + 2)) : p.findFstEq r =
    (p.cells.find? (fun c ↦ r ≤ c.1)).get
      (List.find?_isSome.2 (by simpa using p.exists_mem_le_fst r)) := by
  rw [findFstEq]
  convert! rfl using 2
  refine (find?_eq_eq_find?_le p.nonempty ?_ p.valid_move_seq).symm
  simp [p.head_first_row]

lemma Path.firstMonster_isSome {p : Path N} {m : MonsterData N} :
    (p.firstMonster m).isSome = true ↔ ∃ x, x ∈ p.cells ∧ x ∈ m.monsterCells := by
  convert! List.find?_isSome
  simp

lemma Path.firstMonster_eq_none {p : Path N} {m : MonsterData N} :
    (p.firstMonster m) = none ↔ ∀ x, x ∈ p.cells → x ∉ m.monsterCells := by
  convert! List.find?_eq_none
  simp

lemma Path.one_lt_length_cells (p : Path N) : 1 < p.cells.length := by
  by_contra hl
  have h : p.cells.length = 0 ∨ p.cells.length = 1 := by lia
  rcases h with h | h
  · rw [List.length_eq_zero_iff] at h
    exact p.nonempty h
  · rw [List.length_eq_one_iff] at h
    rcases h with ⟨c, hc⟩
    have h1 := p.head_first_row
    simp_rw [hc, List.head_cons] at h1
    have h2 := p.last_last_row
    simp [hc, List.getLast_singleton, h1] at h2

/-- Remove the first cell from a path, if the second cell is also on the first row. -/
def Path.tail (p : Path N) : Path N where
  cells := if (p.cells[1]'p.one_lt_length_cells).1 = 0 then p.cells.tail else p.cells
  nonempty := by
    split_ifs
    · rw [← List.length_pos_iff_ne_nil, List.length_tail, Nat.sub_pos_iff_lt]
      exact p.one_lt_length_cells
    · exact p.nonempty
  head_first_row := by
    split_ifs with h
    · convert! h
      rw [List.head_tail]
    · exact p.head_first_row
  last_last_row := by
    split_ifs
    · rw [← p.last_last_row, List.getLast_tail]
    · exact p.last_last_row
  valid_move_seq := by
    split_ifs
    · exact List.IsChain.tail p.valid_move_seq
    · exact p.valid_move_seq

lemma Path.tail_induction {motive : Path N → Prop} (ind : ∀ p, motive p.tail → motive p)
    (base : ∀ p, (p.cells[1]'p.one_lt_length_cells).1 ≠ 0 → motive p) (p : Path N) : motive p := by
  rcases p with ⟨cells, nonempty, head_first_row, last_last_row, valid_move_seq⟩
  let p' : Path N := ⟨cells, nonempty, head_first_row, last_last_row, valid_move_seq⟩
  induction cells
  case nil => simp at nonempty
  case cons head tail hi =>
    by_cases h : (p'.cells[1]'p'.one_lt_length_cells).1 = 0
    · refine ind p' ?_
      simp_rw [Path.tail, ite_eq_left h, p', List.tail_cons]
      exact hi _ _ _ _
    · exact base p' h

lemma Path.tail_findFstEq (p : Path N) {r : Fin (N + 2)} (hr : r ≠ 0) :
    p.tail.findFstEq r = p.findFstEq r := by
  by_cases h : (p.cells[1]'p.one_lt_length_cells).1 = 0
  · simp_rw [Path.tail, ite_eq_left h]
    nth_rw 2 [Path.findFstEq]
    rcases p with ⟨cells, nonempty, head_first_row, last_last_row, valid_move_seq⟩
    rcases cells with ⟨⟩ | ⟨head, tail⟩
    · simp at nonempty
    · simp only [List.head_cons] at head_first_row
      simp only [List.find?_cons, head_first_row, hr.symm, decide_false]
      rfl
  · simp_rw [Path.tail, ite_eq_right h]

lemma Path.tail_firstMonster (p : Path N) (m : MonsterData N) :
    p.tail.firstMonster m = p.firstMonster m := by
  by_cases h : (p.cells[1]'p.one_lt_length_cells).1 = 0
  · simp_rw [Path.tail, ite_eq_left h]
    nth_rw 2 [Path.firstMonster]
    rcases p with ⟨cells, nonempty, head_first_row, last_last_row, valid_move_seq⟩
    rcases cells with ⟨⟩ | ⟨head, tail⟩
    · simp at nonempty
    · simp only [List.head_cons] at head_first_row
      simp [m.notMem_monsterCells_of_fst_eq_zero head_first_row, firstMonster]
  · simp_rw [Path.tail, ite_eq_right h]

lemma Path.firstMonster_eq_of_findFstEq_mem {p : Path N} {m : MonsterData N}
    (h : p.findFstEq 1 ∈ m.monsterCells) : p.firstMonster m = some (p.findFstEq 1) := by
  induction p using Path.tail_induction
  case base p h0 =>
    have hl := p.one_lt_length_cells
    have adj : Adjacent p.cells[0] p.cells[1] :=
      List.isChain_iff_getElem.1 p.valid_move_seq 0 (by lia)
    simp_rw [Adjacent, Nat.dist] at adj
    have hc0 : (p.cells[0].1 : ℕ) = 0 := by
      convert! Fin.ext_iff.1 p.head_first_row
      exact List.getElem_zero _
    have hc1 : (p.cells[1].1 : ℕ) ≠ 0 := Fin.val_ne_iff.2 h0
    have h1 : (p.cells[1].1 : ℕ) = 1 := by lia
    simp_rw [firstMonster, findFstEq]
    rcases p with ⟨cells, nonempty, head_first_row, last_last_row, valid_move_seq⟩
    rcases cells with ⟨⟩ | ⟨head, tail⟩
    · simp at nonempty
    · simp only [List.head_cons] at head_first_row
      simp only [List.getElem_cons_succ] at h1
      simp only [List.length_cons, lt_add_iff_pos_left, List.length_pos_iff_ne_nil] at hl
      simp only [m.notMem_monsterCells_of_fst_eq_zero head_first_row, decide_false,
        Bool.false_eq_true, not_false_eq_true, List.find?_cons_of_neg, head_first_row,
        Fin.zero_eq_one_iff, Nat.reduceEqDiff, Option.some_get]
      simp only [findFstEq, head_first_row, Fin.zero_eq_one_iff, Nat.reduceEqDiff, decide_false,
        Bool.false_eq_true, not_false_eq_true, List.find?_cons_of_neg] at h
      rcases tail with ⟨⟩ | ⟨htail, ttail⟩
      · simp at hl
      · simp only [List.getElem_cons_zero] at h1
        have h1' : htail.1 = 1 := by simp [Fin.ext_iff, h1]
        simp only [h1', decide_true, List.find?_cons_of_pos, Option.get_some] at h
        simp only [h1', h, decide_true, List.find?_cons_of_pos]
  case ind p ht =>
    have h1 : (1 : Fin (N + 2)) ≠ 0 := by simp
    rw [p.tail_findFstEq h1, p.tail_firstMonster m] at ht
    exact ht h

lemma Path.findFstEq_fst_sub_one_mem (p : Path N) {r : Fin (N + 2)} (hr : r ≠ 0) :
    (⟨(r : ℕ) - 1, by lia⟩, (p.findFstEq r).2) ∈ p.cells := by
  rw [findFstEq_eq_find?_le]
  have h1 := p.one_lt_length_cells
  obtain ⟨cr, hcrc, hcrr⟩ := p.exists_mem_fst_eq r
  rcases p with ⟨cells, nonempty, head_first_row, last_last_row, valid_move_seq⟩
  dsimp only at h1 hcrc ⊢
  have hd : ∃ c ∈ cells, decide (r ≤ c.1) = true := ⟨cr, hcrc, by simpa using hcrr.symm.le⟩
  have hd' : cells.dropWhile (fun c ↦ !decide (r ≤ c.1)) ≠ [] := by simpa using hd
  have ht : cells.takeWhile (fun c ↦ !decide (r ≤ c.1)) ≠ [] := by
    intro h
    rw [List.takeWhile_eq_nil_iff] at h
    replace h := h (by lia)
    simp [List.getElem_zero, head_first_row, hr] at h
  simp_rw [cells.find?_eq_head_dropWhile_not hd, Option.get_some]
  rw [← cells.takeWhile_append_dropWhile (p := fun c ↦ !decide (r ≤ c.1)),
    List.isChain_append] at valid_move_seq
  have ha := valid_move_seq.2.2
  simp only [List.head?_eq_some_head hd', List.getLast?_eq_some_getLast ht, Option.mem_def,
    Option.some.injEq, forall_eq'] at ha
  nth_rw 1 [← cells.takeWhile_append_dropWhile (p := fun c ↦ !decide (r ≤ c.1))]
  refine List.mem_append_left _ ?_
  convert! List.getLast_mem ht using 1
  have htr : ((List.takeWhile (fun c ↦ !decide (r ≤ c.1)) cells).getLast ht).1 < r := by
    simpa using List.mem_takeWhile_imp (List.getLast_mem ht)
  have hdr : r ≤ ((List.dropWhile (fun c ↦ !decide (r ≤ c.1)) cells).head hd').1 := by
    simpa using cells.head_dropWhile_not (fun c ↦ !decide (r ≤ c.1)) hd'
  simp only [Adjacent, Nat.dist] at ha
  have hrm : N + 1 + (r : ℕ) = N + 2 + (r - 1) := by lia
  simp only [Prod.ext_iff, Fin.ext_iff]
  lia

lemma Path.mem_of_firstMonster_eq_some {p : Path N} {m : MonsterData N} {c : Cell N}
    (h : p.firstMonster m = some c) : c ∈ p.cells ∧ c ∈ m.monsterCells := by
  simp_rw [firstMonster] at h
  have h₁ := List.mem_of_find?_eq_some h
  have h₂ := List.find?_some h
  simp only [decide_eq_true_eq] at h₂
  exact ⟨h₁, h₂⟩

/-- Convert a function giving the cells of a path to a `Path`. -/
def Path.ofFn {m : ℕ} (f : Fin m → Cell N) (hm : m ≠ 0)
    (hf : (f ⟨0, Nat.pos_of_ne_zero hm⟩).1 = 0)
    (hl : (f ⟨m - 1, Nat.sub_one_lt hm⟩).1 = ⟨N + 1, by lia⟩)
    (ha : ∀ (i : ℕ) (hi : i + 1 < m), Adjacent (f ⟨i, Nat.lt_of_succ_lt hi⟩) (f ⟨i + 1, hi⟩)) :
    Path N where
  cells := List.ofFn f
  nonempty := mt List.ofFn_eq_nil_iff.1 hm
  head_first_row := by
    rw [List.head_ofFn, hf]
  last_last_row := by
    simp [Fin.last, List.getLast_ofFn, hl]
  valid_move_seq := by
    rwa [List.isChain_ofFn]

lemma Path.ofFn_cells {m : ℕ} (f : Fin m → Cell N) (hm : m ≠ 0)
    (hf : (f ⟨0, Nat.pos_of_ne_zero hm⟩).1 = 0)
    (hl : (f ⟨m - 1, Nat.sub_one_lt hm⟩).1 = ⟨N + 1, by lia⟩)
    (ha : ∀ (i : ℕ) (hi : i + 1 < m), Adjacent (f ⟨i, Nat.lt_of_succ_lt hi⟩) (f ⟨i + 1, hi⟩)) :
    (Path.ofFn f hm hf hl ha).cells = List.ofFn f :=
  rfl

lemma Path.ofFn_firstMonster_eq_none {m : ℕ} (f : Fin m → Cell N) (hm hf hl ha)
    (m : MonsterData N) :
    ((Path.ofFn f hm hf hl ha).firstMonster m) = none ↔ ∀ i, f i ∉ m.monsterCells := by
  simp [firstMonster_eq_none, ofFn_cells, List.mem_ofFn]

/-- Reflecting a path. -/
def Path.reflect (p : Path N) : Path N where
  cells := p.cells.map Cell.reflect
  nonempty := mt List.map_eq_nil_iff.1 p.nonempty
  head_first_row := by
    rw [List.head_map]
    exact p.head_first_row
  last_last_row := by
    rw [List.getLast_map]
    exact p.last_last_row
  valid_move_seq := by
    refine List.isChain_map_of_isChain _ ?_ p.valid_move_seq
    intro x y h
    simp_rw [Adjacent, Nat.dist, Cell.reflect, Fin.rev] at h ⊢
    lia

set_option backward.isDefEq.respectTransparency false in
lemma Path.firstMonster_reflect (p : Path N) (m : MonsterData N) :
    p.reflect.firstMonster m.reflect = (p.firstMonster m).map Cell.reflect := by
  simp_rw [firstMonster, reflect, List.find?_map]
  convert! rfl
  simp only [Function.comp_apply, decide_eq_decide, MonsterData.monsterCells]
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
  · rcases h with ⟨i, hi⟩
    refine ⟨i, ?_⟩
    simpa [MonsterData.reflect, Cell.reflect, Prod.ext_iff] using hi
  · rcases h with ⟨i, hi⟩
    refine ⟨i, ?_⟩
    simpa [MonsterData.reflect, Cell.reflect, Prod.ext_iff] using hi

/-! ### API definitions and lemmas about `Strategy` -/

lemma Strategy.play_comp_castLE (s : Strategy N) (m : MonsterData N) {k₁ k₂ : ℕ} (hk : k₁ ≤ k₂) :
    s.play m k₂ ∘ Fin.castLE hk = s.play m k₁ := by
  induction hk
  case refl => rfl
  case step k' hk' hki =>
    rw [← hki, ← Fin.castLE_comp_castLE hk' (Nat.le_succ k'), ← Function.comp_assoc]
    convert! rfl
    exact Fin.snoc_comp_castSucc.symm

lemma Strategy.play_apply_of_le (s : Strategy N) (m : MonsterData N) {i k₁ k₂ : ℕ} (hi : i < k₁)
    (hk : k₁ ≤ k₂) : s.play m k₂ ⟨i, hi.trans_le hk⟩ = s.play m k₁ ⟨i, hi⟩ := by
  rw [← s.play_comp_castLE m hk]
  rfl

lemma Strategy.play_zero (s : Strategy N) (m : MonsterData N) {k : ℕ} (hk : 0 < k) :
    s.play m k ⟨0, hk⟩ = (s Fin.elim0).firstMonster m := by
  have hk' : 1 ≤ k := by lia
  rw [s.play_apply_of_le m zero_lt_one hk']
  rfl

lemma Strategy.play_one (s : Strategy N) (m : MonsterData N) {k : ℕ} (hk : 1 < k) :
    s.play m k ⟨1, hk⟩ = (s ![(s Fin.elim0).firstMonster m]).firstMonster m := by
  have hk' : 2 ≤ k := by lia
  rw [s.play_apply_of_le m one_lt_two hk']
  simp only [play, Fin.snoc, lt_self_iff_false, ↓reduceDIte, Nat.reduceAdd, cast_eq,
    Nat.succ_eq_add_one]
  congr
  refine funext fun i ↦ ?_
  simp_rw [Fin.fin_one_eq_zero]
  rfl

lemma Strategy.play_two (s : Strategy N) (m : MonsterData N) {k : ℕ} (hk : 2 < k) :
    s.play m k ⟨2, hk⟩ = (s ![(s Fin.elim0).firstMonster m,
      (s ![(s Fin.elim0).firstMonster m]).firstMonster m]).firstMonster m := by
  have hk' : 3 ≤ k := by lia
  rw [s.play_apply_of_le m (by simp : 2 < 3) hk']
  simp only [play, Fin.snoc, lt_self_iff_false, ↓reduceDIte, Nat.reduceAdd, cast_eq,
    Nat.succ_eq_add_one]
  congr
  refine funext fun i ↦ ?_
  fin_cases i
  · rfl
  · have h : (1 : Fin 2) = Fin.last 1 := rfl
    simp only [Fin.snoc_zero, Nat.reduceAdd, Fin.mk_one, Fin.isValue, Matrix.cons_val]
    simp only [h, Fin.snoc_last]
    convert! rfl
    simp_rw [Fin.fin_one_eq_zero, Matrix.cons_val]

lemma Strategy.WinsIn.mono (s : Strategy N) (m : MonsterData N) {k₁ k₂ : ℕ} (h : s.WinsIn m k₁)
    (hk : k₁ ≤ k₂) : s.WinsIn m k₂ := by
  refine Set.mem_of_mem_of_subset h ?_
  rw [Set.range_subset_range_iff_exists_comp]
  exact ⟨Fin.castLE hk, (s.play_comp_castLE m hk).symm⟩

lemma Strategy.ForcesWinIn.mono (s : Strategy N) {k₁ k₂ : ℕ} (h : s.ForcesWinIn k₁)
    (hk : k₁ ≤ k₂) : s.ForcesWinIn k₂ :=
  fun _ ↦ ((h _).mono) hk

/-! ### Proof of lower bound with constructions used therein -/

set_option backward.isDefEq.respectTransparency false in
/-- An arbitrary choice of monster positions, which is modified to put selected monsters in
desired places. -/
def baseMonsterData (N : ℕ) : MonsterData N where
  toFun := fun ⟨r, _, hrN⟩ ↦ ⟨↑r, by
    rw [Fin.le_def] at hrN
    rw [Nat.lt_add_one_iff]
    exact hrN⟩
  inj' := by
    rintro ⟨⟨x, hx⟩, hx1, hxN⟩ ⟨⟨y, hy⟩, hy1, hyN⟩ h
    simp only [Fin.mk.injEq] at h
    simpa using h

/-- Positions for monsters with specified columns in the second and third rows (rows 1 and 2). -/
def monsterData12 (hN : 2 ≤ N) (c₁ c₂ : Fin (N + 1)) : MonsterData N :=
  ((baseMonsterData N).setValue (row2 hN) c₂).setValue (row1 hN) c₁

set_option backward.isDefEq.respectTransparency false in
lemma monsterData12_apply_row2 (hN : 2 ≤ N) {c₁ c₂ : Fin (N + 1)} (h : c₁ ≠ c₂) :
    monsterData12 hN c₁ c₂ (row2 hN) = c₂ := by
  rw [monsterData12, Function.Embedding.setValue_eq_of_ne]
  · exact Function.Embedding.setValue_eq _ _ _
  · simp only [row1, row2, ne_eq, Subtype.mk.injEq]
    simp only [Fin.ext_iff, Fin.val_one]
    lia
  · rw [Function.Embedding.setValue_eq]
    exact h.symm

lemma row1_mem_monsterCells_monsterData12 (hN : 2 ≤ N) (c₁ c₂ : Fin (N + 1)) :
    (1, c₁) ∈ (monsterData12 hN c₁ c₂).monsterCells := by
  exact Set.mem_range_self (row1 hN)

lemma row2_mem_monsterCells_monsterData12 (hN : 2 ≤ N) {c₁ c₂ : Fin (N + 1)} (h : c₁ ≠ c₂) :
    (⟨2, by lia⟩, c₂) ∈ (monsterData12 hN c₁ c₂).monsterCells := by
  convert! Set.mem_range_self (row2 hN)
  exact (monsterData12_apply_row2 hN h).symm

lemma Strategy.not_forcesWinIn_two (s : Strategy N) (hN : 2 ≤ N) : ¬ s.ForcesWinIn 2 := by
  have : NeZero N := ⟨by lia⟩
  have : 0 < N := by lia
  simp only [ForcesWinIn, WinsIn, Set.mem_range, not_forall, not_exists, Option.ne_none_iff_isSome]
  let m1 : Cell N := (s Fin.elim0).findFstEq 1
  let m2 : Cell N := (s ![m1]).findFstEq 2
  let m : MonsterData N := monsterData12 hN m1.2 m2.2
  have h1r : m1.1 = 1 := Path.findFstEq_fst _ _
  have h2r : m2.1 = 2 := Path.findFstEq_fst _ _
  have h1 : m1 ∈ m.monsterCells := by
    convert! row1_mem_monsterCells_monsterData12 hN m1.2 m2.2
  refine ⟨m, fun i ↦ ?_⟩
  fin_cases i
  · simp only [Strategy.play_zero, Path.firstMonster_eq_of_findFstEq_mem h1, Option.isSome_some]
  · simp only [Strategy.play_one]
    suffices ((s ![some m1]).firstMonster m).isSome = true by
      rwa [Path.firstMonster_eq_of_findFstEq_mem h1]
    simp_rw [m]
    by_cases h : m1.2 = m2.2
    · rw [Path.firstMonster_isSome]
      refine ⟨m1, ?_, h1⟩
      have h' : m1 = (⟨(((2 : Fin (N + 2)) : ℕ) - 1 : ℕ), by omega⟩, m2.2) := by
        simpa [Prod.ext_iff, h1r, h2r, h]
      nth_rw 2 [h']
      exact Path.findFstEq_fst_sub_one_mem _ two_ne_zero
    · rw [Path.firstMonster_isSome]
      refine ⟨m2, Path.findFstEq_mem_cells _ _, ?_⟩
      convert! row2_mem_monsterCells_monsterData12 hN h using 1
      simpa [Prod.ext_iff, h2r, Fin.ext_iff]

lemma Strategy.ForcesWinIn.three_le {s : Strategy N} {k : ℕ} (hf : s.ForcesWinIn k)
    (hN : 2 ≤ N) : 3 ≤ k := by
  by_contra hk
  exact s.not_forcesWinIn_two hN (Strategy.ForcesWinIn.mono s hf (by lia))

/-! ### Proof of upper bound and constructions used therein -/

/-- The first attempt in a winning strategy, as a function: first pass along the second row to
locate the monster there. -/
def fn0 (N : ℕ) : Fin (2 * N + 2) → Cell N :=
  fun i ↦ if i = 0 then (0, 0) else
    if h : (i : ℕ) < N + 2 then (1, ⟨(i : ℕ) - 1, by lia⟩) else
    (⟨i - N, by lia⟩, ⟨N, by lia⟩)

lemma injective_fn0 (N : ℕ) : Function.Injective (fn0 N) := by
  intro ⟨a₁, _⟩ ⟨a₂, _⟩
  simp_rw [fn0]
  split_ifs <;> simp [Prod.ext_iff] at * <;> lia

/-- The first attempt in a winning strategy, as a `Path`. -/
def path0 (hN : 2 ≤ N) : Path N := Path.ofFn (fn0 N) (by lia) (by simp [fn0])
  (by simp only [fn0]; split_ifs with h <;> simp at h ⊢ <;> lia)
  (by
    simp only [fn0]
    intro i hi
    split_ifs <;> simp [Adjacent, Nat.dist] at * <;> lia)

/-- The second attempt in a winning strategy, as a function, if the monster in the second row
is not at an edge: pass to the left of that monster then along its column. -/
def fn1OfNotEdge {c₁ : Fin (N + 1)} (hc₁ : c₁ ≠ 0) : Fin (N + 3) → Cell N :=
  fun i ↦ if h : (i : ℕ) ≤ 2 then (⟨(i : ℕ), by lia⟩, ⟨(c₁ : ℕ) - 1, by lia⟩) else
    (⟨(i : ℕ) - 1, by lia⟩, c₁)

/-- The second attempt in a winning strategy, as a function, if the monster in the second row
is not at an edge. -/
def path1OfNotEdge {c₁ : Fin (N + 1)} (hc₁ : c₁ ≠ 0) : Path N := Path.ofFn (fn1OfNotEdge hc₁)
    (by lia) (by simp [fn1OfNotEdge])
    (by simp only [fn1OfNotEdge]; split_ifs <;> simp; lia)
    (by
      simp only [fn1OfNotEdge]
      intro i hi
      rcases c₁ with ⟨c₁, hc₁N⟩
      rw [← Fin.val_ne_iff] at hc₁
      split_ifs <;> simp [Adjacent, Nat.dist] at * <;> lia)

/-- The third attempt in a winning strategy, as a function, if the monster in the second row
is not at an edge: pass to the right of that monster then along its column. -/
def fn2OfNotEdge {c₁ : Fin (N + 1)} (hc₁ : (c₁ : ℕ) ≠ N) : Fin (N + 3) → Cell N :=
  fun i ↦ if h : (i : ℕ) ≤ 2 then (⟨(i : ℕ), by lia⟩, ⟨(c₁ : ℕ) + 1, by lia⟩) else
    (⟨(i : ℕ) - 1, by lia⟩, c₁)

/-- The third attempt in a winning strategy, as a function, if the monster in the second row
is not at an edge. -/
def path2OfNotEdge {c₁ : Fin (N + 1)} (hc₁ : (c₁ : ℕ) ≠ N) : Path N := Path.ofFn (fn2OfNotEdge hc₁)
    (by lia) (by simp [fn2OfNotEdge])
    (by simp only [fn2OfNotEdge]; split_ifs <;> simp; lia)
    (by
      simp only [fn2OfNotEdge]
      intro i hi
      split_ifs <;> simp [Adjacent, Nat.dist] at * <;> lia)

/-- The second attempt in a winning strategy, as a function, if the monster in the second row
is at the left edge: zigzag across the board so that, if we encounter a monster, we have a third
path we know will not encounter a monster. -/
def fn1OfEdge0 (N : ℕ) : Fin (2 * N + 1) → Cell N :=
  fun i ↦ if h : (i : ℕ) = 2 * N then (⟨N + 1, by lia⟩, ⟨N, by lia⟩) else
    (⟨((i : ℕ) + 1) / 2, by lia⟩, ⟨(i : ℕ) / 2 + 1, by lia⟩)

/-- The second attempt in a winning strategy, as a `Path`, if the monster in the second row
is at the left edge. -/
def path1OfEdge0 (hN : 2 ≤ N) : Path N := Path.ofFn (fn1OfEdge0 N) (by lia)
    (by simp only [fn1OfEdge0]; split_ifs <;> simp; lia)
    (by simp [fn1OfEdge0])
    (by
      simp only [fn1OfEdge0]
      intro i hi
      split_ifs <;> simp [Adjacent, Nat.dist] at * <;> lia)

/-- The second attempt in a winning strategy, as a `Path`, if the monster in the second row
is at the right edge. -/
def path1OfEdgeN (hN : 2 ≤ N) : Path N := (path1OfEdge0 hN).reflect

/-- The third attempt in a winning strategy, as a function, if the monster in the second row
is at the left edge and the second (zigzag) attempt encountered a monster: on the row before
the monster was encountered, move to the following row one place earlier, proceed to the left
edge and then along that column. -/
def fn2OfEdge0 {r : Fin (N + 2)} (hr : (r : ℕ) ≤ N) : Fin (N + 2 * r - 1) → Cell N :=
  fun i ↦ if h₁ : (i : ℕ) + 2 < 2 * (r : ℕ) then
    (⟨((i : ℕ) + 1) / 2, by lia⟩, ⟨(i : ℕ) / 2 + 1, by lia⟩) else
    if h₂ : (i : ℕ) + 2 < 3 * (r : ℕ) then (r, ⟨3 * (r : ℕ) - 3 - (i : ℕ), by lia⟩) else
    (⟨i + 3 - 2 * r, by lia⟩, 0)

lemma fn2OfEdge0_apply_eq_fn1OfEdge0_apply_of_lt {r : Fin (N + 2)} (hr : (r : ℕ) ≤ N) {i : ℕ}
    (h : i + 2 < 2 * (r : ℕ)) : fn2OfEdge0 hr ⟨i, by lia⟩ = fn1OfEdge0 N ⟨i, by lia⟩ := by
  rw [fn1OfEdge0, fn2OfEdge0]
  split_ifs with h₁ h₂ <;> simp [Prod.ext_iff] at * <;> lia

/-- The third attempt in a winning strategy, as a `Path`, if the monster in the second row
is at the left edge and the second (zigzag) attempt encountered a monster. -/
def path2OfEdge0 (hN : 2 ≤ N) {r : Fin (N + 2)} (hr2 : 2 ≤ (r : ℕ)) (hrN : (r : ℕ) ≤ N) : Path N :=
  Path.ofFn (fn2OfEdge0 hrN) (by lia)
    (by simp only [fn2OfEdge0]; split_ifs <;> simp <;> lia)
    (by simp only [fn2OfEdge0]; split_ifs <;> simp <;> lia)
    (by
      simp only [fn2OfEdge0]
      intro i hi
      split_ifs <;> simp [Adjacent, Nat.dist] at * <;> lia)

/-- The third attempt in a winning strategy, as a `Path`, if the monster in the second row
is at the left edge and the second (zigzag) attempt encountered a monster, version that works
with junk values of `r` for convenience in defining the strategy before needing to prove things
about exactly where it can encounter monsters. -/
def path2OfEdge0Def (hN : 2 ≤ N) (r : Fin (N + 2)) : Path N :=
  if h : 2 ≤ (r : ℕ) ∧ (r : ℕ) ≤ N then path2OfEdge0 hN h.1 h.2 else
    path2OfEdge0 hN (r := ⟨2, by lia⟩) (le_refl _) hN

/-- The third attempt in a winning strategy, as a `Path`, if the monster in the second row
is at the right edge and the second (zigzag) attempt encountered a monster. -/
def path2OfEdgeNDef (hN : 2 ≤ N) (r : Fin (N + 2)) : Path N :=
  (path2OfEdge0Def hN r).reflect

/-- The second attempt in a winning strategy, given the column of the monster in the second row,
as a `Path`. -/
def path1 (hN : 2 ≤ N) (c₁ : Fin (N + 1)) : Path N :=
  if hc₁ : c₁ = 0 then path1OfEdge0 hN else
    if (c₁ : ℕ) = N then path1OfEdgeN hN else
    path1OfNotEdge hc₁

/-- The third attempt in a winning strategy, given the column of the monster in the second row
and the row of the monster (if any) found in the second attempt, as a `Path`. -/
def path2 (hN : 2 ≤ N) (c₁ : Fin (N + 1)) (r : Fin (N + 2)) : Path N :=
  if c₁ = 0 then path2OfEdge0Def hN r else
    if hc₁N : (c₁ : ℕ) = N then path2OfEdgeNDef hN r else
    path2OfNotEdge hc₁N

/-- A strategy that wins in three attempts. -/
def winningStrategy (hN : 2 ≤ N) : Strategy N
  | 0 => fun _ ↦ path0 hN
  | 1 => fun r => path1 hN ((r 0).getD 0).2
  | _ + 2 => fun r => path2 hN ((r 0).getD 0).2 ((r 1).getD 0).1

set_option backward.isDefEq.respectTransparency false in
lemma path0_firstMonster_eq_apply_row1 (hN : 2 ≤ N) (m : MonsterData N) :
    (path0 hN).firstMonster m = some (1, m (row1 hN)) := by
  simp_rw [path0, Path.firstMonster, Path.ofFn]
  have h : (1, m (row1 hN)) = fn0 N ⟨(m (row1 hN) : ℕ) + 1, by lia⟩ := by
    simp_rw [fn0]
    split_ifs <;> simp [Prod.ext_iff] at *; lia
  rw [h, List.find?_ofFn_eq_some_of_injective (injective_fn0 N)]
  refine ⟨?_, fun j hj ↦ ?_⟩
  · rw [fn0]
    split_ifs
    · simp [Prod.ext_iff] at *
    · have hm1 : (1, m (row1 hN)) ∈ m.monsterCells := Set.mem_range_self (row1 hN)
      simpa [Prod.ext_iff] using hm1
  · rw [fn0]
    split_ifs with h₁
    · simp [h₁, MonsterData.monsterCells]
    · have hjN2 : (j : ℕ) < N + 2 := by
        rw [Fin.lt_def] at hj
        refine hj.trans ?_
        simp
      simp only [MonsterData.monsterCells, h₁, ↓reduceIte, hjN2, ↓reduceDIte, Set.mem_range,
        Prod.mk.injEq, Fin.ext_iff, Fin.val_one, Subtype.exists, Set.mem_Icc, exists_and_left,
        decide_eq_true_eq, not_exists, not_and]
      rintro i hi hix hij
      simp only [Fin.ext_iff, Fin.val_zero] at h₁
      rw [eq_comm, Nat.sub_eq_iff_eq_add (by lia)] at hij
      have hr : row1 hN = ⟨↑i, hix⟩ := by
        simp_rw [Subtype.ext_iff, Fin.ext_iff, hi]
        rfl
      rw [Fin.lt_def, hij, add_lt_add_iff_right, Fin.val_fin_lt, hr] at hj
      exact Nat.lt_irrefl _ hj

lemma winningStrategy_play_one (hN : 2 ≤ N) (m : MonsterData N) :
    (winningStrategy hN).play m 3 ⟨1, by simp⟩ =
      (path1 hN (m (row1 hN))).firstMonster m := by
  simp_rw [Strategy.play_one, winningStrategy, path0_firstMonster_eq_apply_row1]
  rfl

lemma winningStrategy_play_two (hN : 2 ≤ N) (m : MonsterData N) :
    (winningStrategy hN).play m 3 ⟨2, by simp⟩ =
      (path2 hN (m (row1 hN))
             (((path1 hN (m (row1 hN))).firstMonster m).getD 0).1).firstMonster m := by
  simp_rw [Strategy.play_two, winningStrategy, path0_firstMonster_eq_apply_row1]
  rfl

lemma path1_firstMonster_of_not_edge (hN : 2 ≤ N) {m : MonsterData N} (hc₁0 : m (row1 hN) ≠ 0)
    (hc₁N : (m (row1 hN) : ℕ) ≠ N) :
    (path1 hN (m (row1 hN))).firstMonster m = none ∨
      (path1 hN (m (row1 hN))).firstMonster m =
        some (⟨2, by lia⟩, ⟨(m (row1 hN) : ℕ) - 1, by omega⟩) := by
  suffices h : ∀ c ∈ (path1 hN (m (row1 hN))).cells, c ∉ m.monsterCells ∨
      c = (⟨2, by lia⟩, ⟨(m (row1 hN) : ℕ) - 1, by omega⟩) by
    simp only [Path.firstMonster]
    by_cases hn : List.find? (fun x ↦ decide (x ∈ m.monsterCells))
                             (path1 hN (m (row1 hN))).cells = none
    · exact .inl hn
    · rw [← ne_eq, Option.ne_none_iff_exists'] at hn
      rcases hn with ⟨x, hx⟩
      have hxm := List.find?_some hx
      have hx' := h _ (List.mem_of_find?_eq_some hx)
      simp only [decide_eq_true_eq] at hxm
      simp only [hxm, not_true_eq_false, false_or] at hx'
      rw [hx'] at hx
      exact .inr hx
  simp only [path1, hc₁0, ↓reduceDIte, hc₁N, ↓reduceIte, path1OfNotEdge, Path.ofFn_cells,
    List.forall_mem_ofFn_iff, fn1OfNotEdge]
  intro j
  split_ifs with h
  · rcases j with ⟨j, hj⟩
    simp only at h
    interval_cases j
    · exact .inl (m.notMem_monsterCells_of_fst_eq_zero rfl)
    · simp only [Fin.mk_one, Prod.mk.injEq, Fin.ext_iff, Fin.val_one, OfNat.one_ne_ofNat,
        and_true, or_false]
      have hN' : 1 ≤ N := by lia
      rw [m.mk_mem_monsterCells_iff_of_le (le_refl _) hN', ← ne_eq, ← Fin.val_ne_iff]
      rw [← Fin.val_ne_iff] at hc₁0
      exact (Nat.sub_one_lt hc₁0).ne'
    · exact .inr rfl
  · refine .inl ?_
    simp only [MonsterData.monsterCells, Set.mem_range, Prod.mk.injEq, Fin.ext_iff,
      EmbeddingLike.apply_eq_iff_eq, exists_eq_right, coe_coe_row1]
    lia

lemma path2_firstMonster_of_not_edge (hN : 2 ≤ N) {m : MonsterData N} (hc₁0 : m (row1 hN) ≠ 0)
    (hc₁N : (m (row1 hN) : ℕ) ≠ N) (r : Fin (N + 2)) :
    (path2 hN (m (row1 hN)) r).firstMonster m = none ∨
      (path2 hN (m (row1 hN)) r).firstMonster m =
        some (⟨2, by lia⟩, ⟨(m (row1 hN) : ℕ) + 1, by lia⟩) := by
  suffices h : ∀ c ∈ (path2 hN (m (row1 hN)) r).cells, c ∉ m.monsterCells ∨
      c = (⟨2, by lia⟩, ⟨(m (row1 hN) : ℕ) + 1, by lia⟩) by
    simp only [Path.firstMonster]
    by_cases hn : List.find? (fun x ↦ decide (x ∈ m.monsterCells))
                             (path2 hN (m (row1 hN)) r).cells = none
    · exact .inl hn
    · rw [← ne_eq, Option.ne_none_iff_exists'] at hn
      rcases hn with ⟨x, hx⟩
      have hxm := List.find?_some hx
      have hx' := h _ (List.mem_of_find?_eq_some hx)
      simp only [decide_eq_true_eq] at hxm
      simp only [hxm, not_true_eq_false, false_or] at hx'
      rw [hx'] at hx
      exact .inr hx
  simp only [path2, hc₁0, ↓reduceDIte, hc₁N, ↓reduceIte, path2OfNotEdge, Path.ofFn_cells,
    List.forall_mem_ofFn_iff, fn2OfNotEdge]
  intro j
  split_ifs with h
  · rcases j with ⟨j, hj⟩
    simp only at h
    interval_cases j
    · exact .inl (m.notMem_monsterCells_of_fst_eq_zero rfl)
    · simp only [Fin.mk_one, Prod.mk.injEq, Fin.ext_iff, Fin.val_one, OfNat.one_ne_ofNat,
        and_true, or_false]
      have hN' : 1 ≤ N := by lia
      rw [m.mk_mem_monsterCells_iff_of_le (le_refl _) hN', ← ne_eq, ← Fin.val_ne_iff]
      exact Nat.ne_add_one _
    · exact .inr rfl
  · refine .inl ?_
    simp only [MonsterData.monsterCells, Set.mem_range, Prod.mk.injEq, Fin.ext_iff,
      EmbeddingLike.apply_eq_iff_eq, exists_eq_right, coe_coe_row1]
    lia

lemma winningStrategy_play_one_eq_none_or_play_two_eq_none_of_not_edge (hN : 2 ≤ N)
    {m : MonsterData N} (hc₁0 : m (row1 hN) ≠ 0) (hc₁N : (m (row1 hN) : ℕ) ≠ N) :
    (winningStrategy hN).play m 3 ⟨1, by simp⟩ = none ∨
      (winningStrategy hN).play m 3 ⟨2, by simp⟩ = none := by
  rw [winningStrategy_play_one, winningStrategy_play_two]
  have h1 := path1_firstMonster_of_not_edge hN hc₁0 hc₁N
  rcases h1 with h1 | h1
  · exact .inl h1
  · refine .inr ?_
    have h2 := path2_firstMonster_of_not_edge hN hc₁0 hc₁N
    rcases h2 (((path1 hN (m (row1 hN))).firstMonster m).getD 0).1 with h2r | h2r
    · exact h2r
    · have h1' := (Path.mem_of_firstMonster_eq_some h1).2
      have h2' := (Path.mem_of_firstMonster_eq_some h2r).2
      have h12 : 1 ≤ (⟨2, by lia⟩ : Fin (N + 2)) := by
        rw [Fin.le_def]
        norm_num
      rw [m.mk_mem_monsterCells_iff_of_le h12 hN] at h1' h2'
      rw [h1'] at h2'
      simp at h2'

lemma path2OfEdge0_firstMonster_eq_none_of_path1OfEdge0_firstMonster_eq_some (hN : 2 ≤ N)
    {x : Cell N} (hx2 : 2 ≤ (x.1 : ℕ)) (hxN : (x.1 : ℕ) ≤ N) {m : MonsterData N}
    (hc₁0 : m (row1 hN) = 0) (hx : (path1OfEdge0 hN).firstMonster m = some x) :
    (path2OfEdge0 hN hx2 hxN).firstMonster m = none := by
  rw [path2OfEdge0, Path.ofFn_firstMonster_eq_none]
  rw [path1OfEdge0, Path.firstMonster, Path.ofFn_cells, List.find?_ofFn_eq_some] at hx
  simp only [decide_eq_true_eq] at hx
  rcases hx with ⟨hx, i, hix, hnm⟩
  have hi : (x.1 : ℕ) = ((i : ℕ) + 1) / 2 := by
    rw [fn1OfEdge0] at hix
    split_ifs at hix <;> simp [Prod.ext_iff, Fin.ext_iff] at hix <;> lia
  have hi' : (i : ℕ) ≠ 2 * N := by
    intro h
    rw [← hix, fn1OfEdge0, dite_eq_left h] at hx
    have h' := MonsterData.le_N_of_mem_monsterCells hx
    simp at h'
  intro j
  by_cases h : (j : ℕ) + 2 < 2 * (x.1 : ℕ)
  · rw [fn2OfEdge0_apply_eq_fn1OfEdge0_apply_of_lt hxN h]
    simp_rw [Fin.lt_def] at hnm
    refine hnm _ ?_
    simp only
    lia
  · rw [fn2OfEdge0, dite_eq_right h]
    split_ifs with h'
    · have hx1 : 1 ≤ x.1 := by
        rw [Fin.le_def, Fin.val_one]
        lia
      rw [MonsterData.mk_mem_monsterCells_iff_of_le hx1 hxN]
      rw [MonsterData.mem_monsterCells_iff_of_le hx1 hxN] at hx
      simp_rw [hx, ← hix, fn1OfEdge0]
      split_ifs <;> simp <;> lia
    · rw [MonsterData.mk_mem_monsterCells_iff]
      simp only [not_exists]
      intro h1 hN'
      rw [← hc₁0]
      refine m.injective.ne ?_
      rw [← Subtype.coe_ne_coe, ← Fin.val_ne_iff, coe_coe_row1]
      simp only
      lia

lemma winningStrategy_play_one_eq_none_or_play_two_eq_none_of_edge_zero (hN : 2 ≤ N)
    {m : MonsterData N} (hc₁0 : m (row1 hN) = 0) :
    (winningStrategy hN).play m 3 ⟨1, by simp⟩ = none ∨
      (winningStrategy hN).play m 3 ⟨2, by simp⟩ = none := by
  rw [or_iff_not_imp_left]
  intro h
  rw [← ne_eq, Option.ne_none_iff_exists] at h
  rcases h with ⟨x, hx⟩
  rw [winningStrategy_play_one] at hx
  rw [winningStrategy_play_two, ← hx, Option.getD_some]
  rw [path1, dite_eq_left hc₁0] at hx
  have h1 := Path.mem_of_firstMonster_eq_some (hx.symm)
  have hx2N : 2 ≤ (x.1 : ℕ) ∧ (x.1 : ℕ) ≤ N := by
    rw [path1OfEdge0, Path.ofFn_cells, List.mem_ofFn] at h1
    rcases h1 with ⟨⟨i, rfl⟩, hm⟩
    refine ⟨?_, m.le_N_of_mem_monsterCells hm⟩
    rcases i with ⟨i, hi⟩
    have hi : 3 ≤ i := by
      by_contra hi
      interval_cases i <;> rw [fn1OfEdge0] at hm <;> split_ifs at hm with h
      · simp at h
        lia
      · simp_rw [zero_add] at hm
        exact m.notMem_monsterCells_of_fst_eq_zero rfl hm
      · simp [eqComm] at h
      · dsimp only [Nat.reduceAdd, Nat.reduceDiv, Fin.mk_one] at hm
        have h1N : 1 ≤ N := by lia
        rw [m.mk_mem_monsterCells_iff_of_le (le_refl _) h1N] at hm
        rw [row1, hm, Fin.ext_iff] at hc₁0
        simp at hc₁0
      · simp at h
        lia
      · simp only at hm
        norm_num at hm
        have h1N : 1 ≤ N := by lia
        rw [m.mk_mem_monsterCells_iff_of_le (le_refl _) h1N] at hm
        rw [row1, hm] at hc₁0
        simp at hc₁0
    rw [fn1OfEdge0]
    split_ifs <;> simp <;> lia
  rw [path2, ite_eq_left hc₁0, path2OfEdge0Def, dite_eq_left hx2N]
  exact path2OfEdge0_firstMonster_eq_none_of_path1OfEdge0_firstMonster_eq_some hN hx2N.1
    hx2N.2 hc₁0 hx.symm

set_option backward.isDefEq.respectTransparency false in
lemma winningStrategy_play_one_of_edge_N (hN : 2 ≤ N) {m : MonsterData N}
    (hc₁N : (m (row1 hN) : ℕ) = N) : (winningStrategy hN).play m 3 ⟨1, by simp⟩ =
      ((winningStrategy hN).play m.reflect 3 ⟨1, by simp⟩).map Cell.reflect := by
  have hc₁0 : m (row1 hN) ≠ 0 := by
    rw [← Fin.val_ne_iff, hc₁N, Fin.val_zero]
    lia
  have hc₁r0 : m.reflect (row1 hN) = 0 := by
    simp only [MonsterData.reflect, Function.Embedding.coeFn_mk, Function.comp_apply,
      ← Fin.rev_last, Fin.rev_inj]
    rw [Fin.ext_iff]
    exact hc₁N
  simp_rw [winningStrategy_play_one hN, path1, path1OfEdgeN, dite_eq_right hc₁0, ite_eq_left hc₁N,
    dite_eq_left hc₁r0, ← Path.firstMonster_reflect, MonsterData.reflect_reflect]

set_option backward.isDefEq.respectTransparency false in
lemma winningStrategy_play_two_of_edge_N (hN : 2 ≤ N) {m : MonsterData N}
    (hc₁N : (m (row1 hN) : ℕ) = N) : (winningStrategy hN).play m 3 ⟨2, by simp⟩ =
      ((winningStrategy hN).play m.reflect 3 ⟨2, by simp⟩).map Cell.reflect := by
  have hc₁0 : m (row1 hN) ≠ 0 := by
    rw [← Fin.val_ne_iff, hc₁N, Fin.val_zero]
    lia
  have hc₁r0 : m.reflect (row1 hN) = 0 := by
    simp only [MonsterData.reflect, Function.Embedding.coeFn_mk, Function.comp_apply,
      ← Fin.rev_last, Fin.rev_inj]
    rw [Fin.ext_iff]
    exact hc₁N
  simp_rw [winningStrategy_play_two hN, path1, path1OfEdgeN, path2, path2OfEdgeNDef,
    ite_eq_right hc₁0, dite_eq_right hc₁0, ite_eq_left hc₁N, dite_eq_left hc₁N, ite_eq_left hc₁r0,
    dite_eq_left hc₁r0, ← Path.firstMonster_reflect, MonsterData.reflect_reflect]
  convert! rfl using 4
  nth_rw 2 [← m.reflect_reflect]
  rw [Path.firstMonster_reflect]
  rcases ((path1OfEdge0 hN).firstMonster m.reflect).eq_none_or_eq_some with h | h
  · simp [h]
  · rcases h with ⟨x, hx⟩
    simp [hx, Cell.reflect]

set_option backward.isDefEq.respectTransparency false in
lemma winningStrategy_play_one_eq_none_or_play_two_eq_none_of_edge_N (hN : 2 ≤ N)
    {m : MonsterData N} (hc₁N : (m (row1 hN) : ℕ) = N) :
    (winningStrategy hN).play m 3 ⟨1, by simp⟩ = none ∨
      (winningStrategy hN).play m 3 ⟨2, by simp⟩ = none := by
  simp_rw [winningStrategy_play_one_of_edge_N hN hc₁N, winningStrategy_play_two_of_edge_N hN hc₁N,
    Option.map_eq_none_iff]
  have hc₁r0 : m.reflect (row1 hN) = 0 := by
    simp only [MonsterData.reflect, Function.Embedding.coeFn_mk, Function.comp_apply,
      ← Fin.rev_last, Fin.rev_inj]
    rw [Fin.ext_iff]
    exact hc₁N
  exact winningStrategy_play_one_eq_none_or_play_two_eq_none_of_edge_zero hN hc₁r0

lemma winningStrategy_play_one_eq_none_or_play_two_eq_none (hN : 2 ≤ N) (m : MonsterData N) :
    (winningStrategy hN).play m 3 ⟨1, by simp⟩ = none ∨
      (winningStrategy hN).play m 3 ⟨2, by simp⟩ = none := by
  by_cases hc₁0 : m (row1 hN) = 0
  · exact winningStrategy_play_one_eq_none_or_play_two_eq_none_of_edge_zero hN hc₁0
  · by_cases hc₁N : (m (row1 hN) : ℕ) = N
    · exact winningStrategy_play_one_eq_none_or_play_two_eq_none_of_edge_N hN hc₁N
    · exact winningStrategy_play_one_eq_none_or_play_two_eq_none_of_not_edge hN hc₁0 hc₁N

lemma winningStrategy_forcesWinIn_three (hN : 2 ≤ N) :
    (winningStrategy hN).ForcesWinIn 3 := by
  intro m
  rcases winningStrategy_play_one_eq_none_or_play_two_eq_none hN m with h | h
  · rw [Strategy.WinsIn]
    convert! Set.mem_range_self (⟨1, by simp⟩ : Fin 3)
    exact h.symm
  · rw [Strategy.WinsIn]
    convert! Set.mem_range_self (⟨2, by simp⟩ : Fin 3)
    exact h.symm

/-- This is to be determined by the solver of the original problem (and much of the difficulty
for contestants is in finding this answer rather than spending time attempting to prove a
conjectured answer around log₂ 2023). -/
def answer : ℕ := 3

/-- The final result, combining upper and lower bounds. -/
theorem result : IsLeast {k | ∃ s : Strategy 2022, s.ForcesWinIn k} answer := by
  simp_rw [IsLeast, mem_lowerBounds, Set.mem_ofPred, forall_exists_index]
  exact ⟨⟨winningStrategy (by simp), winningStrategy_forcesWinIn_three (by simp)⟩,
    fun k s h ↦ h.three_le (by simp)⟩

end Imo2024Q5