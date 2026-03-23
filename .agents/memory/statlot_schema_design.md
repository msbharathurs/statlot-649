# StatLot 649 — Entity Schema Redesign
## Date: 2026-03-23

## App ID: 69af461ff68942d25938a9b3

---

## Entities

### 1. Draw
One record per individual lottery draw. Replaces the monolithic DrawData blob.

Fields:
- draw_number (integer) — e.g. 4154
- draw_date (string) — ISO date string e.g. "2024-03-02"
- n1..n6 (integer) — the 6 main numbers
- additional (integer) — the bonus/additional number
- sum (integer) — sum of 6 numbers
- odd_count (integer) — count of odd numbers
- even_count (integer) — count of even numbers
- low_count (integer) — numbers 1-24
- high_count (integer) — numbers 25-49
- decade_1 (integer) — count in 1-10
- decade_2 (integer) — count in 11-20
- decade_3 (integer) — count in 21-30
- decade_4 (integer) — count in 31-40
- decade_5 (integer) — count in 41-49
- consecutive_count (integer) — pairs of consecutive numbers
- repeat_from_prev (integer) — how many numbers repeated from previous draw
- source (string) — "manual", "import", "verified"
- notes (string) — any notes

### 2. Prediction
One record per prediction session (replaces localStorage lotto_saved_predictions).

Fields:
- draw_label (string) — label for next expected draw e.g. "Draw 4155"
- generated_date (string) — ISO date when generated
- combos (array) — array of 6-number arrays, the top predicted combinations
- additional_picks (array) — top predicted additional numbers
- tuned_weights_active (boolean) — whether tuned weights were used
- total_draws_used (integer) — how many historical draws fed into engine
- engine_version (string) — version tag of the engine config used
- verified (boolean) — whether actual result has been entered
- actual_draw (array) — the 6 actual numbers after draw happens
- actual_additional (integer) — actual additional number
- best_match (integer) — best match count across all combos
- match_details (array) — per-combo match details [{combo, matches, matched_numbers}]
- notes (string)

### 3. EngineConfig
Stores the active and historical weight configurations for the prediction engine.
Replaces localStorage lotto_tuned_weights.

Fields:
- version (string) — e.g. "v1", "tuned-2026-03-23"
- is_active (boolean) — which config is currently in use
- freq_weight (number)
- pair_weight (number)
- triplet_weight (number)
- aging_weight (number)
- markov_weight (number)
- hot_cold_weight (number)
- decade_weight (number)
- odd_even_weight (number)
- gap_weight (number)
- additional_weight (number)
- source (string) — "manual", "backtest_tuned", "default"
- backtest_result_id (string) — reference to BacktestResult if tuned
- notes (string)

### 4. BacktestResult
Stores the outcome of each backtest run for historical tracking and weight tuning.
Replaces ephemeral backtest data.

Fields:
- run_date (string) — ISO date of when backtest was run
- test_count (integer) — number of draws tested
- pred_count (integer) — predictions generated per test draw
- avg_match (number) — average matches across all test draws
- best_match (integer) — best single draw match count
- three_plus_rate (number) — % of draws with 3+ matches
- random_avg_match (number) — baseline random avg match
- lift_pct (number) — % improvement over random
- tuned_weights (object) — the resulting tuned weights object
- draw_range_start (integer) — first draw number tested
- draw_range_end (integer) — last draw number tested
- engine_config_id (string) — which engine config was used as input
- notes (string)

---

## Migration Plan
1. Parse existing DrawData raw_text blob → individual Draw records
2. Keep DrawData entity temporarily for backward compatibility, mark deprecated
3. Eventually remove DrawData after migration confirmed

