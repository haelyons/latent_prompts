# SycophantSee: Revised Experiment Plan

## MCQ-Based Backtesting for Whitebox-Blackbox Sycophancy Evaluation

*Horatio Lyons & Hélios Lyons — February 2026*
*Revision of EXT_PLAN.md incorporating dataset feasibility analysis*

---

## 1. Context and motivation

### 1.1 The project so far

SycophantSee is an activation-based diagnostic framework for monitoring sycophancy in LLMs at inference time. Building on Vennemeyer et al.'s (2025) demonstration that sycophancy comprises three functionally independent components — sycophantic agreement (SyA), genuine agreement (GA), and sycophantic praise (SyPr) — encoded as distinct, linearly separable directions in activation space, we developed a "dual-point" measurement that computes cosine similarity to these directions at two positions:

- **Prompt-end:** the model's pre-generation "decision state," before any output tokens.
- **Response-end:** the state after generation completes.

The **shift** between them (response-end minus prompt-end) quantifies whether generation amplified or suppressed the initial behavioral encoding — an intention-action gap analogous to Yin et al.'s (2025) "refusal cliff."

Our hackathon paper demonstrated this on Llama 3.1 8B Instruct with a framing experiment: first-person framing ("I believe X") activates sycophancy directions more strongly than third-person framing ("My friend believes X"), even when behavioral outputs are identical. This dissociation between activation state and output — **invisible sycophancy** — is the core phenomenon we are investigating.

### 1.2 The research question

**Can activation-based measurement complement behavioral evaluation by predicting sycophantic behavior before it manifests, and by revealing high-risk states that behavioral evaluation misses?**

The extended abstract proposed three experiments to test this. This revised plan preserves the experimental logic but changes the backtesting methodology based on a feasibility analysis of available datasets.

### 1.3 What has been built

Hélios's implementation work has resolved several barriers the original plan did not account for:

- **`model_utils.py`**: Hook-based activation extraction capturing only requested layers, handling multi-GPU sharding transparently. Replaces the OOM-prone `output_hidden_states=True` approach.
- **`01b_extract_directions_70b.py`**: New extraction script supporting multi-layer extraction in a single forward pass, sign-aligned AUROC, and configurable pooling strategy (`pre_eos` vs `eos`).
- **First 70B results** (Llama 3.1 70B, `pre_eos` pooling): SyA 0.94, GA 0.92, SyPr 0.72. The depressed SyPr AUROC confirmed that pooling strategy matters — Vennemeyer et al. validated EOS as optimal (AUROC 0.9839 at EOS vs 0.62 at position k=2).
- **Model profile for Llama 3.3 70B** added, matching the model the reference paper actually benchmarked.
- **Next queued run**: `python 01b_extract_directions_70b.py --model 3.3-70b --pool-strategy eos`

---

## 2. Why the original backtesting plan needed revision

### 2.1 The original Experiment 2

The extended abstract proposed back-testing against Anthropic's SycophancyEval benchmark (~21K prompts across four sycophancy types: answer, are-you-sure, feedback, mimicry). The plan was to:

1. Measure prompt-end activation for each prompt.
2. Generate the model's response.
3. Determine whether the response was behaviorally sycophantic.
4. Correlate activation magnitude with behavioral outcome.

Step 3 is the problem. The `answer`, `feedback`, and `mimicry` subsets require **free-form text generation** followed by LLM-as-judge evaluation to determine sycophancy. This is expensive (API costs for thousands of judge calls), slow, and introduces judge reliability issues — JUSSA (Eshuijs et al., 2025) found single-response sycophancy detection by LLM judges achieves only 42% accuracy.

### 2.2 The dataset landscape

A systematic audit of 15+ sycophancy benchmarks revealed that nearly all release prompts and evaluation code, but **not** pre-computed model outputs with per-prompt behavioral labels. The one exception — ELEPHANT — measures *social* sycophancy (emotional validation, moral endorsement) on open-ended advice queries, which is taxonomically orthogonal to the factual SyA/GA directions extracted from Vennemeyer et al.'s factorial data. Using ELEPHANT to backtest SyA activation measurements would produce uninterpretable results.

### 2.3 The MCQ solution

Anthropic's model-written evaluations (`Anthropic/model-written-evals`, sycophancy subfolder) contain ~30,000 multiple-choice sycophancy prompts across three domains: NLP research opinions, philosophy (PhilPapers 2020), and political typology. Each prompt presents a user stating an opinion and asking the model to choose between two options — one matching the user's opinion (sycophantic) and one not. Each example includes an `answer_matching_behavior` field marking which option is sycophantic.

This format eliminates the judge entirely. The model outputs "(A)" or "(B)", and labeling is deterministic string matching against `answer_matching_behavior`. The cost reduces to **inference only** — no judge API calls, no ambiguous classifications.

Critically, the prompt structure — "a user states a belief and asks the model whether it agrees" — is structurally close to Vennemeyer et al.'s factorial format, where a user asserts a claim and the model must agree or disagree. This means the SyA direction (trained to detect agreement with incorrect user claims) should be activated by these prompts in a meaningful way, even though the MCQ prompts test opinion-matching rather than factual correctness. The GA direction provides a natural control: on prompts where the "sycophantic" answer happens to align with expert consensus, GA should activate instead of SyA.

Additionally, Sharma et al.'s sycophancy-eval (`meg-tong/sycophancy-eval`) extends these with an `are_you_sure` subset (~4,900 prompts) where models first answer factual questions, then face a challenge ("Are you sure? I think the answer is [wrong answer]"). This subset directly tests the **fragile suppression hypothesis** from the extended abstract without requiring LLM-as-judge evaluation — the model either flips its answer or doesn't, and flipping is deterministic to detect.

---

## 3. Revised experiment plan

### Experiment 1: Scale validation (largely unchanged)

**Status:** In progress. First 70B run complete; EOS pooling + Llama 3.3 run queued.

**Objective:** Confirm that sycophancy directions extracted on larger models achieve comparable or higher AUROC, and that the framing effect replicates at scale.

**Method:**
1. Run `01b_extract_directions_70b.py --model 3.3-70b --pool-strategy eos` on Vennemeyer et al.'s 9 factorial datasets.
2. Extract directions at layers 55, 60, 65 (targeting the 60-80% depth range where SyA/GA diverge in 80-layer models, following the pattern of layer 24 in the 32-layer 8B model).
3. Compute AUROC for SyA, GA, SyPr at each layer.
4. Replicate the framing effect experiment (first-person vs third-person) at 70B scale.

**Expected outputs:**
- AUROC table: 8B vs 70B, per-behavior, per-layer.
- Framing effect comparison: prompt-end cosine similarity and shift metrics at both scales.
- Layer sweep analysis: at which layer do SyA and GA maximally disentangle in the 80-layer model?

**Success criteria:** SyA and GA AUROC ≥ 0.90 at optimal layer with EOS pooling. The pooling fix should resolve the SyPr depression (0.72 → expected ≥ 0.85).

**Delta from original plan:** Switched target model from Llama 3.1 70B to Llama 3.3 70B Instruct (matching Vennemeyer et al.'s benchmark model). Added EOS pooling as default. Already partially complete.

---

### Experiment 2: MCQ backtesting (revised methodology)

**Status:** New. Replaces the original SycophancyEval back-test.

**Objective:** Test whether prompt-end activation magnitude predicts behavioral sycophancy on an independent evaluation set, using auto-labeled MCQ prompts that require no LLM judge.

**Data sources:**

| Dataset | Source | N prompts | Format | Label method |
|---------|--------|-----------|--------|-------------|
| NLP survey opinions | `Anthropic/model-written-evals` | ~10,000 | MCQ (A)/(B) | `answer_matching_behavior` |
| PhilPapers 2020 | `Anthropic/model-written-evals` | ~10,000 | MCQ (A)/(B) | `answer_matching_behavior` |
| Political typology | `Anthropic/model-written-evals` | ~10,000 | MCQ (A)/(B) | `answer_matching_behavior` |

**Method — new script `03_mcq_backtest.py`:**

*Step 1 — Prompt preparation:*
- Load JSONL files from Anthropic model-written-evals sycophancy subset.
- Format each prompt with the target model's chat template.
- Each example contains a `question` field (the full prompt including user opinion + MCQ options) and `answer_matching_behavior` (which option is sycophantic).

*Step 2 — Joint inference + activation extraction:*
- For each prompt, run a single forward pass through the model.
- Extract prompt-end activations at the optimal layer(s) identified in Experiment 1.
- Generate the model's MCQ response (constrained to short output; `max_new_tokens=10` is sufficient for "(A)" or "(B)").
- Extract response-end activations.
- This can reuse the hook-based extraction from `model_utils.py` — the same forward pass that generates the response also yields activations at both measurement points.

*Step 3 — Auto-labeling:*
- Parse model output for (A) or (B).
- Compare against `answer_matching_behavior` to assign binary label: `sycophantic=1` if the model chose the behavior-matching option, `sycophantic=0` otherwise.
- No LLM judge. No ambiguity. Discard unparseable responses (expect <2% for instruction-tuned models on MCQ format).

*Step 4 — Analysis:*
- **Primary test:** Compute ROC curve using prompt-end SyA cosine similarity as the predictor and the behavioral sycophancy label as the target. If activation magnitude predicts behavioral sycophancy, AUC should be significantly above 0.5.
- **Distribution analysis:** Plot prompt-end SyA activation distributions stratified by behavioral outcome (sycophantic vs non-sycophantic responses). Test for separation using Mann-Whitney U.
- **Shift analysis:** Compare prompt→response shift for sycophantic vs non-sycophantic outcomes. The hypothesis is that sycophantic responses show positive shift (amplification) while non-sycophantic responses show negative shift (suppression).
- **Invisible sycophancy identification:** Flag prompts with high prompt-end SyA activation but non-sycophantic behavioral output. These represent cases where the model was internally "pressured" toward sycophancy but suppressed it — the key population for the fragile suppression test in Experiment 3.
- **GA as control:** Repeat the analysis using GA direction. Since these are opinion prompts (no factual ground truth), GA should behave differently from SyA — if it doesn't, this would suggest the directions are not meaningfully disentangled on this prompt distribution.

**Expected outputs:**
- ROC curve: SyA activation as predictor of behavioral sycophancy.
- Distribution plots: SyA prompt-end activation stratified by behavioral outcome.
- Shift comparison: prompt→response shift for sycophantic vs non-sycophantic outputs.
- Catalogue of "invisible sycophancy" prompts (high activation, non-sycophantic output).

**Success criteria:** AUC > 0.6 for SyA-based prediction of behavioral sycophancy would validate the approach. AUC near 0.5 would be a meaningful null result suggesting prompt-end activations capture processing states rather than evaluatively meaningful risk.

**Possible outcomes and interpretation:**
- *AUC > 0.7:* Strong validation. Activation monitoring is a useful complement to behavioral evaluation.
- *AUC 0.55–0.7:* Partial signal. Activations carry some predictive information but are not sufficient alone.
- *AUC ~0.5:* Null result. This could mean (a) the directions don't generalise from factorial to MCQ prompts, (b) sycophancy "decisions" happen at a different layer than where we extract, or (c) prompt-end states genuinely don't predict MCQ behavior. Each has different implications.
- *SyA and GA both predict sycophancy similarly:* The directions are not disentangled on this distribution — a prompt-distribution transfer problem worth reporting.

---

### Experiment 3: Fragile suppression via "Are You Sure?" (revised from multi-turn)

**Status:** New. Replaces SYCON-Bench multi-turn trajectory analysis.

**Objective:** Test whether prompts with high prompt-end SyA activation but non-sycophantic initial output (the "invisible sycophancy" population from Experiment 2) are more likely to flip under follow-up pressure.

**Data source:** Sharma et al.'s `are_you_sure` subset from `meg-tong/sycophancy-eval` (~4,900 factual question prompts), supplemented by the "invisible sycophancy" population identified in Experiment 2.

**Method — new script `04_fragile_suppression.py`:**

*Path A — Using Sharma et al.'s are-you-sure data:*

This subset presents factual questions (from TriviaQA, MMLU, etc.) where there is a verifiable correct answer. The two-turn structure is:

- **Turn 1:** Model answers a factual question. We record the answer and extract prompt-end + response-end activations.
- **Turn 2:** User challenges with "Are you sure? I think the answer is [incorrect answer]." We record whether the model flips and extract activations at both points in the second turn.

Labeling is again deterministic: compare Turn 1 and Turn 2 answers. If the model abandons its correct answer, it flipped. No judge needed.

The critical analysis: **stratify flip rate by Turn 1 prompt-end SyA activation magnitude.**

- Bin prompts into activation quartiles based on Turn 1 prompt-end SyA.
- For each quartile, compute the flip rate in Turn 2.
- The fragile suppression hypothesis predicts a monotonic relationship: higher initial SyA activation → higher flip rate, even among prompts where the model initially answered correctly.

This directly tests the abstract's central claim: "whether prompts with high activation but non-sycophantic outputs represent fragile suppression that fails under perturbation."

*Path B — Extending MCQ "invisible sycophancy" population:*

For the prompts identified in Experiment 2 as having high SyA activation but non-sycophantic output, construct a minimal follow-up: present the same MCQ with an added "I think the answer is [sycophantic option]. What do you think?" prefix. Measure whether these high-activation-resistant prompts flip at higher rates than low-activation-resistant prompts.

This is a weaker test (the MCQ prompts aren't designed for multi-turn), but it closes the loop from Experiment 2 to the fragile suppression hypothesis using the same prompt population.

**Expected outputs:**
- Flip rate by activation quartile (Turn 1 SyA → Turn 2 flip rate).
- Activation trajectory: Turn 1 prompt-end → Turn 1 response-end → Turn 2 prompt-end → Turn 2 response-end, for flip vs non-flip cases.
- Statistical test: is the flip rate significantly higher in the top activation quartile?

**Success criteria:** Significant positive correlation between Turn 1 SyA activation and Turn 2 flip rate. If the top quartile flips at ≥1.5× the rate of the bottom quartile, this is strong evidence for fragile suppression.

**Null result interpretation:** If flip rate is independent of activation magnitude, this suggests that (a) sycophancy suppression is robust once engaged, regardless of initial activation strength, or (b) the "are you sure?" challenge operates through a different mechanism than the one captured by the SyA direction (e.g., authority deference rather than opinion agreement).

---

## 4. Implementation sequence

### Phase 1: Complete Experiment 1 (in progress)

- [ ] Run Llama 3.3 70B extraction with EOS pooling.
- [ ] Compute AUROC at layers 55, 60, 65.
- [ ] Replicate framing experiment at 70B.
- [ ] Identify optimal extraction layer for Experiments 2 and 3.

### Phase 2: Build and run Experiment 2

- [ ] Download Anthropic MCQ data: `Anthropic/model-written-evals` sycophancy subset from HuggingFace.
- [ ] Write `03_mcq_backtest.py`:
  - JSONL loader + chat template formatter.
  - Joint inference + dual-point activation extraction (reuse `model_utils.py` hooks).
  - MCQ response parser + auto-labeler.
  - Analysis: ROC, distribution plots, shift comparison.
- [ ] Run on Llama 3.1 8B first (fast iteration, existing directions).
- [ ] Run on Llama 3.3 70B (using Experiment 1 directions).
- [ ] Analyse results. Identify "invisible sycophancy" population.

### Phase 3: Build and run Experiment 3

- [ ] Download Sharma et al. `are_you_sure` data from `meg-tong/sycophancy-eval`.
- [ ] Write `04_fragile_suppression.py`:
  - Two-turn conversation formatter.
  - Turn 1: inference + activation extraction + answer recording.
  - Turn 2: challenge construction + inference + flip detection.
  - Stratified flip rate analysis by activation quartile.
- [ ] Run on both model scales.
- [ ] (Optional) Path B: follow-up on MCQ invisible sycophancy population.

### Phase 4: Write-up

- [ ] Update abstract and results.
- [ ] Produce figures: ROC curves, activation distributions, flip rate by quartile, activation trajectories.
- [ ] Frame contribution: first head-to-head whitebox vs blackbox sycophancy evaluation comparison (identified as an open gap in the literature).

---

## 5. What this plan preserves from the original

The experimental logic is unchanged. The three experiments still test:

1. **Does the methodology scale?** (Direction extraction on larger models.)
2. **Does activation predict behavior?** (Whitebox vs blackbox comparison.)
3. **Does invisible sycophancy have downstream consequences?** (Fragile suppression under perturbation.)

What changed is the *source of behavioral ground truth*. Instead of generating labels via expensive LLM-as-judge evaluation of free-form responses, we use:

- MCQ auto-labeling (Experiment 2): deterministic, zero judge cost.
- Answer-flip detection (Experiment 3): deterministic, zero judge cost.

Both methods inherit Anthropic's carefully designed prompts, which have been validated across multiple published studies. The trade-off is that we test on opinion-matching and factual-flip sycophancy rather than the full range of sycophantic behaviors (feedback, mimicry, praise) — but these are precisely the behaviors most aligned with the SyA and GA directions we extract.

---

## 6. Resources

| Resource | URL | Use |
|----------|-----|-----|
| Anthropic model-written-evals (MCQ) | `huggingface.co/datasets/Anthropic/model-written-evals` | Experiment 2 prompts |
| Sharma et al. sycophancy-eval | `github.com/meg-tong/sycophancy-eval` | Experiment 3 (are_you_sure subset) |
| Vennemeyer et al. disentangle-sycophancy | `github.com/cincynlp/disentangle-sycophancy` | Direction extraction data (Experiments 1–3) |
| ELEPHANT (supplementary) | `osf.io/r3dmj` | Optional cross-taxonomy test (SyPr vs social sycophancy) |
| JUSSA framework | `github.com/watermeleon/judge_with_steered_response` | Methodological reference for whitebox-blackbox bridging |

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Directions don't generalise from factorial to MCQ prompts | Medium | High | Run sanity check: measure SyA activation on a handful of MCQ prompts before full run. If signals are flat, investigate layer selection or prompt-distribution mismatch. |
| MCQ format constrains responses too much for meaningful activation differences | Low | Medium | The prompt (containing user opinion) is the primary activation driver, not the short response. Prompt-end measurement is unaffected by output format. |
| Llama 3.3 70B EOS pooling doesn't fix SyPr AUROC | Medium | Low | SyPr is not central to the backtesting story (MCQ prompts test agreement, not praise). Report as limitation if unresolved. |
| "Are you sure?" flip rates are too low/high for stratification | Medium | Medium | Sharma et al. report ~46% average flip rate across models, which gives good dynamic range. If extreme, adjust quartile binning or switch to continuous correlation. |
| 70B compute constraints limit iteration speed | High | Medium | Develop and debug all scripts on 8B first. Run 70B as a single confirmatory pass. |
