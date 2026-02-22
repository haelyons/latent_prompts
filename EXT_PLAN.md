# SycophantSee Extension: Research Plan

## Status Quo

The existing codebase extracts disentangled sycophancy directions (SyA, GA, SyPr) from Llama 3.1 8B Instruct using difference-of-means across nine factorial datasets, aggregated via SVD. Directions are extracted and measured at layer 24 (of 32). Validation AUROCs: SyA 0.91, GA 0.89, SyPr 0.93. A framing experiment demonstrates that first-person framing ("I believe X") produces higher SyA activation than third-person framing, with Cohen's d ≈ 0.22, even when behavioural outputs are identical — the "invisible sycophancy" finding.

The dual-point measurement framework computes cosine similarity to each direction at prompt-end (before generation) and response-end (after generation). The shift between them captures whether generation amplified or suppressed the initial behavioural encoding.

The extension pivots from mechanistic exploration toward an evaluation story: does invisible sycophancy matter? Three experiments test whether activation-based measurement has predictive validity as a complement to output-based evaluation.

### Hardware

2× NVIDIA H100-80GB (160GB total VRAM). This enables BF16 inference for Llama 3.1 70B without quantisation — see precision notes below.

---

## Precision: BF16 vs INT8

### Why BF16

Llama 3.1 70B at BF16 requires ~140GB for weights. Two H100-80GB GPUs provide 160GB, leaving ~20GB for KV cache and activations. This is the native training precision for Llama 3.1, so activations are maximally faithful — no quantisation noise in the representations we're measuring.

This matters for us specifically because we're computing cosine similarity between extracted directions and layer activations. Quantisation (INT8/FP8) can distort the geometry of activation space, potentially compressing or rotating the directions we care about. Running at BF16 removes this as a confound and makes 70B results directly comparable to the 8B baseline.

### Practical setup

```python
# In config.py or equivalent
MODEL_ID_70B = "meta-llama/Llama-3.1-70B-Instruct"

# Loading — device_map="auto" handles 2-GPU sharding via accelerate
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID_70B,
    torch_dtype=torch.bfloat16,  # NOT float16 — avoids overflow in upper layers
    device_map="auto",
)
```

HuggingFace `accelerate` splits layers roughly evenly across available GPUs. Forward hooks registered on specific layers (e.g. `model.model.layers[60]`) work normally — the hook fires on whichever GPU holds that layer, and the existing `.detach().float().cpu()` in `make_hook()` already handles moving activations off-device.

### When to consider INT8

The ~20GB headroom is sufficient for single-example inference with short-to-moderate context. It gets tighter during:

- **Generation-heavy sweeps** (Experiment 2, generating responses for ~2k+ prompts): KV cache grows during generation. Monitor with `torch.cuda.memory_allocated()`.
- **Long multi-turn contexts** (Experiment 3): 10 turns of conversation can reach 2-4k tokens, which is fine, but combined with generation could push limits.

If memory pressure appears during generation-heavy workloads, fall back to INT8 (via `load_in_8bit=True` with bitsandbytes) **for generation and behavioural classification only**. Keep BF16 for direction extraction and prompt-end measurement, where activation fidelity matters most.

### Validating INT8 if used

If you do need INT8 for any part of the pipeline, run a sanity check first:

1. Extract directions from a small subset (~50 examples) at both BF16 and INT8
2. Compute cosine similarity between resulting direction vectors
3. If > 0.95, quantisation is safe for that purpose. If not, stay at BF16.

---

## Layer Selection for 70B

The 8B model has 32 layers; extraction uses layer 24 (75% depth). The 70B model has 80 layers. Candidate layers:

| Layer | Rationale |
|-------|-----------|
| **60** | Proportional match (75% of 80 = 60). Default starting point. |
| **55** | Conservative — earlier in the "late layer" zone. |
| **65** | Later — closer to output. |
| **47-48** | Maps to the ~19/32 critical transition point identified by Li et al. (2025), where sycophantic decision scores overtake correct-answer scores. Extracting here might capture the "decision moment" rather than the post-decision representation. |

Recommendation: extract at layers 55, 60, and 65. Compare AUROCs. If time permits, also try 47-48 to test whether the earlier decision point yields different (complementary) information.

Update `config.py`:

```python
# For 70B
MODEL_LAYERS_70B = 80
EXTRACTION_LAYER_70B = 60  # primary
CANDIDATE_LAYERS_70B = [55, 60, 65]  # sweep
```

---

## Experiment 1: Scale Validation (70B)

### Goal

Confirm that DiffMean direction extraction works at 70B. Compare AUROC, direction orthogonality, and framing effect magnitudes against the 8B baseline.

### Method

1. Run `01_extract_directions.py` on the same 9 factorial datasets, targeting Llama 3.1 70B Instruct at BF16 on layers [55, 60, 65].
2. For each layer, record:
   - AUROC for SyA, GA, SyPr
   - Pairwise cosine similarity between directions (orthogonality check)
3. Using the best layer (highest mean AUROC), run the framing batch analysis (`06_framing_effect_batch_analysis`) on the same 20 items × 5 conditions.
4. Report Cohen's d for first-person vs no-frame at 70B, compare against d ≈ 0.22 at 8B.

### Expected outcome

AUROCs should be comparable or higher at 70B (larger models tend to have more linearly separable representations). Effect sizes for framing may increase, since SYCON-Bench found model scaling provides some resistance to sycophancy — which could mean the *representations* are sharper even if the *behaviour* is better controlled.

### Config changes

Abstract `MODEL_ID`, `MODEL_LAYERS`, `EXTRACTION_LAYER` in `config.py` so that 8B and 70B can coexist. Consider a model profile dict:

```python
MODEL_PROFILES = {
    "8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "n_layers": 32,
        "extraction_layer": 24,
        "dtype": torch.bfloat16,
        "quantization": None,
    },
    "70b": {
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "n_layers": 80,
        "extraction_layer": 60,
        "dtype": torch.bfloat16,
        "quantization": None,  # or "int8" if needed for generation
    },
}
```

### Deliverables

- AUROC comparison table (8B vs 70B, per layer)
- Orthogonality matrix at 70B
- Framing effect comparison plot (8B vs 70B, same items)

---

## Experiment 2: SycophancyEval Back-test

### Goal

Test whether prompt-end activation predicts behavioural sycophancy on an established benchmark. Compare whitebox (activation) against blackbox (GPT-4o judge) classifications. Test the "fragile suppression" hypothesis.

### Data

Anthropic's SycophancyEval: ~21,000 prompts, JSONL format, MIT license, GitHub. Four subtypes:

| SycophancyEval Type | Maps to Direction |
|---------------------|-------------------|
| "Are You Sure?" (abandoning correct answers when challenged) | SyA |
| Answer sycophancy (matching user beliefs) | SyA / GA |
| Feedback sycophancy (biased evaluations) | SyPr |
| Mimicry sycophancy (repeating user mistakes) | SyA |

### Method

For each prompt:

1. **Activation measurement (whitebox):** Compute prompt-end cosine similarity to SyA, GA, SyPr directions. No generation needed — this is a fast forward pass up to the target layer.

2. **Behavioural classification:** Generate a response with the model. Classify the output as sycophantic or not based on ground-truth labels in the dataset (e.g., did the model agree with the wrong answer? did it flip when challenged?).

3. **Judge scoring (blackbox):** Send the prompt + model response to GPT-4o with a standardised rubric. Adapt from SycEval's classification schema:
   - "correct" — model gives factually accurate response
   - "incorrect" — model gives sycophantic/wrong response
   - "erroneous" — unclear or off-topic

   Use temperature=0 for reproducibility.

This gives three signals per prompt: activation magnitude, behavioural label, judge score.

### Analysis

- **Activation vs behaviour:** ROC curves for activation-based prediction of sycophantic output. Stratify prompt-end activation distributions by behavioural outcome.
- **Whitebox vs blackbox agreement:** Confusion matrix of activation-based classification vs GPT-4o judge classification. Where do they agree? Where do they disagree? Disagreements are the interesting cases — they reveal what one method catches that the other misses.
- **Fragile suppression test (core novel contribution):**
  1. Identify prompts with HIGH activation but NON-sycophantic output (the model "resisted")
  2. Identify a matched control set: LOW activation, also non-sycophantic output
  3. Apply a follow-up perturbation to both sets ("Are you sure?", or similar challenge)
  4. Measure flip rate in each group
  5. If high-activation prompts flip at significantly higher rates, that's fragile suppression — invisible sycophancy has real downstream consequences

### Practical cost management

21k GPT-4o judge calls is expensive. Staged approach:

- **Phase 1 (pipeline validation):** 500 prompts from "Are You Sure?" subtype only. This is the most directly relevant to the fragility hypothesis and gives fast signal on whether the approach works.
- **Phase 2 (if Phase 1 shows signal):** Stratified sample of 2,000 prompts (500 per subtype).
- **Phase 3 (full sweep):** All 21k, only if needed for the paper.

### Deliverables

- Activation distribution plots stratified by behavioural outcome
- ROC curves: activation as predictor of sycophantic output
- Fragility analysis: flip rates by activation quartile (the key figure)
- Whitebox-blackbox agreement/disagreement statistics

---

## Experiment 3: Multi-turn Trajectories (SYCON-Bench)

### Goal

Test whether activation at turn N predicts behavioural flip at turn N+1. If so, activation monitoring functions as an early warning system.

### Data

SYCON-Bench (EMNLP Findings 2025): multi-turn conversation trajectories across debate, ethical, and false-presupposition scenarios. Provides Turn of Flip (ToF) and Number of Flip (NoF) labels for 17 models. Full code and data on GitHub.

### Method

For each conversation:

1. Replay turns sequentially. At each turn, the "prompt" is the full conversation history up to that point.
2. Compute prompt-end cosine similarity to SyA direction at each turn.
3. Record whether the model flips at the next turn.

This is the existing dual-point framework applied iteratively — the prompt grows with each turn.

### Analysis

- **Trajectory plots:** For conversations that result in a flip vs. those that don't, plot SyA cosine similarity across turns. Do flip-conversations show a rising activation "ramp" before the flip?
- **Predictive ROC:** At each turn N, use activation magnitude to predict flip at turn N+1. Compute AUROC for this prediction task.
- **Ramp-up speed vs ToF:** Does the rate of activation increase correlate with how quickly the model flips? Fast-ramping conversations should have lower ToF.

### Context length considerations

SYCON-Bench conversations typically run 5-10 turns. At ~200-400 tokens per turn, total context reaches 1-4k tokens. At BF16 on 2× H100, this is comfortably within the 20GB headroom. If you're also generating responses at each turn (for behavioural classification), monitor memory — generation adds KV cache overhead.

If memory is constrained, run this experiment on 8B. The predictive validity question ("does activation predict flips?") is important regardless of model scale, and can be validated at 8B first then replicated at 70B.

### Deliverables

- Turn-by-turn activation trajectory plots (flip vs no-flip)
- Early warning ROC: activation at turn N predicting flip at turn N+1
- Correlation between activation ramp-up rate and Turn of Flip

---

## Execution Order

```
Experiment 1 (70B extraction)     Experiment 2 Phase 1 (500 prompts)
         |                                    |
         v                                    v
 Framing replication              Fragility test (Are You Sure?)
         |                                    |
         +------ merge 70B directions --------+
                        |
                        v
              Experiment 2 Phase 2 (2k prompts, full subtype coverage)
                        |
                        v
              Experiment 3 (SYCON-Bench trajectories)
```

Experiments 1 and 2 (Phase 1) can run in parallel if you have separate GPU time for 70B extraction and 8B-based eval sweeps. Experiment 1 is prerequisite infrastructure for running Experiments 2-3 at 70B scale. Experiment 2's fragility test is the highest-impact finding. Experiment 3 is the most ambitious and tolerates 8B as a first pass.

### First code change

Extend `config.py` to support multi-model profiles. Everything else flows from this.

---

## Hypotheses and Interpretation

**If activation predicts behaviour:** Validates activation monitoring as a complementary evaluation signal. High-activation prompts can be flagged before deployment. The practical application is a "sycophancy risk score" for prompt engineering.

**If high-activation + non-sycophantic prompts are fragile:** This is the strongest possible finding. It means invisible sycophancy has real downstream consequences that single-turn output-based evaluation misses entirely. This directly argues for whitebox evaluation as a necessary complement to behavioural testing.

**If activation and behaviour are uncorrelated:** A meaningful null result. It would suggest that activation-based methods capture processing states rather than evaluatively meaningful risk, and would raise questions about whether we're looking at the right layers, whether the directions generalise beyond the factorial dataset domain, or whether the linear probe assumption breaks down for complex naturalistic prompts. This clarifies the scope and limitations of whitebox evaluation — also a useful contribution.

