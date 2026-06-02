# Main Paper Outline

Working title:

**Brain-Aligned Memorability Signals Improve Video Generation Selection**

## Abstract Draft

Generated videos are increasingly evaluated by automatic measures of visual
quality and text-video alignment, but these metrics do not directly target what
viewers will remember. We study whether brain-aligned video representations can
provide a useful memorability signal for selecting among generated candidates.
Using BOLD Moments memorability labels and TRIBE fMRI-predicted cortical
responses, we learn a compact supervised memorability direction and evaluate it
against visual and multimodal baselines. We then use the direction as a
test-time selector over multiple generated candidates per prompt, with
preservation gates to avoid selecting low-quality or off-prompt clips. The
current pilot instantiates this comparison against CLIP preservation and V-JEPA
memorability selectors on the exact same candidate pool; the planned blinded
human evaluation asks whether TRIBE-selected videos are judged more memorable
than candidates selected by those non-brain baselines. The result would position
brain-aligned representations as practical reward signals for cognitive
properties of generated media, while separating selection-time gains from
stronger claims about direct model steering.

## Contributions

1. A reproducible brain-aligned memorability direction derived from BMD/TRIBE.
2. A selector protocol for generated video candidates with quality/preservation
   gates.
3. A human evaluation testing whether brain-aligned selection improves perceived
   memorability over non-brain baselines.
4. A critical audit showing where the method works, where V-JEPA matches it, and
   why direct steering remains unsolved.

## Section Plan

1. Introduction
   - Problem: generated videos are easy to make but hard to rank for cognitive
     effect.
   - Hypothesis: brain-aligned models expose viewer-response axes not captured
     by generic quality metrics.
   - Narrow claim: memorability selection, not global engagement.

2. Related Work
   - Image/video memorability.
   - BOLD Moments and brain encoding.
   - Video generation evaluation.
   - Best-of-N and preference alignment.
   - Brief discussion-only frame: representation choice, active inference, and
     anti-reification of learned axes.

3. Learning A Brain-Aligned Memorability Direction
   - BMD data and exclusions.
   - TRIBE features and BMD labels.
   - Cross-validation protocol.
   - Baselines: V-JEPA, CLIP, caption/text, generic VLM.

4. Selector Design
   - Candidate generation.
   - Selector policies.
   - Preservation/quality gates.
   - Predeclared primary endpoint.

5. Human Evaluation
   - Study design.
   - Attention checks.
   - Pairwise and delayed-recognition variants.
   - Prompt-clustered statistics.

6. Results
   - BMD prediction.
   - Selector win rates against baselines.
   - Ablations: no gate, quality-only gate, TRIBE-only, V-JEPA-only.
   - Failure cases.

7. Analysis And Limitations
   - V-JEPA competitiveness.
   - Reward circularity avoided by human eval.
   - Persona axes as exploratory only.
   - Mechanistic probes as confound analysis.
   - Learned directions as frame-dependent readouts rather than ontological
     cognitive primitives.

8. Conclusion
   - Brain-aligned signals can be practical selectors for cognitive media
     properties, but model tuning remains future work.
   - Broader program: generated media as controlled intervention on human
     generative models, with memorability as the first validated axis.

## Figures

1. Method diagram: BMD/TRIBE -> direction -> candidate selector -> human eval.
2. Cross-validated BMD prediction and baseline comparison.
3. Selector policies and human win rates.
4. Failure-case grid.
5. Optional appendix: persona axis and mechanistic audit.
