# Project Summary: Learning to Drive Social Cohesion Through Content Generation

## One-Sentence Summary

This project asks whether language models, multimodal generative systems, and brain-aligned models can learn interpretable “social cohesion vectors”: latent directions in text, media, and neural response space that shift individuals and groups away from self-protective or adversarial behavior and toward cooperative, prosocial, truth-seeking, and conflict-resolving action.

## Motivation

Modern AI systems are increasingly capable of generating persuasive language, images, audio, and video. Most work on persuasion, recommendation, and content optimization is framed around engagement, conversion, attention, or preference satisfaction. This project proposes a different objective: can generative AI be used to cultivate social cohesion?

The philosophical frame is old. In Aristotle’s *Nicomachean Ethics*, the highest good is not merely private satisfaction but human flourishing, and the good of the political community is treated as more complete than the good of one person alone. Robert Axelrod’s *The Evolution of Cooperation* gives a computational and game-theoretic frame: under repeated interaction, strategies such as reciprocity can make cooperation stable even among self-interested agents. A contemporary AI ethics frame appears in recent Vatican work on AI, including *Antiqua et Nova*, which emphasizes that AI should be ordered toward human dignity, truth, social responsibility, and the common good.

The research question is whether we can formalize part of this moral and social problem computationally. Can we learn the representational directions that correspond to selfishness, distrust, factional escalation, reconciliation, mutual recognition, and social volition? And once learned, can these directions guide content generation toward healthier group outcomes?

## Core Hypothesis

The central hypothesis is that “social cohesion” is not a single scalar property, but it may still have learnable latent structure across three spaces:

1. **LLM activation space**: features or directions that activate when agents reason in cooperative, selfish, conciliatory, status-seeking, revenge-seeking, or bridge-building modes.
2. **Content space**: linguistic, visual, and narrative features that shift audiences toward trust, shared identity, fairness, patience, forgiveness, or constructive action.
3. **Human neural response space**: fMRI, EEG, fNIRS, or other brain-response patterns associated with willingness to cooperate, reconcile, help, listen, compromise, or solve group problems.

The long-term scientific question is whether these spaces align. If an LLM-derived “cohesion vector” can be learned from simulations and interpretability tools, does it correspond to measurable patterns in human neural response and behavior?

## Working Definition: Social Cohesion Vector

For this project, a “social cohesion vector” should not be treated as a mystical or moral essence. It should be operationalized as a measurable direction, feature set, or latent subspace that predicts movement toward behaviors such as:

- increased willingness to cooperate in repeated games;
- reduced desire to punish or defect when mutual benefit is possible;
- higher trust calibration;
- greater perspective-taking and mentalization;
- lower dehumanization or outgroup hostility;
- better group problem-solving outcomes;
- increased preference for fair or Pareto-improving solutions;
- greater willingness to enter dialogue after conflict.

Similarly, a “selfishness vector” should be treated carefully: not as normal self-interest, but as adversarial or socially corrosive behavior such as zero-sum reasoning, domination, opportunistic defection, dehumanization, or refusal to update in light of another person’s legitimate interest.

## Research Program

### Phase 1: Low-Cost LLM Simulation

The first phase can be done entirely with language models.

Create multi-agent simulations in which agents play roles in repeated social dilemmas: prisoner’s dilemma, public goods games, negotiation games, hostage negotiation, labor disputes, family conflict, political deliberation, or group planning. Give agents different incentives, backgrounds, emotional states, and communication constraints.

Then evaluate what kinds of messages or interventions move the system toward cooperation. Outcomes can include agreement rate, trust recovery, willingness to continue dialogue, joint payoff, fairness, reduction in hostile language, and stability of cooperation over repeated rounds.

This phase asks:

- Which conversational strategies reliably increase cooperation among agents?
- Which prompts increase selfish or adversarial behavior?
- Can we learn a reward model for “cohesion-promoting” communication?
- Do different LLMs share similar internal features when producing cohesive vs divisive responses?

### Phase 2: Mechanistic Interpretability in LLMs

The next phase studies the models themselves.

Use sparse autoencoders (SAEs), activation patching, steering vectors, probing, and representation analysis to find features that activate during cooperative, selfish, conciliatory, or conflict-escalating reasoning. Recent mechanistic interpretability work shows that SAEs can identify more interpretable features in language model activations than raw neurons or arbitrary directions.

Possible experiments:

- Generate paired prompts: selfish vs cooperative, punitive vs restorative, polarizing vs bridging.
- Record internal activations from open LLMs.
- Train SAEs on activations from social-reasoning contexts.
- Identify features associated with reciprocity, fairness, revenge, empathy, shared identity, duty, forgiveness, dominance, and dehumanization.
- Intervene on these features to test whether they causally shift model outputs.
- Compare learned features across models and tasks.

This phase produces candidate “LLM social cohesion vectors.”

### Phase 3: Content Generation and Ranking

Once candidate vectors exist, use them to guide content generation.

The goal is not simply to produce “nice” messages. The goal is to generate content that measurably changes social orientation while preserving truthfulness, autonomy, and context-sensitivity.

The system could generate many candidate messages, images, videos, or dialogue strategies, then rank them using a cohesion reward model. This mirrors a practical best-of-N workflow:

1. Generate several candidate interventions.
2. Score each candidate for predicted social cohesion.
3. Score each candidate for truthfulness, manipulation risk, semantic preservation, and context fit.
4. Select the candidate that maximizes cohesion without violating constraints.

Applications could include conflict mediation, hostage negotiation support, public-health messaging, deliberative democracy, group decision-making, education, restorative justice, and online community moderation.

### Phase 4: Brain-Aligned Modeling With Existing Models

Before collecting new fMRI data, the project can use existing brain-aligned models such as TRIBE-like video or multimodal brain-encoding systems. These models can provide an initial bridge between content features and predicted human brain response.

The question becomes:

- Do LLM-derived cohesion strategies produce distinct predicted brain-response patterns?
- Are these patterns closer to known cooperation, reward, mentalizing, or agency-related systems?
- Can we compare “selfishness-driving” and “cohesion-driving” content in a brain-aligned embedding space?

This is a low-to-medium-cost bridge between pure LLM work and human neuroscience.

### Phase 5: Human Behavioral Experiments

Before fMRI, run behavioral studies on Prolific or in person.

Participants can be shown short messages, conversations, images, or videos, then asked to make decisions in social tasks:

- cooperate or defect in repeated games;
- allocate resources in dictator/ultimatum/public-goods games;
- choose whether to punish, forgive, or continue dialogue;
- rate trust, empathy, threat, fairness, and willingness to collaborate;
- solve group problems after exposure to different content.

This phase provides human labels for tuning the cohesion reward model.

### Phase 6: Neural Experiments With fMRI, EEG, or fNIRS

The most ambitious phase is collecting new neural data.

Participants could be exposed to content or live group interactions designed to vary in predicted cohesion. During or after exposure, measure fMRI, EEG, fNIRS, physiology, and behavioral choices.

Candidate designs:

- **Individual fMRI**: participants watch or read content, then make cooperation decisions.
- **EEG/fNIRS hyperscanning**: multiple participants solve problems or negotiate while their neural synchrony is measured.
- **Group interaction tasks**: teams attempt to resolve conflicts after receiving different AI-generated interventions.
- **Longitudinal study**: repeated exposure to content and measurement of trust/cooperation over time.

Prior fMRI work on cooperation has found mutual cooperation in an iterated Prisoner’s Dilemma associated with reward-related regions such as nucleus accumbens, caudate, orbitofrontal/ventromedial frontal cortex, and rostral anterior cingulate. Hyperscanning research studies inter-brain synchrony during real social interaction and may be especially relevant for measuring group-level cohesion.

## Master’s Thesis Version

For a master’s thesis, the project should be scoped tightly:

**Thesis question:** Can LLM activation features learned from multi-agent social-dilemma simulations predict or steer cooperative vs selfish behavior, and do those features correspond to behavioral or brain-aligned proxies of social cohesion?

Recommended thesis scope:

1. Build a suite of multi-agent social-dilemma simulations.
2. Generate paired cooperative/selfish/conflict-resolving transcripts.
3. Train probes or SAEs on open-model activations.
4. Identify interpretable features associated with cohesion and selfishness.
5. Test causal steering in the LLM by amplifying or suppressing these features.
6. Validate outputs with a small human behavioral study or existing brain-aligned model.

This is feasible without collecting new fMRI data. The fMRI/EEG component can be framed as the future extension.

## Safety and Ethics

This project sits near persuasion and behavioral influence, so the ethical boundary must be explicit from the beginning.

The goal should be to support agency-preserving, truth-aligned, non-coercive social cohesion. It should not become a system for emotional manipulation, ideological conformity, propaganda, or suppression of legitimate dissent.

Ethical constraints:

- preserve autonomy and informed consent;
- avoid deception unless formally approved in a research protocol;
- optimize for truthfulness and mutual understanding, not compliance;
- distinguish social cohesion from mere agreement;
- preserve room for moral dissent and principled conflict;
- test for manipulation, dependency, and unequal effects across groups;
- use IRB review for human and neural experiments;
- treat “selfishness” carefully, avoiding pathologizing ordinary self-protection or minority interests.

The positive aspiration is constructive: to build tools that help people repair trust, deliberate better, and act toward a common good. The cautionary frame is equally important: systems that coordinate human motivation can be used destructively if they serve domination rather than dignity.

## Near-Term Deliverables

1. **Simulation benchmark**: a library of social dilemmas and conflict-resolution scenarios.
2. **Cohesion scoring rubric**: behavioral metrics for cooperation, trust, fairness, and dialogue quality.
3. **LLM activation dataset**: paired examples of cohesive vs selfish/adversarial reasoning.
4. **SAE/probe analysis**: interpretable features associated with social cohesion.
5. **Steering experiments**: interventions that increase or reduce cooperative outputs.
6. **Human pilot**: Prolific study validating whether model-selected interventions improve human cooperation judgments.
7. **Brain-aligned bridge**: comparison between LLM-derived strategies and TRIBE-like predicted neural-response embeddings.

## Research Questions

- Can social cohesion be represented as a stable direction or sparse feature set in LLM activation space?
- Do LLM-derived cohesion features generalize across games, domains, and models?
- Can these features be causally steered without degrading truthfulness or autonomy?
- Do humans judge cohesion-steered content as more cooperative, trustworthy, and conflict-resolving?
- Do brain-aligned models predict distinct neural-response patterns for cohesion-promoting vs selfishness-promoting content?
- In future fMRI/EEG/fNIRS studies, do these model-derived vectors correspond to measurable human neural dynamics during cooperation?

## Practical First Step

Start with the cheapest version:

1. Build 20-30 simulated social-dilemma scenarios.
2. Generate many agent conversations under different intervention strategies.
3. Score outcomes computationally.
4. Train a simple classifier/probe for cooperative vs selfish trajectories.
5. Use an open LLM and SAE/probing tools to identify internal features.
6. Run a small Prolific validation on the best and worst interventions.

If this works, move to multimodal content and brain-aligned modeling. If it fails, the project still produces a valuable negative result about the limits of simple “cohesion vectors.”

## Key References and Starting Points

- Robert Axelrod, *The Evolution of Cooperation* (1984), and the iterated Prisoner’s Dilemma tradition.
- Rilling et al., “A neural basis for social cooperation,” an fMRI study linking mutual cooperation to reward-related brain systems.
- Hyperscanning literature on EEG/fMRI/fNIRS measures of inter-brain synchrony during social interaction.
- Anthropic’s “Towards Monosemanticity” and related sparse autoencoder work for interpretable model features.
- Vatican AI ethics work such as *Antiqua et Nova*, especially its emphasis on human dignity, truth, and the common good.

## Framing for a Friend or Advisor

This project is an attempt to connect three fields that rarely meet cleanly: moral philosophy/game theory, mechanistic interpretability, and social neuroscience. The first milestone is not to “solve” social cohesion, but to learn whether cooperative social orientation has measurable, steerable structure in LLMs. If it does, the next question is whether that structure predicts anything about human behavior or neural response. If the answer is yes, the project could become a foundation for AI systems that generate content and dialogue strategies optimized not for attention, outrage, or conversion, but for trust, reconciliation, and the common good.
