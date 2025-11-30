## Why RL? Why Now?

2025 is the year (or decade) AI moves from "talking" to "doing."
For the last two years, we've optimised for *plausibility* (does it sound right?). Now, we optimise for **verifiability** (did it work?).

Reinforcement Learning is the engine of this shift. It is crucial for problems where:

1. **Multiple solutions exist** (Creativity > Pattern Matching).
2. **No training data exists** (We can't clone human behaviour; we must discover new strategies).
3. **The environment is non-differentiable** (Black-box software, compilers, games, biology).

Every project in this hackathon should address some part of the loop:
`Agent` → `Action` → `Environment` → `Reward` → `Update`

Importantly, RL isn’t just about training. Rich environments with realistic and verifiable tasks are the new “gold” for data, and research and development in these areas is just as, or perhaps even more valuable. As such we’ve organised around 3 themes/tracks: Environments, Tasks, and Training.

---

## 🏁 The Tracks

### Track 1: Building Environments

The model can only be as smart as the world it lives in. This track is about wrapping real software, games, or business logic into Gyms (environments with a `step()` function).

**The main question**: How do we create new, novel, challenging RL environments out of datasets, existing software, or entirely from scratch?

**Ideas to Push:**

- **Simulation:** How can you translate an existing real world application, game, or other system into an environment suitable for training, and how much of that can you automate? E.g. post-train a model to convert HTML into React for building clones.
- **Multi-environment tasks:** Can you build interesting RL environments that aren’t just on one piece of software or app but across multiple?

### Track 2: Building Task Curricula

An environment is useless without tasks and their reward functions.

**The main question:** How can we find Interesting ways to automate the production of interesting, diverse tasks with progressive difficulty?

**Ideas to Push:**

- **Exploration:** Can the generation of tasks in some domain be automated through existing agentic exploration of software or APIs?
- **Distillation:** Can you distill knowledge andskill from Frontier Models (o1/DeepSeek) into a dataset of verifiable tasks for small models?
- **Curriculum Generation:** Can you develop a progressively harder set of tasks from a baseline set (e.g SQL queries or math proofs) to challenge the best models?

### Track 3: Training Agents

For the Machine Learning Engineers. Take an environment and make a number go up.

**The main question:** Can you successfully train agents through RL and how well can you do this with with respect to compute limits, sample efficiency, model size?

**Ideas to Push:**

- **Efficiency:** What is the smallest model (e.g., 3B) that can solve a task usually reserved for 70B models with RL?
- **Method Shootout:** Direct Weight Training (GRPO) vs. Prompt Optimization (GEPA). Which converges faster?
- **Cold Start:** How quickly can an agent learn a completely new game or challenge with zero priors?

**IMPORTANT NOTE**: Although these three tracks are the primary focus of the hackathon, participants are strongly encouraged to pursue any compelling or creative RL-related ideas they’d like to explore. If you have an exciting direction that doesn’t fit neatly into a track but pushes the boundaries of what’s possible in RL, we want to see it!

---

## 🧰 The Stack: Recommended Resources

**Training Frameworks**

- [**Hugging Face TRL](https://huggingface.co/docs/trl/en/index):** The industry standard for PPO/GRPO with Transformers.
- [**PrimeRL](https://github.com/PrimeIntellect-ai/prime-rl):** High-performance, scalable RL training infrastructure.

**Existing Environments**

- [**Prime Intellect Environments](https://app.primeintellect.ai/dashboard/environments):** Pre-packaged, verified environments (SWE-bench, AIME, MiniWoB).
- [**ADE-Bench](https://github.com/thedatamates/ade-bench/):** Gym for Data Science & SQL agents.
- [**Factorio Learning Environment (FLE)](https://jackhopkins.github.io/factorio-learning-environment/versions/0.3.0.html):** For complex logistics and planning.
- [**NetHack Learning Environment (NLE)](https://github.com/facebookresearch/nle):** The "Dark Souls" of RL. Procedurally generated, incredibly hard.

**Prompt Optimization (DSPy & GEPA)**

- [**GEPA](https://github.com/gepa-ai/gepa):** A new evolutionary algorithm (Genetic-Pareto) that optimizes prompts as if they were weights.
- [**DSPy + GEPA Tutorial](https://dspy.ai/tutorials/gepa_ai_program/):** How to run prompt optimization loops instead of weight fine-tuning.