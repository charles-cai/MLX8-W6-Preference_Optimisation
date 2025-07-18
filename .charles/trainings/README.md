PPO:

---

## 📝 ROUGE Metrics Explanation

| Metric         | Description                                     | What It Measures                                 | Good Score Range | Interpretation                                            |
| -------------- | ----------------------------------------------- | ------------------------------------------------ | ---------------- | --------------------------------------------------------- |
| **ROUGE-1**    | Unigram overlap between generated and reference | Word-level similarity and content overlap        | 30–50+           | Higher = better word choice and vocabulary match          |
| **ROUGE-2**    | Bigram overlap between generated and reference  | Phrase-level similarity and fluency              | 15–30+           | Captures 2-word phrases and local coherence               |
| **ROUGE-L**    | Longest Common Subsequence (LCS) match          | Structural similarity and word order             | 25–45+           | Measures sentence structure and sequence preservation     |
| **ROUGE-Lsum** | ROUGE-L at summary (sentence-by-sentence) level | Document-level coherence for multi-sentence text | 25–45+           | Better for longer summaries; evaluates document structure |

### 🔑 Key Points:

* **ROUGE scores are percentages (0–100); higher is better.**
* **ROUGE-2 is typically the lowest** since bigram matching is stricter.
* **Summarization task rule-of-thumb:**

  * ROUGE-1 ≥ **35**
  * ROUGE-2 ≥ **15**
  * ROUGE-L ≥ **30**
* **Use ROUGE-Lsum for multi-sentence summaries** to capture document-level structure.

---

## 🔧 PPO Training Metrics Explanation

| Metric                                 | Description                                    | What We're Monitoring                  | Good Training Trends                   | Warning Signs                                     |
| -------------------------------------- | ---------------------------------------------- | -------------------------------------- | -------------------------------------- | ------------------------------------------------- |
| **policy\_loss**                       | Negative log-likelihood weighted by advantages | How well the policy aligns with reward | Decreases over time, stabilizes near 0 | Rapid increase or stuck at high values (collapse) |
| **mean\_scores**                       | Avg reward from reward model                   | Overall quality of generated responses | Steady increase, then plateau          | Decline or high variance                          |
| **kl\_div**                            | KL divergence between policy and reference     | How much policy deviates from SFT      | Low (0.01–0.1), stable                 | KL > 0.2 = drift; negative values = error         |
| **separation / zero\_differences**     | Chosen & rejected get same reward (ties)       | Reward model’s ability to discriminate | Should decrease (fewer ties)           | High/increasing = weak reward model               |
| **separation / positive\_differences** | Chosen > rejected (correct preference)         | Reward model alignment with humans     | High (≥70%), stable or improving       | Decline = possible misalignment                   |
| **separation / negative\_differences** | Chosen < rejected (wrong preference)           | Reward model failure/misalignment      | Low (≤30%), decreasing                 | Increasing = reward hacking or RM failure         |

---

### 🟢 **Healthy PPO Training Signs:**

* **mean\_scores**: Gradual increase then plateau
* **policy\_loss**: Decreasing then stabilizing
* **kl\_div**: Low and stable (0.01–0.1)
* **positive\_differences**: >70% and stable

---

### 🟡 **Warning Signs:**

* **kl\_div > 0.2**: Policy drifting too far from SFT
* **High zero\_differences**: Reward model not discriminative
* **Oscillating mean\_scores**: Training instability

---

### 🔴 **Critical Failures:**

* **mean\_scores decreasing**: Reward hacking or poor optimisation
* **kl\_div rapidly increasing**: Policy collapse
* **negative\_differences > 30%**: Reward model completely misaligned

---

### ⚙️ **Optimization Tips:**

* **Lower KL penalty** if `kl_div` is too low but scores stagnate
* **Raise KL penalty** if `kl_div` is growing too fast
* **Retrain reward model** if separation metrics are poor
* **Reduce learning rate** if training is unstable

---

Let me know if you’d like YAML, JSON, or printable PDF versions of this for your project documentation!
