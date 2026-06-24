# Best methods for link prediction found by **AntEvolve (AE)** code-evolution 

The indices $\text{A}$ and $\text{B}$ denote different training data configurations: **Version A** utilizes a training set of 6 synthetic and 4 real networks, whereas **Version B** relies on a reduced set of 6 synthetic networks and only 1 real network.

---

### 1. $\text{AE-Gemini-A}$
* **Core Architecture:** Built using a single randomized trees classifier applied over $24$ selected node- and link-based features.
* **Algorithmic Complexity:** Features are highly optimized, running at $\mathcal{O}(1)$ or $\mathcal{O}(k_u k_v)$ time complexity (where $k_i$ represents the degree of node $i$). This makes it extremely efficient for sparse, large-scale networks.
* **Performance:** Reaches the highest overall mean Area-Under-the-Curve ($\text{AUC}$) score of $0.913$ on the $550$ real networks test ensemble, completely independent of node label configuration.

### 2. $\text{AE-Gemini-B}$
* **Core Architecture:** Discovered through the Google Gemini framework but restricted to the sparser training configuration **B**.
* **Characteristics:** Developed to test the robust generalization of the code-evolution framework. Despite having access to only a single empirical network during training, it successfully establishes a predictive policy that outperforms traditional human-designed methods.

### 3. $\text{AE-Qwen-A}$
* **Core Architecture:** Utilizes an open-source alternative foundation model ($\text{Qwen3}$). It implements an ensemble stacking framework (via linear regression) that merges 5 distinct classifiers, combining low-variance models ($\text{logistic regression}$) with low-bias models ($\text{random forest}$ and $\text{gradient boosting}$) over $40$ features.
* **Algorithmic Complexity:** Retains some heavy topological metrics like betweenness centrality, causing it to effectively scale at $\mathcal{O}(|N|^2)$ in sparse environments.
* **Key Finding:** Exploits a unique node identification ($\text{ID}$) difference feature. Because of this, its performance drops a bit when node $\text{IDs}$ are randomly shuffled. It yields a strong baseline test score of $\text{AUC} = 0.877$ on the main test ensemble.
* **IMPORTANT:** This method uses node-id based features, so its performance drops slightly when node IDs are randomly shuffled.

### 4. $\text{AE-Qwen-B}$
* **Core Architecture:** Built via the $\text{Qwen3}$ pipeline using training configuration **B**.
* **Characteristics:** Serves as a direct benchmark to explore the boundaries of open-source algorithmic discovery under limited empirical exposure. Along with the other evolved frameworks, it successfully validates that machine-designed combinatorial strategies are highly adaptable.
* **IMPORTANT:** This method uses node-id based features, so its performance drops when node IDs are randomly shuffled.

### 5. $\text{AE-Gemini-2k}$
* **Core Architecture:** An extended-scale variation of the baseline Gemini model.
* **Characteristics:** Rather than stopping at the standard limit of $M = 1,002$ accepted programs, the evolutionary timeline was pushed to generate $M = 2,000$ accepted programs. This extended selection pressure provided a continuous exploration of the programming space, yielding a slight further optimization in the training $\text{AUC}$ frontier. 
* **IMPORTANT:** This method uses node-id based features, so its performance drops when node IDs are randomly shuffled.

---

### Performance Summary Table

The global predictive accuracy ($\text{AUC}$) across the primary test suites highlights how these evolved frameworks consistently outperform prominent human-designed models:

| Method | $8\text{ Synthetic Networks}$ | $30\text{ Large Networks}$ | $550\text{ Networks}$ |
| :--- | :---: | :---: | :---: |
| *Human-Designed Stacked Model* | $0.634$ | $0.869$ | $0.778$ |
| **$\text{AE-Qwen-A}$** | $0.803$ | $0.940$ | $0.877$ |
| **$\text{AE-Gemini-A}$** | $0.755$ | $0.951$ | $0.913$ |