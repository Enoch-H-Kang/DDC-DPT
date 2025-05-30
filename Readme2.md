## Reviewer #396A

- **Counterfactuals & Transitions:**  
  Great point. One possibility is to use offline reinforcement learning (e.g., Fitted Q-iteration). In a counterfactual scenario, we can use our existing data and estimated rewards ($\hat{r}$) to sample from $(s_0, a_0, \hat{r}(s_0, a_0), s_1, a_1, \hat{r}(s_1, a_1), s_2, \ldots)$.  
  Reinforcement learning can then be applied to this augmented data. We will add a discussion on this in the paper.

- **Statistical Inference:**  
  Great point. We will address this in the next draft, as our theoretical results can be refined into a finite-sample bound, rather than solely providing an asymptotic rate.

- **Computation Time & Properties:**  
  In addition to providing the finite-sample bound for sample complexity analysis, we will add average computation times for all methods to the experimental section in the revision.

- **Connections to Econ Literature:**  
  Thank you for these excellent references. We will incorporate them in the revised manuscript.

- **Misspecified Transitions/Beliefs, Heterogeneity, Games:**  
  We agree that these are highly relevant and interesting extensions. We will mention them.

---

## Reviewer #396B

- **Performance vs. Oracles:**  
  This is a critical point, and we will add more explanation to our discussion regarding this (see text below Table 1 on page 16). In dynamic problems where the data collection policy visits a few states much more frequently than others, “the use of a projection operator onto a space of function approximators with respect to a distribution induced by the behavior policy can result in poor performance if that distribution does not sufficiently cover the state space.” (Tsitsiklis & Van Roy (1997)) 

- **Anchor Action Assumption (Assumption 3.3):**  
  This assumption aids identification, akin to normalization or specifying an outside option's utility in discrete choice models. It ensures a unique $Q^*$ and $r$ can be recovered (Theorem 3.1). We will clarify its role and provide examples (e.g., a known baseline action like `wait` with a normalized reward of 0).

- **Lambda ($\lambda$) Selection:**  
  We will include the discussion on the theoretical justification of the arbitrary lambda choice from the Tikhonov regularization-based bi-level optimization viewpoint ("An explicit descent method for bilevel convex optimization," Solodov 2007). Empirically, we found performance to be relatively robust across a range of values (e.g., 0.1 to 10).

- **Distribution Shift:**  
  To clarify, we meant that the learned $r(s, a)$ is not inherently tied to a specific transition function, allowing it to be potentially used with different transition functions. We will refine this discussion.

---

## Reviewer #396C

- **Convergence & Approximation:**  
  The sample complexity analysis (e.g., complexity of temporal difference (TD) term) is not new. The contribution of this paper is mostly about proving global convergence, which is not achievable with TD-based gradient-based methods due to the double sampling issue.

- **Linearity assumption required?**  
  We apologize for the confusion. The linearity assumption is discussed in Lemma 6.1 *only* as one example function class satisfying the Jacobian condition (Assumption 6.2) needed for the *theoretical proof* of the PL condition. Lemma 6.2 shows NNs also satisfy it. Our *algorithm (GLADIUS)* and implementation *do not require* linearity and work with flexible approximators like the NNs used in our experiments (Section 7.1).

- **Estimating Transition Kernel:**  
  You correctly point out that, when the environment is not excessively high-dimensional, it is feasible to estimate the transition kernel and apply it for later estimation steps. However, in such approaches, errors from the transition kernel estimation can propagate and compound in the second stage. Circumventing this error propagation is a key motivation for our proposed method.

- **Bellman Error without Rewards:**  
  Crucially, the Bellman error term in our loss (Eq. 10/11) relies on the TD error $\mathcal{L}_{TD}$, which involves $\hat{\mathcal{T}}Q(s,a,s') = r(s,a) + \beta V_Q(s')$. However, this term is only active when $\mathbf{1}_{a=a_s}=1$ (i.e., for the anchor action $a_s$). By Assumption 3.3, the reward $r(s,a_s)$ for the anchor action *is known*. Therefore, the loss computation does *not* require access to unknown rewards $r(s,a)$ for $a \neq a_s$. We will clarify this mechanism.

- **Ill-posedness of LBE:**  
  This is a great point. While IRL typically resolves ill-posedness by considering reward transformations (Ng, Harada, and Russell 1999), DDC resolves it by making the anchor action assumption (Assumption 3.3), which allows us to compute LBE only for $(s,a)$ pairs that correspond to the anchor action.

- **Minimizing BE Only:**  
  Minimizing only $\mathcal{L}_{BE}$ might be considered if one were to estimate $r$ directly instead of $Q$. If transition probabilities (or estimates) were available, one could compute $Q$ from the estimated $r$ and then calculate a loss based on $\mathcal{L}_{BE}$. Without known transition probabilities, computing $Q$ from an estimated $r$ is non-trivial, typically requiring estimation of the transition probabilities first. (This leads back to a two-stage approach—estimate $P$, then solve for $Q$/$r$—which can compound errors and is often infeasible in high dimensions.)
