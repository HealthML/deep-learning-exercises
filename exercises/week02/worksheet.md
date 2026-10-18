Two 90-minute sessions. Follow the book's **Sections 8–10**:
[concepts](https://healthml.github.io/deep-learning-book/chapters/02-linear-classification/concepts.html),
[lab](https://healthml.github.io/deep-learning-book/chapters/02-linear-classification/lab.html),
and [optional additional material](https://healthml.github.io/deep-learning-book/chapters/02-linear-classification/additional-material.html).
The class extracts short experiments from the lab; the complete lab remains the
after-class practice, rather than another assignment to finish during class.

Open `logistic_regression.ipynb` using the
[setup instructions](https://github.com/HealthML/deep-learning-exercises/blob/main/exercises/week02/README.md).
All supplied cells run. Write answers in the notebook's **Your record** cells;
optional implementation challenges do not block later work. Predict on paper
before running each experiment, then keep the prediction alongside the evidence.
Work in pairs: the **driver** operates the notebook and the **explainer** states
the prediction and interprets the result. Swap after every lettered task. Each
person completes the exit tickets independently, without an assistant.

## Our fixed experiment

| Choice | Value |
|---|---|
| Dataset | UCI WDBC, same 569-row teaching subset as the book |
| Predictors | `concavity_mean`, `texture_mean`, in that order |
| Positive label | `malignant = 1`; benign = 0 |
| Parameters | `theta = (b, w_concavity, w_texture)`; intercept first |
| Split | Book's stratified seed 8: training 341, validation 113, test 115 |
| Preprocessing | Training mean and population standard deviation only |
| Fitting objective | Mean unpenalized Bernoulli negative log likelihood |
| Optimizer controls | CPU float64, full batches, initial logits zero |
| GD rule | `eta = 1 / lambda_max(A.T @ A / (4*n))` in each coordinate system |
| Comparison | First training loss gap to stationary reference ≤ `1e-6` |
| Primary final model | Unpenalized two-feature MLE, threshold 0.5 |

The test set remains closed during class and model exploration. Its final cell
is disabled by default, including during **Run All**. The primary model and
threshold are predeclared; short regularization/threshold explorations do not
replace that primary model. This historical teaching dataset does not constitute
a clinical diagnostic system.

## Session 1: try, understand, improve

| Minutes | Activity | Produce |
|---|---|---|
| 0–10 | Opening, Week 1 recall, start notebook | One proposed boundary |
| 10–25 | A: score, probability, decision | Hand calculation and confidence prediction |
| 25–45 | B: likelihood and log loss | Two-observation comparison and explanation |
| 45–60 | C: reuse GD, diagnose slow fitting | Competing explanations and intervention predictions |
| 60–82 | D: derive, check, standardize | Gradient check and one evidence-based diagnosis |
| 82–90 | Individual exit ticket and discussion | Three short answers |

### A. Keep score, probability, and decision separate — 15 minutes

Use $z=b+w_1x_1+w_2x_2$, $p=\sigma(z)$, and
$\hat y=\mathbf1[p\geq\tau]$. The notebook first plots **training data only**.

1. Sketch a straight boundary. Explain which mistakes a straight line cannot
   avoid. Why is always predicting benign a useful baseline but insufficient?
2. With `theta=(-3,20,0.05)`, calculate the score for inputs `(0.1,20)` and
   `(0.04,14)`. State each probability approximately and the decision at 0.5.
   State explicitly how equality at the threshold is handled.
3. Predict what changes if all three parameters are multiplied by ten:
   the 0.5 boundary, decisions, probabilities, and mean log loss. Run the cell.
4. Change at most **three** manual parameter settings. Record your reason and
   observed training loss; aim for a reasoned intervention, not a long search.

**Your record:** boundary sketch; two calculations; original prediction;
`theta → reason → training loss` table; one revised explanation.

**Support:** use the provided score formula and sigmoid plot; make the two
decisions before estimating probabilities. **Extension:** show that a threshold
$\tau\ne0.5$ corresponds to $z=\log[\tau/(1-\tau)]$.

### B. Build the objective from assigned label probabilities — 20 minutes

For labels `(1,0)`, compare proposed probabilities `(0.8,0.3)` and `(0.6,0.4)`.

1. Before running, write the probability each model assigns to **each observed
   label**, multiply, and decide which likelihood is larger. Do their accuracies
   differ at threshold 0.5?
2. Check $p^y(1-p)^{1-y}$ for $y=0$ and $y=1$. Explain why a product of these
   terms becomes a sum after taking logs and why we negate it for minimization.
3. Connect the three expressions: likelihood, mean BCE, and
   $\ell(z,y)=\log(1+e^z)-yz$. Does taking a mean instead of a sum change the
   minimizer? Does it change a gradient's magnitude?
4. Run the extreme-score check. Why do we pass **scores** into
   `BCEWithLogitsLoss`, rather than applying sigmoid first?

**Your record:** likelihood calculations, a two-sentence derivation, and a
one-sentence explanation of the logits API. No matrix derivation is needed here.

**Support:** for $y=0$ the assigned probability is $1-p$.
**Extension:** derive the stable logit loss algebraically from the Bernoulli term.

### C. Decreasing loss is not evidence of a finished fit — 15 minutes

The notebook applies Week 1 full-batch GD to the raw features for 2,000 updates.
The learning rate comes from a safe curvature bound, rather than being tuned to
create a failure.

1. Predict whether the result will be close to the best achievable fit. Run it.
2. Compare the initial and final loss with the training-prevalence baseline.
   A poor fit has at least two possible explanations: insufficient model capacity
   or unfinished optimization. What observation could distinguish them?
3. Predict the effects of more updates, changing feature units, and changing
   the update rule. Keep the rows, labels, and fitting objective fixed.

**Your record:** losses; two competing explanations; one prediction per
intervention. Do not use test data to decide which explanation is correct.

### D. Derive the update, then change the coordinates — 22 minutes

For one observation, $\partial\ell/\partial z=p-y$.

1. At zero parameters, input `(0.1,20)`, and $y=1$, obtain that observation's
   three gradient contributions in intercept-first order. Which direction should
   its score move after a small GD update?
2. Connect your scalar result to $g=A^\top(p-y)/n$ and annotate shapes:
   `A (n,3)`, `theta (3,)`, `p and y (n,)`, `g (3,)`. Explain the danger of a
   target with shape `(n,1)`. Why are thresholded predictions inappropriate here?
3. Predict the sign of each derivative at `theta=(-1,3,0.03)`. Run central
   finite differences at that **nonzero** point and compare all three entries.
4. Standardize using training statistics only; rerun GD with the same step-size
   **rule** and update budget. Before running, predict which will change:
   correlation, representable scores, attainable minimum, optimization path.
5. Map fitted standardized parameters back to raw units and check equal logits.
   State the evidence for your diagnosis of the original slow fit.

**Your record:** one-example derivative, shape annotation, numerical check,
before/after evidence, and one revised prediction.

**Support:** use $\partial z/\partial(b,w_1,w_2)=(1,x_1,x_2)$.
**Optional coding after class:** write your own gradient and reproduce 100 NumPy
GD updates in PyTorch with zero initialization and the same learning rate.

### Session 1 exit ticket — individual, 8 minutes

Without executing code or consulting an assistant:

1. Give a pair of models with identical decisions but different likelihoods.
2. If $y=1$ and $p=0.2$, give the sign of $\partial\ell/\partial z$ and the
   direction in which a loss-reducing update moves the score.
3. Name one quantity preserved by standardization and one that can change.

**Between sessions — about 30 minutes:** read Section 8 from the Newton
introduction onward; annotate one Hessian entry; rerun D and explain the scaling
result in three sentences. Bring a prediction for E. Section 10 is optional depth.

## Session 2: curvature, evaluation, and short transfers

| Minutes | Activity | Produce |
|---|---|---|
| 0–10 | Retrieval and one-dimensional Newton launch | A local quadratic update |
| 10–45 | E: solve with curvature; isolate correlation | Fair optimizer comparison and controlled intervention |
| 45–70 | F: evaluate held-out probabilities and decisions | Validation trade-off and calibration reading |
| 70–83 | G: softmax and shrinkage micro-experiments | Two transfer explanations |
| 83–90 | Individual exit ticket and next steps | Three short answers |

### E. A better local update solves a system — 35 minutes

1. For $f(u,v)=\tfrac12(4u^2+v^2)$ at `(1,2)`, write the gradient and Hessian.
   Solve $Hs=g$ by hand and subtract $s$. Compare one GD update with $\eta=1/4$.
2. For logistic regression, read the supplied `hessian` and `newton` source.
   Explain $H=A^\top\operatorname{diag}[p_i(1-p_i)]A/n$. Identify the linear
   solve, the sufficiently-decreasing-loss check, and the stopping diagnostic.
   Why do curvature and weights need to change on the next iteration?
3. Predict and compare GD and Newton, raw and standardized, from zero initial
   logits. Use the **same loss-gap target `1e-6`** for all four trajectories;
   distinguish this target from Newton's internal stopping rule. Record first
   updates to target or `not reached within 2000`. Compare fitted logits across
   coordinate systems, rather than comparing coefficient values.
4. Now isolate correlation: with standardized features form
   $u=z_1$, $v=z_1+0.08z_2$. Predict whether any information or representable
   affine scores are lost. Run GD/Newton again and compare scales, correlation,
   initial Hessian conditioning, and training loss gaps. The native features are
   moderately correlated; this deliberately transformed pair is nearly redundant.
5. Explain why fewer Newton updates do not prove a wall-clock advantage on a
   large neural network. Name one case where a Hessian solve or finite MLE fails.

**Your record:** toy update; comparison table; one paragraph distinguishing
scale, correlation, curvature, model capacity, and cost per update.

**Support:** the zero-parameter Hessian has all row weights $1/4$; supplied code
handles the solve and line search. **Optional Section 10 depth:** compute the
working response $z+(y-p)/[p(1-p)]$, and check that one weighted `lstsq` update
equals a full Newton update. Explain why one ordinary least-squares fit is not
logistic MLE. Optional implementation prompts appear after the common core.

### F. Evaluate probabilities and decisions separately — 25 minutes

Freeze the primary model as the unpenalized two-feature MLE and its primary
threshold as **0.5**. All outputs in this task use **validation**, never test.

1. Predict how lowering the threshold from 0.7 to 0.3 changes sensitivity and
   specificity. Define both using counts and evaluate at 0.3, 0.5, 0.7.
2. For the primary model at 0.5, record the confusion matrix with row/column
   labels, positive class, log loss, Brier score, and AUROC. Compare against the
   constant probability estimated from training labels. Which metric answers
   ranking, probability quality, or threshold-specific mistakes?
3. Read the five-bin validation reliability plot, including bin counts. What
   does one small bin permit you to conclude? Could a model rank very well while
   reporting poor probabilities? Why is this plot insufficient to claim clinical
   calibration?
4. Explain why choosing a threshold on the test set compromises final evaluation.
   Our threshold comparison describes validation behavior and does not alter the
   predeclared primary model.

**Your record:** threshold table, labeled primary confusion matrix, probability
metrics, and two careful statements about calibration and data use.

**Support:** sensitivity $=TP/(TP+FN)$; specificity $=TN/(TN+FP)$.
**Extension:** rescale the primary logits by a positive constant on validation;
predict which ranking and probability metrics can change, while keeping the
predeclared primary model unchanged.

### G. Transfer the loss to more classes; change the objective — 13 minutes

These are brief interpretation experiments, not full new model-building tasks.
The book lab supplies the full multiclass training and regularization exercises.

**Softmax, 6 minutes.** A separate synthetic score vector `(2,1,-1)` represents
three classes. Predict its winning class and what adding 1000 to every score
does. Run the stable softmax/CE and the two-class `(0,z)` sigmoid-equivalence
checks. Explain why multiclass CE expects scores and integer class indices, not
probabilities and binary floating-point labels. Do not invent a third WDBC label.

**L1/L2, 7 minutes.** Predict which penalty can produce exact zero weights. Run
the fixed standardized coefficient paths for the book's seven strengths. Read
one exact L1 zero and compare the L2 coefficient at that same strength. State
which parameter is unpenalized, why standardization matters for the penalty,
and why selecting a penalty uses **unpenalized validation log loss**, rather
than comparing two different penalized training objectives. Sparse coefficients
do not establish causal irrelevance. These paths are a separate validation
exercise and do not replace the frozen primary model.

**Your record:** one softmax explanation and one shrinkage explanation.
**After class:** finish lab checkpoints 5–6, including synthetic three-class
training, the larger invariance check, and interpretation of full paths.

### Session 2 exit ticket — individual, 7 minutes

1. Why is logistic stationarity not the linear normal equation from Week 1?
2. What is preserved when raw and standardized Newton fits agree, and why do
   coefficient values differ? Give one reason not to use dense Newton for a
   neural network with millions of parameters.
3. A validation model has high AUROC. Name two further questions you must answer
   before interpreting its probabilities and decisions.

## Final checkpoint after explanations and choices are written

The final notebook cell sets `OPEN_TEST = False`. Leave it unchanged until your
primary model, threshold, preprocessing, explanations, and validation work are
finished. Then change it to `True` once to assess the frozen model. Report all
results, keep the same model and threshold, and do not tune after seeing test
results. A new exploratory model needs new evaluation data. If class ends first,
leave this checkpoint for the lab.

Keep predictions, pair explanations, individual exit tickets, and a runnable
notebook. The code remains readable and transparent in
`src/dlexercises/week02.py`. There is no answer key in the student materials.

## Sources

Adapted from HealthML/deep-learning-book, Chapter 02 (concepts, lab, additional
material) and `src/dlbook/logistic.py`, commit
`7803e0bcbbe1b20d7563db09cee97f4976242e6b`, Christoph Lippert and contributors.
Prose and exercises: CC BY-SA 4.0; source code: MIT. WDBC: Wolberg, Mangasarian,
Street & Street (1993), [UCI DOI 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B),
CC BY 4.0. The data notes and checksum manifest are packaged with the CSV.
