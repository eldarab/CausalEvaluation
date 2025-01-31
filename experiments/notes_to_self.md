# Notes To Self

## TODO

### 28.06.2021
* Good enough causal graph.
* Major sections of the writing, Overleaf project.
* Saved models that give results.
* Create a spreadsheet with predictions.
* Add label date to annotations


### 29.06.2021 Meeting with Amir
* Create an agressive version of confounding.
* Label the full the version of the concept linguistic acceptability.
* Implement INLP baseline.
* Read [continuation of INLP](https://arxiv.org/abs/2105.06965), existence of concept instead of counterfactual
* Ask Roi what he thinks about להאדיר INLP


1. Data Collection
2. Run CausaLM
3. Baselines (INLP)
4. Writing


* Draw an outline of the paper. Will deduce the experimental setup and data collection. 2 versions on 2 concepts.
* Subsection: optimizing adversarial methods for causal estimation.
* A lot of data creation and data curation.
* Expand concept 


#### Action items:
0. Schedule week-level
1. eCommerce Lecture 
2. Parse new Amitava data
3. Run experiments of acceptability with new data from Amitava
4. Create an experimental setup - tons of causal graphs and concepts.
5. Map holes in project:
    * Develop code according to the setup chosen (token level classification).
    * Implement INLP.
    * Collect more data for concepts
6. Run all experiments with new data and code pipeline: baselines & token level.
7. Choose among these experiments the best ones.
8. Start writing 18.07.2021


## Code assignments
* Token-level classification.
* INLP Baseline.
* Make sure metrics are correct.
* Aggressive-correlation experiments.
* Full experimental pipeline:
   * Did we forget the TC?
   * Did we remember the CC?
   * ATE_gt
   * INLP_ATE
   * CONEXP
   * TReATE
   * BERT-O


## random

* What does that mean to compute ate on TC/CC?
* Multiple control concepts?
* How to compute CONEXP on a continuous concept?
* Is Masked Product?
* Domain-Related Entity
* the Predicted sentiment vs. the Generating sentiment



## Why estimation using INLP may fail

1. It's like comparing oranges and apples. CausaLM makes causal model explanation, and INLP only takes the CLS token of the representation.
2. I'm not sure if I should project the training data (X @ P) before training a classifier or at inference time. 
3. INLP has **linear** guardedness.
4. INLP has no control. 
5. Estimates 0 when no ATE_gt is 0.
6. Can't count on INLP code.
7. Can't count on old Causalm code.
8. can only guard "sequence level" attributes.
