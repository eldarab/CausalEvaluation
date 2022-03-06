# Causal Evaluation
A Causal Benchmark for Explanation Methods in NLP

## Authors 
[Eldar Abraham](https://www.youtube.com/watch?v=dQw4w9WgXcQ), [Amir Feder](https://scholar.google.com/citations?user=ERwoPLIAAAAJ&hl=en&oi=ao), [Roi Reichart](https://ie.technion.ac.il/~roiri/)

## Purpose
Provding a causal benchmark for explanation nethods (e.g. [CausaLM](https://arxiv.org/abs/2005.13407)) and conducting experiments on it.

## Abstract
Deep neural networks for natural language processing boast remarkable predictive capabilities. But can you trust your model? Is it safe? Does it provide ethical predictions? Tools for explaining such opaque learning machines are quickly emerging, yet the criteria for assessing these tools’ quality are still unclear. There have been some attempts to construct evaluation frameworks for such explanation tools, but unfortunately they are lacking; they often mix between correlation and causation, and they do not rigorously state their (causal) assumptions. To bridge this gap, we introduce the causal estimation challenge for explanations and provide CausalConceptData, a humangenerated counterfactual-based NLP dataset. Equipped with the dataset, we conduct experiments comparing the quality of several explanation methods and demonstrate the complexity of the causal estimation challenge.

## Experiments
All experiments appear under "experiments" folder, and include a detailed README file regarding each experiment.
