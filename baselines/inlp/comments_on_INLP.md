# Why estimation using INLP may fail

1. It's like comparing oranges and apples. CausaLM makes causal model explanation, and INLP only takes the CLS token of the representation.
2. I'm not sure if I should project the training data (X @ P) before training a classifier or at inference time. 
3. INLP has **linear** guardedness.
4. INLP has no control. 
5. Estimates 0 when no ATE_gt is 0.
6. Can't count on INLP code.
7. Can't count on old Causalm code.