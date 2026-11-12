# Known Issues in the Reported Results

This folder documents four protocol issues in the reported experiments and holds the code and
results used to check them.

## 1. Concorde and PRISM were trained for different numbers of steps

**Issue.** PRISM takes many more gradient steps than Concorde. At 5% labels, Concorde takes 1,521
steps, V3 takes 6,084 and PRISM takes 13,689.

**Cause.** We matched Concorde's 1,521 *epochs* but did not account for the different number of
batches per epoch. As a result, we attributed the extra training steps to the unlabeled data.

**Check** (`step_matched/`, CPI MAPE %, mean of 3 seeds):

| Model | Steps | 5% | 20% | 50% |
|---|---|---|---|---|
| Concorde | 1,521 | 9.84 | 6.63 | 3.62 |
| Concorde | 6,084 | 9.17 | 5.30 | 3.33 |
| Concorde | 12,168 | 9.00 | 4.50 | – |
| PRISM | 13,689 | 7.45 | 4.18 | 3.26 |
| PRISM, L_const_self off | 13,689 | 7.43 | 4.13 | 3.34 |

More steps alone substantially lower Concorde's error (6.63 → 4.50 at 20%). Turning off the
unlabeled consistency loss barely changes PRISM's error.

## 2. Checkpoints were selected on the test set

**Issue.** Concorde and PRISM both pick the best checkpoint on the test split. Only the OOD
experiment has a separate validation set.

**Cause.** We followed Concorde's training protocol, a 15:1 train/test split per benchmark, and did
not check that it lacked a validation split.

## 3. Better seeds were reported in the label-efficiency and OOD experiments

**Issue.** The label-efficiency and OOD results report the better-performing seeds rather than a
mean over seeds. The label-efficiency numbers come from `micro_exp/exp4/results_77/` (one run per
label ratio; the JSONs do not record the seed). Multi-seed means should be reported instead (see
`k4_none_labeled_only/`, `ood_32/`, ood_32/results_ood_lowlabel_grid_nod3_wfix_1026). 

## 4. Sec. 5 (DSE): L2 disabled and feature mismatch

**Issue.** The Sec. 5 simulations ran without an L2 cache. In addition, the gem5 validation MAPE was
large.

**Cause.** The DSE engine computed features differently from the training feature pipeline. After
recomputing the features design by design with the training pipeline
(`data_gen/concord_analytical`), the MAPE decreased (see `traj/`). The reported traj results use
only benchmarks whose validation checkpoints are close to the training checkpoints; imagick is
excluded (see `traj/validation/`, `checkpoint_distance.md`).

## Future Protocol (transparent development and traceable result record)

1. Commit the code, results and logs behind every number in the paper to a dedicated repository.
2. Provide a script that reproduces each number.
3. Every time making a commit to the repository, make a log of what/why.
4. Include a detailed README that maps each number in the paper to its script, result file and log
   in the repository, so that an agent can recheck all of them systematically.

## Future Plan

1. **Find more useful invariants.** The current invariants do not help, mainly because they are
   too loose. They do not pin the predicted values the way the physics constraints in a PINN do.
2. **Use the MSCS predictions themselves.** No prior work has clustered CPI stacks or analyzed their
   balance with Jain's index across a whole design space. The open questions are how to make these
   results actionable and what they mean for design. Different CPI-stack accounting algorithms
   should also be compared.
3. **Experiment plan.**
   - Regenerate the training data with a design space closer to Arm Neoverse N1. Simulate more
     checkpoints per benchmark, so that the dataset is more diverse and windows overlap less.
   - Compare the MSCS against top-down slot accounting measured on real Neoverse N1 hardware.
   - Generate MSCS labels with different accounting algorithms and compare them.
   - Add MSCS clustering and Jain's-index analysis to the agent DSE loop as skills.
   - Keep L_const, but hard-code it into the model so that consistency holds by construction,
     instead of adding it as a weighted soft loss term.
