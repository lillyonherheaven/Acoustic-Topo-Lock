# Replay benchmark report

No ASVspoof PA training, benchmark metrics, or deployment model have been produced in this workspace. The official PA ZIP could not be reached from the workspace shell, and a Colab runtime was not available here. The workflow tests use generated audio and verify software behavior only. No Acoustic Topo-Lock controlled recordings were used.

## Measurement protocol

The four candidates are prespecified: LFCC plus scaled Logistic Regression, all features plus scaled Logistic Regression, all features plus scaled RBF SVM, and all features plus Random Forest. The initial class balancing is fit on each candidate's training data. The SVM defaults to a reproducible class-stratified sample of up to 12,000 **training** trials for runtime; set the cap to `null` to fit all surviving training trials. Reported fit counts expose differences between candidates. The candidate with lowest development EER wins; ties go to higher development ROC-AUC and then the candidate name.

Development threshold selection minimizes absolute FAR minus FRR across realizable score cutoffs, then mean error, then favors the lower threshold to reject more replays. Exact score ties are never split. EER separately interpolates between score cutoffs where FAR and FRR cross. The interpolated EER is a ranking summary, not the deployed operating point. Evaluation uses the selected model and unchanged development threshold. Evaluation ROC, EER and DET summarize ranking across operating points for reporting only.

- Accuracy uses the frozen binary decisions. Precision, recall, F1, and ROC-AUC use replay as the positive class; precision is zero if there are no positive predictions.
- FAR here is the fraction of replay trials called live, also reported as Replay Attack Acceptance Rate. FRR is the fraction of bona fide trials called replay. These are countermeasure-only rates, not speaker verification or end-to-end authentication metrics.
- The confusion matrix is ordered live, replay on both axes, with actual labels on rows. ROC x is FRR and y is replay recall. DET plots FRR and FAR on normal deviate scales; endpoints are clipped only for display to 0.01% and 99.99%. Diagnostics mark the frozen score cutoff.
- The latency sample times ZIP read, decode, validation, feature extraction, and per-example model scoring for up to 32 evenly spaced accepted trials, with the pipeline already loaded. The JSON records the split and p50/p95. Benchmark hardware and decoder details are recorded separately; do not compare these timings directly with application wall-clock latency.
- Evaluation output includes counts and acceptance rates by attack code and environment/code pair, and whether each category or pair occurred in accepted training rows. Review trial counts before comparing groups. Real replay hardware effects and project-specific performance require separate controlled trials.

The report does not implement t-DCF. An ASV-specific protocol and compatible ASV scores would be needed for that comparison. Before production integration, run the full official benchmark, review exclusions and class-specific errors, then independently validate the operating point using consented controlled recordings representative of deployment.
