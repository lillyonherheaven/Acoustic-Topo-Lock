# ASVspoof 2019 simulated PA protocol

The [challenge page](https://www.asvspoof.org/index2019.html) distinguishes the simulated 2019 PA release from the later, small real PA collection. This workflow uses only the simulated [Edinburgh DataShare deposit](https://doi.org/10.7488/ds/2555), where the publisher provides PA.zip and attributes the public corpus under the [Open Data Commons Attribution License](https://datashare.ed.ac.uk/bitstreams/60c7de6d-37d3-45c1-b52e-9f18b0338e47/download). If the provider asks for an access step, stop and obtain authorization at the source. The code does not handle passwords, cookies, or access tokens.

The [release README](https://datashare.ed.ac.uk/bitstreams/765597d6-633f-418a-96b0-39274a8d56c9/download) documents the PA directory layout. The [official evaluation plan v0.4](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf) identifies the CM keys, and the [organizers' baseline code](https://github.com/asvspoof-challenge/2021/blob/main/PA/Baseline-CQCC-GMM/matlab/CQCC_GMM_ASVspoof_2021_baseline.m) reads the 2019 PA training key with five whitespace-separated fields:

| Position | Meaning |
|---|---|
| 1 | Speaker ID |
| 2 | Utterance ID, matched to a FLAC basename in its own partition |
| 3 | PA acoustic environment category |
| 4 | PA replay condition category (recording distance and playback quality), or `-` for bona fide |
| 5 | Key: exactly `bonafide` (live, 0) or `spoof` (replay, 1) for labeled trials |

The module records actual member paths, key SHA-256 hashes, column counts, class counts and speaker counts at runtime. It locates the unique official PA CM key directory, checks exact labeled field counts, utterance IDs and corresponding audio paths, rejects duplicate members/trials and checks speaker disjointness before accepting a partition. It does not walk ASV enrollment keys, infer labels from audio or filenames, use LA data or create new train/test splits. Expected challenge train and development CM totals are 54,000 and 29,700 before quality exclusions; these are published reference counts. Missing or recognized label-free evaluation keys suppress evaluation. Malformed or partially labeled keys fail closed.

**Verification status:** Publisher documentation, release link and baseline format were reviewed from this workspace. Network access to the official ZIP from the shell failed; the actual archive contents and key records were therefore not independently inspected here. The Colab inspection cell must succeed on the real ZIP before the smoke or full run. Tests use a generated archive with the documented layout and exercise both missing and labeled evaluation keys.

In PA, evaluation uses different simulated acoustic and replay configurations drawn from the same categories, as described in Section 4.2 of the evaluation plan and in the [dataset paper](https://www.eurecom.fr/publication/6352/download/sec-publi-6352.pdf). Replay condition breakdowns tabulate observed evaluation categories and environment/category pairs, including membership among retained training rows. The CM fields do not identify individual loudspeakers in this workflow. No device-held-out split or performance on Acoustic Topo-Lock's controlled rooms can be inferred.

For dissemination, cite Todisco et al., *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection*, Interspeech 2019, and attribute the [original corpus deposit](https://doi.org/10.7488/ds/2555).
