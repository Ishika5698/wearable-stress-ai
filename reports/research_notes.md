# Research notes: wearable-stress-ai

## Status

**Model execution: pending.** No WESAD participant pickle files were found in the workspace when this project was prepared. No participant counts from local files, plots, model metrics, or experimental findings are claimed. The notebooks only produce data-derived outputs after the dataset is obtained and inspected.

## Research question and protocol

The practice question is whether chest ECG, electrodermal activity (EDA), and temperature features distinguish the WESAD baseline, stress-task, and amusement-task labels for participants held out from model fitting. The initial task uses 30-second windows with 15-second steps, retaining only windows with a single target label throughout. The supplied feature extractor computes mean, standard deviation, minimum, maximum, median, interquartile range, RMS, and finite-value fraction. Evaluation compares a most-frequent-class dummy model with a class-weighted random forest using five-fold stratified group cross-validation, with participants as groups. Imputation is fitted within each training fold. Window start time is excluded from model inputs.

This configuration is a starting point for learning and should be reviewed against the actual release README and file structure before interpreting results. The code checks expected channel names and label alignment at runtime. The standard label mapping is represented in code, but must be confirmed against the README that accompanies the official archive.

## Dataset and measurement context

WESAD is a controlled laboratory wearable-sensing dataset. Its documentation describes chest-worn RespiBAN signals including ECG, EDA, EMG, respiration, temperature and three-axis acceleration, sampled at 700 Hz. The wrist-worn Empatica E4 provides BVP at 64 Hz, EDA and temperature at 4 Hz, and three-axis acceleration at 32 Hz. The original paper describes 15 retained participants after two recruited participants were excluded for sensor malfunction. It describes a neutral-reading baseline, amusement videos, and a stress task involving public speaking and mental arithmetic (Trier Social Stress Test), along with self-report questionnaires.

These labels are experimental protocol conditions and do not directly measure an individual's internal state. Task content, movement, posture, timing, device placement, individual physiology, and signal quality may all affect model predictions. Participant-grouped evaluation reduces participant leakage, but it does not demonstrate transfer beyond this laboratory dataset. The model must not be presented as a stress diagnosis, driver-fatigue detector, or tool for medical or motorsport decisions.

## Access and terms

The University of Siegen source page links a roughly 2.5 GB archive and states use for scientific, non-commercial purposes with attribution when publishing. Use the current official page and follow any terms presented by the download host. Do not redistribute the archive or put participant records or derived participant-level data on GitHub. The project's `.gitignore` excludes raw and processed data and generated metric CSVs. The UCI dataset page is useful metadata, while access should follow the owners' source page.

## Results template (to fill only after a real run)

- Dataset archive/version and date obtained:
- Participants actually inspected:
- Actual channels, shapes and rates found:
- Label mapping confirmed from release README:
- Window duration / step:
- Number of retained windows by class and participant:
- Grouped fold design and participant overlap audit:
- Majority baseline accuracy / macro F1:
- Random forest accuracy / macro F1:
- Per-class precision / recall / F1:
- Confusion matrix interpretation:
- Signal quality, missingness, and deviations from expected structure:
- Limitations and follow-up experiments:

Do not fill these fields with estimates or numbers copied from another paper. The original paper's results use its own benchmark design and are not results for this project's pipeline.

## Sources

- University of Siegen, official WESAD dataset page: <https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/>
- UCI Machine Learning Repository, WESAD metadata: <https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection>
- Schmidt et al., “Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection,” ICMI 2018: <https://kristofvl.github.io/usi/pdf/ubi_icmi2018.pdf>
