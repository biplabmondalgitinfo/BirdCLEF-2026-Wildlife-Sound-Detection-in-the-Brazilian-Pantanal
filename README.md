# BirdCLEF-2026-Wildlife-Sound-Detection-in-the-Brazilian-Pantanal
Automated species identification from passive acoustic monitoring (PAM) recordings across 150,000+ km² of Brazil's Pantanal wetlands — one of the world's most biodiverse and threatened ecosystems.
Table of Contents

Overview
Competition Details
Dataset
Solution Architecture
Repository Structure
Quick Start
Notebook Cells
Configuration
Results
Troubleshooting
Acknowledgements


Overview
This repository contains a complete end-to-end Kaggle notebook solution for the BirdCLEF+ 2026 competition. The goal is to identify 234 wildlife species (birds, amphibians, mammals, reptiles, insects) from 5-second windows of continuous soundscape recordings.
The approach uses:

Log-mel spectrograms as the audio representation
EfficientNet-B0 pretrained on ImageNet, fine-tuned on single-channel mel inputs
SpecAugment + Mixup for data augmentation
Label-smoothed BCE loss with soft secondary labels
3× Test-Time Augmentation (TTA) with time-shifted inference passes
Graceful fallback to species-frequency priors when no audio data is present


Competition Details
PropertyValueCompetitionBirdCLEF+ 2026HostKaggle / LifeCLEFTaskMulti-label species classificationMetricMacro-averaged ROC-AUC (skipping absent classes)Audio formatOGG, 32 kHz, monoSegment length5 secondsClasses234 speciesRuntime constraintCPU ≤ 90 min · GPU disabled (1 min only)Submissionsubmission.csv with probability scores

Dataset
Data is available at the competition page.
/kaggle/input/competitions/birdclef-2026/
│
├── train.csv                        # Metadata for train_audio recordings
├── taxonomy.csv                     # 234 species with class info (Aves, Mammalia, etc.)
├── train_soundscapes_labels.csv     # Expert-annotated 5s segments from soundscapes
├── sample_submission.csv            # Submission format (row_id + 234 species columns)
├── recording_location.txt           # Pantanal recording site info
│
├── train_audio/                     # Individual species recordings (XC + iNat)  ~30 GB
│   └── [species]/[file_id].ogg
│
├── train_soundscapes/               # Continuous 1-min recordings for training  ~2 GB
│   └── BC2026_Train_XXXX_*.ogg
│
└── test_soundscapes/                # 600 × 1-min recordings (hidden at submission)
    └── BC2026_Test_XXXX_*.ogg
    Row ID Format
Each row_id in the submission represents the end time of a 5-second segment:
BC2026_Test_0001_S05_20250227_010002_20
│                                    └─ end time = 20s  (covers 15s–20s)
└─ soundscape filename stem

Solution Architecture
Input OGG (32 kHz)
       │
       ▼
Log-Mel Spectrogram
  n_mels=128, hop=320
  fmin=50, fmax=16000
  shape: (128, 500) per 5s
       │
       ▼
Augmentation (train only)
  SpecAugment: T-mask=40, F-mask=20
  Mixup: α=0.3  (50% of batches)
       │
       ▼
EfficientNet-B0 (timm)
  in_chans=1  (single-channel mel)
  global_pool='avg'
  feat_dim=1280
       │
       ▼
Head: Dropout(0.3) → Linear(1280, 234)
       │
       ▼
Sigmoid → Probabilities [0, 1]

Loss: BCEWithLogitsLoss + label smoothing ε=0.05
      primary weight=1.0 · secondary weight=0.5

Optimizer : AdamW  lr=1e-3  wd=1e-4
Scheduler : CosineAnnealingLR  T_max=15  η_min=1e-6

Inference : 3× TTA (base + shift +0.5s + shift −0.5s)
            clip probs to [0.01, 0.99]

Repository Structure
birdclef-2026/
│
├── README.md                   ← You are here
├── birdclef2026.ipynb          ← Complete Kaggle notebook (16 cells)
├── LICENSE
└── .gitignore
Clone the repo
bashgit clone https://github.com/YOUR_USERNAME/birdclef-2026.git
cd birdclef-2026
2. Upload to Kaggle

Go to kaggle.com/competitions/birdclef-2026
Open your notebook editor → File → Import Notebook
Upload birdclef2026.ipynb
Click Add Data → Competitions → search birdclef-2026 → Add
Set Internet to Off (required for submission)
Set Accelerator to None (GPU runtime = 1 min only)
Click Run All → Save Version → Submit

3. Local development (optional)
bashpip install torch torchvision timm librosa scikit-learn tqdm pandas numpy
jupyter notebook birdclef2026.ipynb

Update CFG.base_dir to point to your local data directory.


Notebook Cells
CellPurposeKey output1File listingPrints all input paths2Install dependenciestimm, librosa3All imports + seedDevice detection4CFG config class + directory pathsAll hyperparameters5Load CSVs + detect audio presencehas_train_audio, has_train_sc, has_test_sc6Audio utilitiesload_clip, to_melspec, spec_augment, parse_start_seconds7Label buildersbuild_sc_label_vector, build_train_label_vector8Build training recordsrecords list (soundscape + audio combined)9BirdDataset classPyTorch Dataset with random crop / exact segment10BirdModel + SmoothBCEEfficientNet-B0 with mel input11Training functionstrain_one_epoch, validate, run_training12Run trainingBest model saved to /kaggle/working/best_model.pt13Inference utilitiesTestDataset, build_test_segments, infer14Run inference + TTAprobs_final array15Build & save submission/kaggle/working/submission.csv16Final validationShape, NaN, Inf, column order checks
Every cell ends with ✓ Cell N complete for easy progress tracking.

Configuration
All hyperparameters live in CFG (Cell 4). Key settings:
pythonclass CFG:
    # Audio
    sr              = 32_000       # sample rate
    duration        = 5            # seconds per segment
    n_mels          = 128          # mel frequency bins
    hop_length      = 320          # 10ms per frame → 500 frames per 5s
    fmin, fmax      = 50, 16_000   # frequency range

    # Training
    epochs          = 15
    batch_size      = 32
    lr              = 1e-3
    label_smoothing = 0.05
    mixup_alpha     = 0.3
    secondary_weight= 0.5          # weight for secondary species labels
    min_rating      = 3.0          # min XC quality rating (0 = keep unrated)

    # Model
    model_name      = 'efficientnet_b0'
    pretrained      = True

    # Inference
    tta             = True         # 3× test-time augmentation
    prob_clip_low   = 0.01
    prob_clip_high  = 0.99
To speed up training for testing, reduce epochs to 5 or batch_size to 16.

Results
ConfigurationVal ROC-AUCPrior-based (no audio)baselineEfficientNet-B0, 15 epochs, soundscape onlyTBDEfficientNet-B0, 15 epochs, soundscape + train_audioTBD+ TTA (3×)TBD

Fill in your scores after running!


Troubleshooting
NameError: name 'train_audio_dir' is not defined
Run cells in order from Cell 1. Directory variables are defined in Cell 4.
ValueError: could not convert string to float: '00:00:00'
Fixed in Cell 6 via parse_start_seconds() — supports HH:MM:SS, MM:SS, and plain float formats.
Total training records: 0
No audio .ogg files found. Add the competition dataset:
Notebook → Add Data → Competitions → birdclef-2026
The minimum needed is train_soundscapes/ (~2 GB).
Submission takes > 90 minutes
Reduce CFG.epochs to 8 and set CFG.tta = False to stay within CPU time limit.
GPU kernel disabled
Expected — GPU notebooks get only 1 minute. Use CPU accelerator.

Acknowledgements
Dataset compiled with support from:

K. Lisa Yang Center for Conservation Bioacoustics
Instituto Nacional de Pesquisa do Pantanal (INPP)
Xeno-canto — Willem-Pier Vellinga, Bob Planqué
iNaturalist — Grant van Horn
LifeCLEF — Alexis Joly, Henning Müller
Chemnitz University of Technology — Stefan Kahl, Mario Lasseck, Maximilian Eibl
Bezos Earth Fund AI for Climate and Nature Grand Challenge

Banner photo: Hyacinth Macaw by Thomas Fuhrmann · Jaguar by Leonardo Ramos.

License
MIT License — see LICENSE for details.

<div align="center">
  <sub>Built for BirdCLEF+ 2026 · Listening carefully to protect the Pantanal 🌿</sub>
</div>
