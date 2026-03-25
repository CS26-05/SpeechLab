# System Tests — SpeechLab Diarization Pipeline

**Branch:** `testing1`
**Tester prerequisites:**
- Access to Gonzaga's ada server and seas-gpu-node-01
- A HuggingFace account with access to:
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- Your HuggingFace token ready

---

## ST-I: Copy Repo to GPU and Verify Reference Data

**Type:** System Test
**Assigned to:** _______________

**Hypothesis:** The repository can be cloned onto the GPU node and the pre-generated reference RTTM files are present and correctly formatted.

---

### Step 1: SSH into ada

```bash
ssh YOUR_USER@ada.gonzaga.edu
```

Were there any issues SSHing into ada?

&nbsp;

---

### Step 2: SSH into seas-gpu-node-01

```bash
ssh YOUR_USER@seas-gpu-node-01
```

Were there any issues SSHing into seas-gpu-node-01?

&nbsp;

---

### Step 3: Clone the repository

```bash
git clone -b testing1 https://github.com/CS26-05/SpeechLab.git
cd SpeechLab
```

Were there any issues cloning the repository?

&nbsp;

---

### Step 4: Verify reference data is present

Check how many reference RTTM files are in `test_reference/`:
```bash
ls test_reference/ | wc -l
```
How many files are listed? (Expected: 159)

&nbsp;

Check how many wav files are in `test_input/`:
```bash
ls test_input/ | wc -l
```
How many files are listed? (Expected: 158)

&nbsp;

---

### Step 5: Verify RTTM format of reference files

Open a reference file and inspect the first 5 lines:
```bash
head -5 test_reference/AR31_021108a.rttm
```

Copy what you see here:

&nbsp;

Check that every line has exactly 10 fields (this command prints nothing if the format is correct):
```bash
awk 'NF != 10 {print "BAD LINE:", NR, $0}' test_reference/AR31_021108a.rttm
```

Did the command print anything? If yes, copy it here:

&nbsp;

What label appears in field 8 of the first line? (Expected: one of `FEM`, `MAL`, `KCHI`, `OCH`, `UNK`, `SIL`)

&nbsp;

**Background — RTTM format reference:**

Each line represents one speech segment:
```
SPEAKER AR31_021108a 1 12.340 2.500 <NA> <NA> FEM <NA> <NA>
```

| Position | Value | Meaning |
|----------|-------|---------|
| 1 | `SPEAKER` | Always this word |
| 2 | `AR31_021108a` | Recording name |
| 3 | `1` | Channel (always 1) |
| 4 | `12.340` | Start time in seconds |
| 5 | `2.500` | Duration in seconds |
| 6–7 | `<NA> <NA>` | Unused placeholders |
| 8 | `FEM` | Speaker label |
| 9–10 | `<NA> <NA>` | Unused placeholders |

---

### Step 6: Inspect the speaker mapping CSV

Open `data/cha_to_vtc2_speaker_map.csv` in a text editor or spreadsheet:
```bash
cat data/cha_to_vtc2_speaker_map.csv
```

What columns do you see?

&nbsp;

Is the `vtc_label` column filled in for every row, or are some rows blank?

&nbsp;

Do the speaker codes (e.g. `MOT`, `CHI`, `FAT`) and their assigned labels make sense based on the `gender` and `role` columns? Are any rows hard to categorize?

&nbsp;

---

### Step 7: Check for UNK labels in the reference files

```bash
grep " UNK " test_reference/*.rttm
```

Did the command print anything? If yes, copy it here (note which files and what speaker codes triggered UNK):

&nbsp;

---

### Step 8: (Optional) Reproducibility check

Run `cha2rttm.py` yourself and diff the output against the pre-provided reference:
```bash
pip install -r requirements.txt
python scripts/cha2rttm.py data/cha_files/ \
  -m data/cha_to_vtc2_speaker_map.csv \
  -o /tmp/st1_output/
diff /tmp/st1_output/AR31_021108a.rttm test_reference/AR31_021108a.rttm
```

Did `diff` print any differences?

&nbsp;

**Pass criteria:** 159 RTTM files present, 158 wav files present, no bad lines printed by `awk`, field 8 is a valid label, no unexpected UNK labels, diff produces no output.

| Result | Notes |
|--------|-------|
| Pass / Fail | |

---

## ST-II: Run the Pipeline on GPU Inside Apptainer

**Type:** System Test
**Assigned to:** _______________

**Hypothesis:** The Apptainer container builds successfully from the `.def` file and processes all wav files in `test_input/`, producing one RTTM output per file in `test_output/`.

---

### Step 1: Build the Apptainer image

```bash
apptainer build --fakeroot speechlab.sif speechlab.def
```

This takes around 10–15 minutes. Did the `.sif` file build successfully?

&nbsp;

Were there any error messages during the build? Copy them here:

&nbsp;

---

### Step 2: Create the output directory

```bash
mkdir -p test_output
```

---

### Step 3: Run the pipeline

Replace `YOUR_USER` with your username and `YOUR_TOKEN` with your HuggingFace token:

```bash
apptainer exec --nv \
  --pwd /app \
  --bind /home/YOUR_USER/SpeechLab/test_input:/data/input \
  --bind /home/YOUR_USER/SpeechLab/test_output:/data/output \
  --env HF_TOKEN="YOUR_TOKEN" \
  --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  /home/YOUR_USER/SpeechLab/speechlab.sif \
  python -m speechlab_diarization.main
```

Did it process the files without errors?

&nbsp;

How many `.rttm` files are in `test_output/`?
```bash
ls test_output/ | wc -l
```

&nbsp;

**Pass criteria:** One `.rttm` file in `test_output/` for each `.wav` in `test_input/`.

| Result | Notes |
|--------|-------|
| Pass / Fail | |

---

## ST-III: Data Sanitization Verification

**Type:** System Test
**Assigned to:** Anh

**Hypothesis:** The pre-generated reference RTTM files in `test_reference/` are correctly formatted and the speaker mapping CSV is complete and reasonable. Any unmapped speaker code produces `UNK` rather than silently dropping the segment.

---

### Step 1: Inspect the speaker mapping CSV

```bash
cat data/cha_to_vtc2_speaker_map.csv
```

What columns do you see?

&nbsp;

Is the `vtc_label` column filled in for every row, or are some rows blank?

&nbsp;

Do the speaker codes (e.g. `MOT`, `CHI`, `FAT`) and their assigned labels make sense based on the `gender` and `role` columns? Are any rows hard to categorize?

&nbsp;

In your opinion, is the mapping clear enough to understand without additional explanation? What would make it easier to verify?

&nbsp;

---

### Step 2: Check for UNK labels in the reference files

```bash
grep " UNK " test_reference/*.rttm
```

Did the command print anything? If yes, copy it here and note which files/codes triggered UNK:

&nbsp;

---

### Step 3: (Optional) Reproducibility check

Run `cha2rttm.py` yourself and diff against the pre-provided reference:

```bash
pip install -r requirements.txt
python scripts/cha2rttm.py data/cha_files/ \
  -m data/cha_to_vtc2_speaker_map.csv \
  -o /tmp/st3_output/
diff /tmp/st3_output/AR31_021108a.rttm test_reference/AR31_021108a.rttm
```

Did `diff` print any differences?

&nbsp;

**Pass criteria:** `vtc_label` filled for all rows, no unexpected UNK labels, diff produces no output.

| Result | Notes |
|--------|-------|
| Pass / Fail | |

---

## ST-IV: Evaluation Script Correctness

**Type:** System Test
**Assigned to:** Chen

**Hypothesis:** The evaluation script correctly computes DER. When reference and hypothesis are identical, DER = 0. When the hypothesis is empty, DER > 0. Batch mode produces a per-file breakdown and an overall average.

> **Note:** DER = 0 is NOT the goal in practice — the pipeline will always produce nonzero DER because that is what we fine-tune against. This test only checks that the metric itself is working correctly.

---

### ST-IV-A: DER = 0 on identical ref/hyp


If reference and hypothesis are the same file, the system should return DER = 0.

```bash
apptainer exec --nv \
  --pwd /app \
  --bind /home/YOUR_USER/SpeechLab:/workspace \
  --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  /home/YOUR_USER/SpeechLab/speechlab.sif \
  python /workspace/speechlab_diarization/evaluate.py \
  --ref /workspace/test_reference/AR31_021108a.rttm \
  --hyp /workspace/test_reference/AR31_021108a.rttm \
  --uri AR31_021108a
```

What DER value is reported?

&nbsp;

**Pass criteria:** DER = 0.0000

| Result | Notes |
|--------|-------|
| Pass / Fail | |

---

### ST-IV-B: DER > 0 on empty hypothesis

When the hypothesis RTTM is empty, all reference speech is missed, so DER should be greater than 0.

```bash
touch /home/YOUR_USER/SpeechLab/test_output/empty.rttm

apptainer exec --nv \
  --pwd /app \
  --bind /home/YOUR_USER/SpeechLab:/workspace \
  --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  /home/YOUR_USER/SpeechLab/speechlab.sif \
  python /workspace/speechlab_diarization/evaluate.py \
  --ref /workspace/test_reference/AR31_021108a.rttm \
  --hyp /workspace/test_output/empty.rttm \
  --uri AR31_021108a
```

What DER value is reported?

&nbsp;

Did the tool handle the empty file without crashing?

&nbsp;

**Pass criteria:** DER > 0, no crash.

| Result | Notes |
|--------|-------|
| Pass / Fail | |

---

### ST-IV-C: Batch evaluation summary

Batch mode should produce a per-file metrics table and a final average row.

```bash
apptainer exec --nv \
  --pwd /app \
  --bind /home/YOUR_USER/SpeechLab:/workspace \
  --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  /home/YOUR_USER/SpeechLab/speechlab.sif \
  python /workspace/speechlab_diarization/evaluate.py \
  --ref_dir /workspace/test_reference/ \
  --hyp_dir /workspace/test_output/
```

Does the output include a per-file row for each recording?

&nbsp;

Does an `AVERAGE` row appear at the bottom?

&nbsp;

What metric columns appear in the table?

&nbsp;

**Pass criteria:** Per-file rows present, AVERAGE row present, columns include DER, JER, Purity, Coverage, F-measure, DetER.

| Result | Notes |
|--------|-------|
| Pass / Fail | |
