
---

## 13. Running jobs on Kaggle

The local machine has no GPU. All training runs on Kaggle through the API.
Kaggle username: bhavishyaadhikari

Each job is a folder under `jobs/`:

    jobs/03_text_branch/
      kernel-metadata.json
      script.py

kernel-metadata.json:

    {
      "id": "bhavishyaadhikari/care-voice-text-branch",
      "title": "care-voice-text-branch",
      "code_file": "script.py",
      "language": "python",
      "kernel_type": "script",
      "is_private": true,
      "enable_gpu": true,
      "enable_internet": true,
      "dataset_sources": ["paultimothymooney/medical-speech-transcription-and-intent"]
    }

Run with:

    kaggle kernels push -p jobs/03_text_branch
    kaggle kernels status your_actual_username/care-voice-text-branch
    kaggle kernels output your_actual_username/care-voice-text-branch -p results/

Rules:
- Scripts write everything to /kaggle/working/ so it comes back in the output.
- Debug logic locally on a 50-row sample before pushing. The push-run-pull cycle costs
  minutes and the GPU quota is 30 hours a week shared by three people.
- Chain jobs by output. Upload a job's results as a private Kaggle Dataset with
  `kaggle datasets create`, then list it in the next job's dataset_sources. Never
  recompute Whisper transcripts or WavLM features.
- Always read the metrics in the output, not just the exit status.
- The medical dataset mounts at:
  /kaggle/input/medical-speech-transcription-and-intent/Medical Speech, Transcription, and Intent/
  with overview-of-recordings.csv and recordings/{train,test,validate}/

---

## 13. Running jobs on Kaggle

The local machine has no GPU. All training runs on Kaggle through the API.
Kaggle username: bhavishyaadhikari

Each job is a folder under `jobs/`:

    jobs/03_text_branch/
      kernel-metadata.json
      script.py

kernel-metadata.json:

    {
      "id": "bhavishyaadhikari/care-voice-text-branch",
      "title": "care-voice-text-branch",
      "code_file": "script.py",
      "language": "python",
      "kernel_type": "script",
      "is_private": true,
      "enable_gpu": true,
      "enable_internet": true,
      "dataset_sources": ["paultimothymooney/medical-speech-transcription-and-intent"]
    }

Run with:

    kaggle kernels push -p jobs/03_text_branch
    kaggle kernels status bhavishyaadhikari/care-voice-text-branch
    kaggle kernels output bhavishyaadhikari/care-voice-text-branch -p results/

Rules:
- Scripts write everything to /kaggle/working/ so it comes back in the output.
- Debug logic locally on a 50-row sample before pushing. The push-run-pull cycle costs
  minutes and the GPU quota is 30 hours a week shared by three people.
- Chain jobs by output. Upload a job's results as a private Kaggle Dataset with
  `kaggle datasets create`, then list it in the next job's dataset_sources. Never
  recompute Whisper transcripts or WavLM features.
- Always read the metrics in the output, not just the exit status.
- The medical dataset mounts at:
  /kaggle/input/medical-speech-transcription-and-intent/Medical Speech, Transcription, and Intent/
  containing overview-of-recordings.csv and recordings/{train,test,validate}/
