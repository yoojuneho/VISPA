# VISPA Reproducibility Code

This directory contains the code used for the VISPA multimodal privacy leakage experiments. It is a cleaned subset of the working repository: dataset-scale generated images, model weights, API keys, local artifact paths, and unrelated exploratory scripts are intentionally excluded.

## 📁 Layout

- `image_generation/`: SUN397 background filtering, prompt generation, Qwen-Image-Edit synthesis, and VLM validation scripts for exposed image construction.
- `ci_seed/`: direct VISPA CI seed construction, user-instruction generation, and seed auditing.
- `main_experiment/`: action-based clean-versus-exposed image-pair evaluation with vLLM or OpenAI-compatible APIs.
- `privacy_risk/`: privacy-awareness risk scoring for exposed images.
- `make_blur.py`: standalone motion-blur generator for clean images or CI seed JSON files.

## ⚙️ Setup

Install the lightweight dependencies first. Model-serving dependencies such as `vllm`, `torch`, `diffusers`, and `transformers` depend on the target CUDA environment and should be installed following the server environment used for the experiments.

```bash
pip install -r requirements.txt
```

Set API keys only through environment variables. Do not hard-code keys in scripts or notebooks.

```bash
export OPENAI_API_KEY="..."
```

All commands below assume the current directory is this `Code` directory unless a section explicitly changes directories.

## 🖼️ 1. Image Generation

The image-generation pipeline starts by applying the SUN397 class allow/exclude list, then runs VLM background screening, prompt generation, synthesis, and validation.

```bash
cd image_generation
python scripts/filter_sun397_backgrounds.py --dataset-dir /path/to/SUN397 --included-class-path config/included_class.txt --exception-class-path config/exception_class.txt --limit 10
python scripts/filter_sun397_classes.py --input-dir outputs/sun397_vlm_filter/included_images --exception-class-path config/exception_class.txt
python scripts/generate_compositing_prompts.py --limit 10
python scripts/synthesize_exposed_images.py --limit 10
python scripts/validate_exposed_images.py --limit 10
```

Use `--dry-run` with `filter_sun397_backgrounds.py` only to inspect candidate inputs without producing `included_images`. `run_image_pipeline.py` is the multi-stage GPU runner used for full generation. Adjust `--face-dir`, `--mapping`, `--model-dir`, `--gpu-ids`, and `--vlm-ports` for the local machine.

## 🧩 2. CI Seed Construction

The CI seed pipeline is direct-only. It starts from the direct image mapping, extracts scene descriptions for clean backgrounds, generates four privacy-condition variants per exposed image, and then generates natural user instructions.

```bash
cd ci_seed
python describe_scenes.py --mapping /path/to/mapping.json
python build_ci_seed.py --mapping /path/to/mapping.json
python generate_user_instructions.py DirectLeakage_Seed.json --skip-existing
python audit_seed.py --seed DirectLeakage_Seed.json --mapping /path/to/mapping.json
```

For shell usage:

```bash
bash run_pipeline.sh
```

## 🌫️ 3. Motion Blur

Generate blur variants for a single clean image:

```bash
python make_blur.py --input path/to/clean.png --output output/clean_blur30.png --shake-level 30
```

Generate blur variants and update a CI seed JSON:

```bash
cp path/to/DirectLeakage_Seed.json path/to/DirectLeakage_Seed_blur.json
python make_blur.py --batch-from-seed-jsons --seed-jsons path/to/DirectLeakage_Seed_blur.json --motion-blur-dir MotionBlur
```

Batch mode updates each seed JSON in place and writes `MotionBlur/{30,50,70,90}/...` paths.

## 🧪 4. Main Action Experiment

Build image-pair instances from the direct leakage seed:

```bash
cd main_experiment
python build_instances.py --direct-seed /path/to/DirectLeakage_Seed_blur.json --image-root /path/to/images
```

Run a vLLM-served model:

```bash
python run.py --mode vllm --model Qwen3-VL-8B-Instruct --host localhost --port 9000 --image-root /path/to/images --blur all
```

Run the privacy-reminder variant:

```bash
python run_privacy_reminder.py --mode vllm --model Qwen3-VL-8B-Instruct --host localhost --port 9000 --image-root /path/to/images --blur all
```

## 📊 5. Privacy-Risk Scoring

Score an image directory with an OpenAI-compatible VLM endpoint:

```bash
cd privacy_risk
python run_privacy_risk.py --image-dir /path/to/exposed_images --model Qwen3-VL-8B-Instruct --base-url http://127.0.0.1:9000/v1 --limit 5
python analyze_privacy_risk.py output/privacy_risk/privacy_risk_results.jsonl
```

For a model list, keep the target model names in `privacy_risk/models/test_model_list.txt` and run:

```bash
python run_model_sweep.py --image-dir /path/to/exposed_images --base-url http://127.0.0.1:9000/v1 --limit 5
```

## 📝 Notes

- The full SUN397 dataset, MLLMU face images, generated exposed images, blur outputs, and VLM model weights are not included.
- All public prompts in this directory are English and scoped to the experiments described in the paper.
