# Attention-Capture Modal Asset Audit

## Verdict

- Retention labels maybe available: False
- External dataset dirs maybe available: False
- Feature caches maybe available: True
- Full multimodal token env present: False
- Claim boundary: This Modal CPU audit checks remote asset availability only. It does not score TRIBE features or validate attentional capture.

## Configuration

- Volumes checked: 20
- Secrets checked: 19

## Blocking Reasons

- no Modal-hosted SnapUGC/VQualA retention label candidate found
- no Modal secret exposes a HuggingFace token env name

## Secret Presence

- Secrets checked: underlying-analyzer-env, fr-dev-internal-api, fr-dev-github-app, fr-dev-llm-api-keys, fr-prd-internal-api, fr-prd-github-app, fr-prd-llm-api-keys, fr-stg-internal-api, fr-stg-github-app, fr-stg-llm-api-keys, flytrap-review-prod-internal-api, flytrap-review-prod-github-app, flytrap-review-prod-llm-api-keys, flytrap-review-staging-internal-api, flytrap-review-staging-github-app, flytrap-review-staging-llm-api-keys, internal-api, github-app, llm-api-keys
- Token envs checked: HF_TOKEN, HUGGINGFACE_TOKEN, HUGGINGFACE_HUB_TOKEN
- Matching env names: none

## Volume Summary

| volume | entries | files | dirs | truncated | labels | datasets | features |
|---|---:|---:|---:|---|---:|---:|---:|
| rde-activation-results | 4 | 3 | 1 | False | 0 | 0 | 4 |
| audience-analyzer-runs-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| wan22-lora-data-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| wan22-lora-outputs-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| wan22-lora-cache-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| svd-weights-v1 | 58 | 41 | 17 | False | 0 | 0 | 2 |
| svd-outputs-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| wan22-weights-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| wan22-outputs-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| cogvideox-outputs-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| cogvideox-weights-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| tribe-v2-weights-v1 | 143 | 102 | 41 | False | 0 | 0 | 120 |
| vjepa-weights-v1 | 24 | 10 | 14 | False | 0 | 0 | 0 |
| bmd-videos-v1 | 314 | 312 | 2 | False | 0 | 0 | 0 |
| fr-dev-data | 0 | 0 | 0 | False | 0 | 0 | 0 |
| fr-prd-data | 0 | 0 | 0 | False | 0 | 0 | 0 |
| fr-stg-data | 0 | 0 | 0 | False | 0 | 0 | 0 |
| flytrap-review-prod-data | 0 | 0 | 0 | False | 0 | 0 | 0 |
| flytrap-review-data | 0 | 0 | 0 | False | 0 | 0 | 0 |
| tac-docker-data | 0 | 0 | 0 | False | 0 | 0 | 0 |

## Label Candidates

| volume | path | kind | claim blocked |
|---|---|---|---|
| none | n/a | n/a | False |

## Dataset Candidates

| volume | path | kind | claim blocked |
|---|---|---|---|
| none | n/a | n/a | False |

## Feature Candidates

| volume | path | kind | claim blocked |
|---|---|---|---|
| rde-activation-results | activation_geometry | dir | False |
| rde-activation-results | activation_geometry/pythia160_layer3_pocket_scale_sweep_seed20260610_raw.json | file | False |
| rde-activation-results | activation_geometry/pythia160_layer3_pocket_seed20260610_raw.json | file | False |
| rde-activation-results | activation_geometry/pythia70_layer3_strict_opt8_seed20260610_raw.json | file | False |
| svd-weights-v1 | hub/models--stabilityai--stable-video-diffusion-img2vid-xt/snapshots/9e43909513c6714f1bc78bcb44d96e733cd242aa/feature_extractor | dir | False |
| svd-weights-v1 | hub/models--stabilityai--stable-video-diffusion-img2vid-xt/snapshots/9e43909513c6714f1bc78bcb44d96e733cd242aa/feature_extractor/preprocessor_config.json | file | False |
| tribe-v2-weights-v1 | hub | dir | False |
| tribe-v2-weights-v1 | xet | dir | False |
| tribe-v2-weights-v1 | hub/.locks | dir | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3 | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2 | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256 | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0 | dir | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B | dir | False |
| tribe-v2-weights-v1 | hub/CACHEDIR.TAG | file | False |
| tribe-v2-weights-v1 | hub/.locks/models--Systran--faster-whisper-large-v3 | dir | False |
| tribe-v2-weights-v1 | hub/.locks/models--facebook--dinov2-large | dir | False |
| tribe-v2-weights-v1 | hub/.locks/models--facebook--tribev2 | dir | False |
| tribe-v2-weights-v1 | hub/.locks/models--facebook--vjepa2-vitg-fpc64-256 | dir | False |
| tribe-v2-weights-v1 | hub/.locks/models--facebook--w2v-bert-2.0 | dir | False |
| tribe-v2-weights-v1 | hub/.locks/models--meta-llama--Llama-3.2-3B | dir | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/blobs | dir | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/refs | dir | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/snapshots | dir | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/blobs/0adcd01e7c237205d593b707e66dd5d7bc785d2d | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/blobs/3a5e2ba63acdcac9a19ba56cf9bd27f185bfff61 | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/blobs/69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1 | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/blobs/75336feae814999bae6ccccdecf177639ffc6f9d | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/blobs/931c77a740890c46365c7ae0c9d350ba3cca908f | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/blobs/a6344aac8c09253b3b630fb776ae94478aa0275b | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/blobs/a84bfa7f20cac02ea5a99efa5eaf687ad58c1caf | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/refs/main | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478 | dir | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478/.gitattributes | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478/README.md | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478/config.json | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478/model.bin | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478/preprocessor_config.json | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478/tokenizer.json | file | False |
| tribe-v2-weights-v1 | hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478/vocabulary.json | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/blobs | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/snapshots | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/blobs/399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/blobs/3d7671e166807a113f7f8d3e1b79ebf0c32bc85d | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/blobs/5def28dd2a0ecb2561a4649703f13c4dc595ee95 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/blobs/8320e4778a7f8850d10f30d97e9138438e1851af1576fea789c43746140cc655 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/blobs/a6344aac8c09253b3b630fb776ae94478aa0275b | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/blobs/ff5b47c2edcd1d3556d63c01a65d93b58b9efce1 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/.gitattributes | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/README.md | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/config.json | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/model.safetensors | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/preprocessor_config.json | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/pytorch_model.bin | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/blobs | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/snapshots | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/blobs/428e59536b10170c4e7412b803db164f0002e4c6 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/blobs/61e6a0bb4183d357bcf27697a34a8a225238cf24 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/blobs/767ba38f2fa96b47382a1b6ad84df160cabcfc12 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/blobs/9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/blobs/a6344aac8c09253b3b630fb776ae94478aa0275b | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f/.gitattributes | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f/LICENSE | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f/README.md | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f/best.ckpt | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f/config.yaml | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/blobs | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/refs | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/snapshots | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/blobs/3534852408cef7f5c0c54dfed6e0842c24492863 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/blobs/67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/blobs/6f4f3743f7c85d064d65519a905c1e04ebae69f8 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/blobs/8be436f25057b709d28aa3a4993614d6f319bb0f | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/blobs/a6344aac8c09253b3b630fb776ae94478aa0275b | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/blobs/b1b39c32857c5141adde7fc076100feaddefdf0d | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/blobs/c5dd15b27643a7fd207aa6f59373f255036b9a8b | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/blobs/f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/refs/main | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/snapshots/875c192b7b704b87d1e1d99345769632dd5f739a | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/snapshots/875c192b7b704b87d1e1d99345769632dd5f739a/original | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/snapshots/875c192b7b704b87d1e1d99345769632dd5f739a/.gitattributes | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/snapshots/875c192b7b704b87d1e1d99345769632dd5f739a/README.md | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/snapshots/875c192b7b704b87d1e1d99345769632dd5f739a/config.json | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/snapshots/875c192b7b704b87d1e1d99345769632dd5f739a/model.safetensors | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/snapshots/875c192b7b704b87d1e1d99345769632dd5f739a/notebook.ipynb | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/snapshots/875c192b7b704b87d1e1d99345769632dd5f739a/notebook_finetuning.ipynb | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/snapshots/875c192b7b704b87d1e1d99345769632dd5f739a/video_preprocessor_config.json | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--vjepa2-vitg-fpc64-256/snapshots/875c192b7b704b87d1e1d99345769632dd5f739a/original/model.pth | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/blobs | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/refs | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/snapshots | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/blobs/23115b6e9bf0776b4cc3d480fde0a90a2ff33011 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/blobs/5db61951cdf5edab6337fd84ee619500c27aaa3d | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/blobs/8310b4270a5b499e92e20c859892dbf7429619347debb5f8feba79eb88f99b4f | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/blobs/a383a594dac18459628cd2837168cd276342a31a | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/blobs/a6344aac8c09253b3b630fb776ae94478aa0275b | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/blobs/eb890c9660ed6e3414b6812e27257b8ce5454365d5490d3ad581ea60b93be043 | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/refs/main | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b | dir | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b/.gitattributes | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b/README.md | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b/config.json | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b/conformer_shaw.pt | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b/model.safetensors | file | False |
| tribe-v2-weights-v1 | hub/models--facebook--w2v-bert-2.0/snapshots/da985ba0987f70aaeb84a80f2851cfac8c697a7b/preprocessor_config.json | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs | dir | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/snapshots | dir | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/085b47c1575cb889b7024030e60b78f54f0b8c9e | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/2d73a6863086ff9d491c28e49df9fb697cd92c2b | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/4719a04514ec2f060240711b7c33ab21187cac730ecaba3040b7a0fd95a9cefb | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/47d4a5aa69cdef91a53b77f5c5583647a578ca0e | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/584d8d3e3f82f7964955174dfe5e3b1cf117a9d859f022cfdf7fcb884856e002 | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/5cc5f00a5b203e90a27a3bd60d1ec393b07971e8 | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/7f70e0141c2257e77489bb5359023477972e2e00 | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55 | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/8cfc88c52654b88fabf74e4777dd729540d88edd | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/a6344aac8c09253b3b630fb776ae94478aa0275b | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/ac3c5f21b9779e3da0677d6d3c587778fe3a331e | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/cb9ec25536e44d86778b10509d3e5bdca459a5cf | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/cfabacc2620186cd3dd4b1dde9a37e057208636e | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/d3a1f0f5f401eeadca0c7a6786bd9e877fd42e58 | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/blobs/e85237a45033593adb3eb7e1a6cc5c410c4e2b360e24a422d3c1ab12a166c441 | file | False |
| tribe-v2-weights-v1 | hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062 | dir | False |
