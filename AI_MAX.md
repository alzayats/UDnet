# UDnet V2 + AI Max (experimental)

AI Max adds a generative detail-restoration stage after UDnet V2. It produces the most detailed
results shown in the [gallery](https://alzayats.github.io/UDnet/), at a substantially higher
computational and installation cost than the other options. For this reason it is kept on this
separate branch, so that the main branch keeps a light installation.

## Method

1. **Colour correction.** UDnet V2 removes the colour cast and haze.
2. **Detail restoration.** [DiffBIR v2.1](https://github.com/XPixelGroup/DiffBIR) (Lin et al., ECCV 2024),
   a blind image restoration method built on a Stable Diffusion 2.1 prior, restores fine detail and
   upscales the V2 output by 2×. It is guided by a fixed text prompt describing underwater scenes.
3. **Colour and texture constraint.** Diffusion models tend to shift colour and to synthesise
   texture in flat regions. The final image therefore combines:
   - the low frequencies (colour and tone) of the V2 output, upscaled bicubically;
   - the high frequencies of the DiffBIR output, but only in regions where the V2 output already
     contains texture, measured by local luminance variance. Flat regions such as open water keep
     the detail of the V2 output.

## Intended use

AI Max is intended for visual inspection, presentation and sharing. It is **not** suitable for
quantitative analysis, measurement, or any application in which image content must be faithful to
what the camera recorded, for example species identification, counting or morphometrics. Use
UDnet V1 or V2 for such purposes.

## Requirements

| | Tested | Notes |
|---|---|---|
| GPU | NVIDIA RTX 4090 (24 GB) | About 10 GB of GPU memory is used with tiled inference. A CPU is not supported. |
| Software | Ubuntu 22.04, conda, CUDA 11.8 PyTorch wheels | Windows and macOS have not been tested. |
| Disk | about 15 GB | About 8 GB for the conda environment and 6.7 GB for model weights. |
| Network | first run only | DiffBIR downloads its weights from Hugging Face on first use. |

## Installation

```bash
git clone -b ai-max https://github.com/alzayats/UDnet.git
cd UDnet
pip install -r requirements.txt     # UDnet itself
bash setup_ai_max.sh                # DiffBIR and its own conda environment, 'udnet-aimax'
```

`setup_ai_max.sh` clones DiffBIR at the tested commit into `third_party/DiffBIR` and installs its
pinned dependencies into a separate conda environment, so the UDnet environment is not modified.

## Usage

```bash
python ai_max.py --input path/to/images --output results_ai_max/
```

| Option | Default | |
|---|---|---|
| `--steps` | `10` | diffusion sampling steps; more steps are slower and may add more detail |
| `--cfg` | `4.0` | prompt guidance scale |
| `--upscale` | `2` | output scale |
| `--seed` | `231` | random seed |
| `--diffbir`, `--env` | `third_party/DiffBIR`, `udnet-aimax` | locations created by `setup_ai_max.sh` |

## Runtime

On an RTX 4090, processing takes about 25 s per image for inputs of roughly 1200 × 700 pixels
(2× output), excluding the one-time weight download. This is several hundred times slower than
UDnet V2 alone, so AI Max is not practical for video.

## Known limitations and potential difficulties

- **Synthesised detail.** The diffusion prior can produce plausible structures that were not
  captured, for example spines, scales or seabed texture. The texture constraint suppresses most
  artefacts in flat regions, but it cannot distinguish restored detail from synthesised detail in
  textured regions.
- **Dependency pinning.** DiffBIR requires PyTorch 2.2.2 with CUDA 11.8 and xformers 0.0.25. These
  builds may not support very recent GPU architectures or drivers. In that case the DiffBIR
  environment must be adapted manually.
- **Memory.** GPUs with less than about 12 GB of memory may run out of memory on large images.
  DiffBIR's tiling options reduce this at the cost of speed.
- **Reproducibility.** Results depend on the seed and may differ slightly between GPUs and driver
  versions.
- **Third-party code.** AI Max relies on an external repository pinned to a specific commit. Later
  changes to that repository or to its hosted weights may require updates to this branch.

## Licences

The UDnet code is MIT-licensed. DiffBIR is distributed under the Apache License 2.0. Its diffusion
weights are derived from Stable Diffusion 2.1, which is released under the CreativeML Open RAIL++-M
licence and includes use-based restrictions. Users are responsible for complying with these
licences.

## References

- X. Lin, J. He, Z. Chen, Z. Lyu, B. Dai, F. Yu, Y. Qiao, W. Ouyang and C. Dong. *DiffBIR: Toward
  Blind Image Restoration with Generative Diffusion Prior.* ECCV 2024.
- R. Rombach, A. Blattmann, D. Lorenz, P. Esser and B. Ommer. *High-Resolution Image Synthesis with
  Latent Diffusion Models.* CVPR 2022.
