# Frozen perceptual metric candidates

Evaluation date: 2026-10-06. The H3CE LPIPS Alex/v0.1 reference is retained;
LPIPS VGG/v0.1 and official DISTS are additional measurement candidates.
All three actually scored the same locked restoration pairs. **No metric
quality winner, human-preference correlation or restoration ranking is
established here.** Independent preference annotations are absent; those
fields remain JSON `null`.

## Interface and loading

`webui/python/vision_metrics.py` exposes
`PerceptualMetricSuite(assets_root, device).evaluate(reference, output, valid_mask)`.
Inputs are equal-size RGB `uint8` images, with at least 32 pixels per side.
The benchmark uses exactly 512×512 inputs for every metric. LPIPS receives
RGB in `[-1,1]`; DISTS receives RGB in `[0,1]`. The original DISTS CLI's
optional resize to a 256-pixel short side is deliberately unused, so these
numbers must not be compared with results using that CLI preprocessing.
These input ranges follow the [official LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity)
and [official DISTS implementation](https://github.com/dingkeyan93/DISTS).

All learned parameters are frozen and run in PyTorch inference mode. Published
backbones and calibration parameters must match full SHA-256 pins before
strict state loading. Architecture construction cannot download pretrained
weights. LPIPS VGG and DISTS share the **same VGG16 convolution objects**;
DISTS retains its official L2 pooling and forward equations, while LPIPS
retains its official max pooling and learned feature calibration. DISTS
source is an unmodified, hash-verified official file loaded from the private
asset cache. Only its private module's VGG factory binding supplies the
already verified backbone; its default `sys.prefix/weights.pt` lookup and
implicit pretrained download are bypassed.

Existing centrally installed dependencies are reused: PyTorch 2.9.1+cu128,
torchvision 0.24.1+cu128 and LPIPS 0.1.4. NumPy is already available. A separate
DISTS package is unnecessary. H3CE application code was read for design
context and was not copied. No training, fine-tuning or optimization occurs.

## Official assets

Optional weights, upstream source, original licenses and per-file
`.provenance.json` records stay in ignored `workspace/.vision-models/metrics`.
Provenance records include the official URL, revision, byte count and SHA-256.
This private evaluation cache is not bundled into releases.

| Asset | SHA-256 |
| --- | --- |
| AlexNet ImageNet backbone `alexnet-owt-7be5be79.pth` | `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` |
| LPIPS Alex v0.1 `alex-v0.1.pth` | `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0` |
| Shared VGG16 ImageNet backbone `vgg16-397923af.pth` | `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0` |
| LPIPS VGG v0.1 `vgg-v0.1.pth` | `a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868` |
| Official DISTS `DISTS_pt.py` | `69b63ebd4e1aece0f79a3b922ba803b5f6fec127babee94869e9eceaa42b4612` |
| Official DISTS calibration `weights.pt` | `f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218` |

LPIPS assets are pinned to official revision
`082bb24f84c091ea94de2867d34c4544f68e0963` with the original BSD-2-Clause
license. DISTS assets are pinned to revision
`1267d8cb626c98706db3697422701c56a85ebf2e` with the original MIT license.
The torchvision BSD-3-Clause license accompanies its official
[VGG16 backbone asset](https://download.pytorch.org/models/vgg16-397923af.pth).
Source repository licenses are preserved; this evaluation does not assert
rights beyond upstream terms for pretrained data or weights.

## Locked pairs and actual results

The existing private restoration protocol selects the first six sources of
the previously frozen twelve-case difficult set before inference. Saved
TUFA landmarks define one FFHQ-style five-point 512 alignment per source.
References are unenhanced aligned originals, **with existing real defects**,
not pristine HQ and not AI-generated targets. One synthetic degradation is
used: 512 reference to 128 with area downsampling. The three restorers are
SwinIR-L x4 PSNR, Real-ESRGAN_x4plus and GFPGANv1.4. Each also has a separate
clean-preservation output from the unchanged 512 reference.

There are 6 sources × 3 restorers × 2 pair types = 36 pairs, scored by all
3 metrics = **108 actual distances**. References, outputs, source-valid masks
and the parent protocol/results must match their recorded hashes. Missing,
duplicate or substituted candidates fail coverage validation. Counts in a
partial report describe only completed pairs, and only a complete report
has `status="complete"`.

Reflected alignment padding is excluded consistently: each prediction's
invalid pixels are replaced by the corresponding reference pixels before
every metric. Convolution receptive fields can still straddle valid-mask
boundaries, and the unchanged background contributes to feature statistics;
this is a shared comparison policy, not an exact masked perceptual estimator.

The following are six-source arithmetic means, in fixed baseline/candidate
order. Each metric has its own calibration; **do not compare values across
columns or use their scales to choose a metric.**

| Restorer | Pair type | LPIPS Alex v0.1 | LPIPS VGG v0.1 | DISTS |
| --- | --- | ---: | ---: | ---: |
| SwinIR-L x4 PSNR | Synthetic down4 | 0.240918 | 0.290203 | 0.165053 |
| SwinIR-L x4 PSNR | Clean preservation | 0.041484 | 0.102197 | 0.063276 |
| Real-ESRGAN_x4plus | Synthetic down4 | 0.166095 | 0.247144 | 0.132995 |
| Real-ESRGAN_x4plus | Clean preservation | 0.049787 | 0.109036 | 0.068937 |
| GFPGANv1.4 | Synthetic down4 | 0.137748 | 0.215566 | 0.104457 |
| GFPGANv1.4 | Clean preservation | 0.126667 | 0.200496 | 0.101914 |

LPIPS Alex reproduced all 36 recorded restoration baseline values with
maximum absolute delta **0.0** in this run. The three metrics give consistent
directions for this small sample's synthetic reconstruction and preservation
tradeoff. That agreement is not validation against human preference: a
lower reference distance does not establish identity, anatomy, occlusion
truth or useful repair of already damaged real originals. No runtime or
throughput term enters a quality decision. Existing restoration quality
formula/results remain separate and are not reweighted using these new
metric candidates.

Private detailed evidence remains in
`workspace/.vision-evaluation/metrics/three-metrics-six-sources-v1.json`.
Original media, image outputs and source paths are not included in this
document or public artifacts.

## Reproduction and checks

```powershell
& ./.venv/Scripts/python.exe -B tools/vision-metric-benchmark.py
& ./.venv/Scripts/python.exe -B tools/tests/test_vision_metrics.py
```

The targeted suite passed **9 tests**, including real official-network
inference, finite changed-pair distances, near-zero identical-pair distances,
repeated deterministic outputs, frozen parameters, first/final shared VGG
parameter identity, automatic-download blockers, invalid input/mask rejection,
tampered source rejection, complete pair coverage and absent-label null fields.
Real-network tests skip explicitly if the optional private weight cache is
missing; they did not skip during this evaluation. Benchmark execution used
FP32 with TF32 disabled. Library warnings about LPIPS's legacy
`pretrained=False` spelling are equivalent to `weights=None` and do not
trigger downloads.

Independent blinded human preferences on held-out pairs are still needed
before reporting rank correlation or choosing a perceptual metric for this
application. This implementation collects reproducible measurements without
manufacturing those labels.
