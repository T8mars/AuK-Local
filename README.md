# AuK Local

[中文说明](README_zh.md) · [Releases](https://github.com/T8mars/AuK-Local/releases) · [Models](https://huggingface.co/t8star/Auk-Comfy) · [Native ComfyUI nodes](https://github.com/T8mars/Comfyui-Auk-T8)

AuK Local is a standalone, light-themed Windows workstation for speech generation, editing, enhancement, and separation with [Tencent Hunyuan AuK](https://github.com/Tencent-Hunyuan/AuK). This repository is maintained by T8star-Aix and is independent from the native ComfyUI node package.

## Start

Download the **complete Windows package (models and Python included)** from [Hugging Face — AuK-Local.rar](https://huggingface.co/t8star/Auk-Comfy/resolve/main/AuK-Local.rar?download=true). This overseas mirror is 20.37 GB (18.97 GiB). Fully extract the archive, then double-click `AuK-Local.exe` inside the `AuK-Local` folder. If your download is interrupted, resume it with a download manager.

SHA-256: `a99966c3eb7336ca9d027c0ea491edaddcfa3582b2eb436444a36cfc2f2e9061`

Use a complete AuK Local folder and double-click `AuK-Local.exe`. The launcher uses the Python, CUDA helpers, FFmpeg, and models inside that folder. It opens `http://127.0.0.1:7860` after the local service is ready and keeps the console visible for logs.

The GitHub Release ZIP is a **code-only update package**. It intentionally excludes model weights and the Python runtime, so it must be copied over an existing complete package. The complete package is distributed separately by the maintainer.

## Automatic updates

The update panel is shown directly below the model status at the top of the app:

- **Check for updates** reads the latest release only from `T8mars/AuK-Local`.
- **Update and restart** downloads the signed code archive, validates the RSA signature, validates the archive SHA-256, and validates every file before staging it.
- The service refuses to update while a task is queued or running.
- The external updater backs up replaced files, installs the staged version, launches it, and checks `/api/v1/health`. A failed copy, final hash check, or startup health check restores the previous program.

The updater has a fixed allowlist and cannot overwrite these package areas:

```text
models/   runtime/   ckpts/   data/   outputs/   logs/   .secrets/   .git/
```

Start the app through `AuK-Local.exe` to use automatic restart. The BAT files remain compatibility and diagnostics entry points.

If an existing 0.2.2 installation stages the update but does not restart, manually overlay the 0.2.3 code-only ZIP once. The 0.2.2 Windows launcher script can fail before handing the staged package to the external updater; the repaired flow in 0.2.3 has passed an end-to-end update and restart test.

## Included interfaces

- Light Chinese Gradio workspace on `127.0.0.1:7860`
- Local FastAPI service with a per-install session token
- Queued jobs, cancellation, retry, history, audio output, and JSON metadata
- Clickable history details with playable generated and source/reference audio, prompts, final model instruction, and full parameters
- AuK Base by default for editing quality, with AuK-Flash available for fast previews
- Models stay resident after a successful task by default; the UI also provides release-after-task and manual VRAM release controls
- Automatic TTS duration to prevent short text from continuing into an internal conditioning marker
- Official per-task duration rules derived from the audio actually submitted after trimming
- Random Seed draw mode by default, with fixed Seed available for reproducibility
- Live Chinese phase and elapsed-time updates while a task is running
- A task-specific usage card with official input constraints and a copyable example for all 17 UI entries, covering all 16 upstream low-level tasks
- Model diagnostics and resumable fixed-revision model downloads
- Signed code updates with backup and rollback

The native ComfyUI package is maintained in [Comfyui-Auk-T8](https://github.com/T8mars/Comfyui-Auk-T8). It loads models directly in the ComfyUI process and does not call the AuK Local 7860 service.

## Task input guide

Selecting a task updates the on-page guide with its audio requirement, allowed values, official input shape, example, and task-specific warning. Speech content edits and lyric edits accept one edit per run. Reference voice cloning accepts only the new target text alongside a clean reference recording; a reference transcript or voice description is not sent to the model.

Lyric editing requires a clean isolated solo vocal (a cappella), without accompaniment. Editing tasks use the bundled Silero VAD and the unpadded speech interval for duration, then add 0.1 seconds only around the model input. Whisper conversion uses the official -44.47 LUFS target; lyric and vocal extraction outputs use the official -14 LUFS downward limiter and 0.95 peak ceiling. Metadata records the actual VAD and loudness path.

Pitch, volume, timbre, de-accent, whisper, enhancement, quality repair, and separation match the actual submitted audio. Emotion follows AuK's official duration coefficients: 1.22× for sad, 1.16× for fearful, and 1.06× for the other supported emotions. Speech and lyric edits estimate their duration from the text added or removed. Nonverbal edits add or remove the official event duration.

Speed editing always locks the duration control and calculates the target as `source duration / speed multiplier`. A manual target is never used for this task, because extra output time can make the model fill the tail with damaged audio.

The browser supports two applied trimming paths. For waveform trimming, drag a range, click the scissors, and then click the **Trim** confirmation shown by the player. Alternatively, enter the trim start and end and click **Apply trim**. Both paths replace the current input and immediately update the submitted duration. The native **↶** button beside the scissors restores the original upload without uploading it again. Source/reference input and generated output are each limited to 30 seconds independently; they are not added together, so a 30-second input may produce a 30-second output. Emotion, de-accenting, and whisper conversion require ordinary spoken speech; singing and already-standard speech are unsuitable validation sources for those tasks.

History keeps the latest 50 tasks. Clicking a row restores the generated result, source/reference audio when present, entered text, final model instruction, and complete request/metadata JSON. Missing or damaged history files are reported instead of being shown as playable.

If Chrome blocks its download link under a security policy, use **Save current result to Downloads** or select a history row and use **Save selected history result to Downloads**. These buttons copy the existing local WAV to the Windows Downloads folder and verify SHA-256 without overwriting an existing file. **Open output folder** also gives direct access to all original results.

## Release contents

`AuK-Local-vX.Y.Z-code-only.zip` contains the launcher, Python source, Windows scripts, UI resources, documentation, and public update key. Release assets also contain:

- `auk-local-release.json`: RSA-SHA256 signed update manifest
- `SHA256SUMS.txt`: hashes for manual verification

The archive build fails if it finds models, runtime files, checkpoints, user data, outputs, logs, or the private signing key.

## Links

- Bilibili: [T8star-Aix](https://space.bilibili.com/385085361)
- YouTube: [@T8star-Aix](https://www.youtube.com/@T8star-Aix/)
- Online AI apps: [RunningHub](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- API: [Seedance API](https://api.seedance.nz/sign-up?aff=5f4w)
- Complete AuK Local package: [Hugging Face / overseas download](https://huggingface.co/t8star/Auk-Comfy/resolve/main/AuK-Local.rar?download=true) · [Quark Drive](https://pan.quark.cn/s/264edb7e36bd)
- Hugging Face: [t8star](https://huggingface.co/t8star)

## Credits and license

The model and core inference implementation come from [Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK). Please cite and credit the upstream project when using its model or research. This repository retains the upstream MIT license in [LICENSE](LICENSE).
