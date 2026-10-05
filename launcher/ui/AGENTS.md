# DFL-PT-WEBUI launcher

Preserve the existing forest-black / emerald workstation shell and official brand mark. This is an independent local PyTorch ME repository. Do not preserve old repository remotes, TensorFlow runtime installation, shared settings directories, or automatic updates. First installation uses the explicitly configured official `LeoSasion/DFL-PT-WEBUI` source ZIP fixed by immutable commit and SHA-256, without requiring Git. Existing projects remain usable and repairable. Source upgrades require user action, an idle project and a retained backup with failure rollback; executable auto-updates stay disabled. Publish the standalone launcher separately and keep it available as root `DFL-PT-WEBUI.exe` in future portable packages.

The local runtime manifest has Node, Python 3.12 / PyTorch 2.9.1 CUDA 12.8, and FFmpeg. CUDA and cuDNN come from the PyTorch wheel. Tool commands must use this repository's fixed registry and project-local runtimes.
