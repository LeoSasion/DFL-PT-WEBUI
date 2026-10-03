# DFL-PT-WEBUI launcher validation

The CPython 3.7 / TensorFlow wheelhouse, CUDA/cuDNN payload downloader, BITS and legacy runtime bootstrap tests were retired with those implementations. Their historical pass counts do not apply to this repository.

pt-runtime.Tests.ps1 validates the actual local Python 3.12 / torch 2.9.1 cu128 manifest, rejects old schemas and refuses missing required FFmpeg. repository-identity.Tests.ps1 rejects all remote URLs while online installation is disabled. project-locator.Tests.ps1 uses the ME entry point as part of the new project identity. Other retained tests cover generic launcher installation, logs, process execution, dependency repair, payload safety, and workspace handling.

Use Windows PowerShell's Pester for launcher/tests, and the project Node to run launcher/server/tests/*.test.mjs. Native launcher builds are verified separately with build-host.ps1 -NoDownload after preparing the pinned Microsoft WebView2 SDK and launcher UI dependencies.
