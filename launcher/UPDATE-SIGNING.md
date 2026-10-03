# 更新通道状态

源码和预览版在 [LeoSasion/DFL-PT-WEBUI](https://github.com/LeoSasion/DFL-PT-WEBUI) 与 GitHub Releases 分发。用户通过源码安装入口或便携包安装。

启动器自动克隆、Git 更新和在线自更新保持关闭，`update-channel.json` 为 `enabled=false`。当前发行使用 SHA-256 文件及逐文件清单供校验，没有声称 Windows Authenticode 或签名自动更新通道已经启用。若以后启用更新，需要为本产品配置地址和签名材料，不能沿用旧项目的通道。
