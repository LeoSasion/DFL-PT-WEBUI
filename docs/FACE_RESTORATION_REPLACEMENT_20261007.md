# 人脸增强替代与整幅推理验证

用户指定旧 FaceEnhancer 由 MambaIRv2 和 Real-ESRGAN 替代。旧 `.py` / `.npy`、懒导出、调用和发行项均移除。Aligned 与 ME 预测脸只允许这两个模型，不能选择 SwinIR，也不静默回退旧链。既有源帧修复属于另一个输入域，其 SwinIR 可选项保留。

## 模型与输入契约

MambaIRv2 使用固定官方 classic-SR Large x4 权重，官方网络与源字节保持原样，仅将 selective scan 导向经官方参考数值验证的纯 PyTorch 实现。它不是官方真实退化修复专用权重，不宣称替代专用真实退化模型的能力。Real-ESRGAN 使用官方 x4plus RRDBNet 权重。两者严格加载固定 SHA，不训练、不自动下载。

两个人脸链均整幅 FP32 推理，关闭 TF32；Real 不使用 128 分块。适配器严格接受有限 BGR float32 `[0,1]`，或明确的 `is_tanh=True` 域，再转换 RGB uint8。模型原生四倍输出按面积采样恢复要求的尺寸；额外反射窗口 padding 在输出裁掉。新增 CPU 测试使用依赖整幅上下文的网络，验证只调用一次，以及奇数画布的 padding 与裁剪。

## 最终盲评

`workspace/.vision-evaluation/aligned-enhancement-20261007-v2` 使用三张实际 WF 512 aligned 和同一检查点产生的三张实际 128 ME 预测脸。输入 SHA 与检查点 SHA 均不变，没有训练。每个域各自随机 A/B 别名；GPT-6 Astra low 实际查看六张图片，先锁评分，再由根任务揭盲。INPUT 是未经增强的输入，不是身份真值。

锁分文件 SHA256：`bf92270495f8b3de611905fa8d83ab788676e9beb8972f18eb2bf9ec9bb2fbca`。

| 域 | 揭盲 | 几何保持 / 5 | 自然细节 / 5 | 皮肤与伪纹理 / 5 | 整体 / 5 | 结论 |
|---|---|---:|---:|---:|---:|---|
| aligned | A = Real-ESRGAN | 4 | 3.5 | 3.5 | 3.5 | 更平滑，部分细纹减弱 |
| aligned | B = MambaIRv2 | 4.5 | 3.5 | 3 | 3.5 | 更接近输入；整体并列，保真维度略偏 Mamba |
| ME 预测脸 | A = MambaIRv2 | 4.5 | 1.5 | 1.5 | 2 | 保留原有眼鼻嘴结构；原预测模糊及色带仍在 |
| ME 预测脸 | B = Real-ESRGAN | 2.5 | 2 | 2 | 1.5 | 更锐但眼睑、鼻孔及牙齿边界有未经输入支持的重描 |

按几何保真优先，aligned 与预测脸均默认 MambaIRv2，Real-ESRGAN 保留可选，合成增强强度默认 0。这里的相对偏好不证明模型恢复了真实细节。六张图中同域样例很接近，不能当作独立统计样本；没有准确率、身份真值、长片时序或跨身份泛化结论。

此前 v1 使用 Real 的分块策略，并有另一轮主观评分；其图、评分与日志完整保留。本轮采用新整幅推理及保真维度的新锁分，不把旧结果改写为通过，也不把两轮主观分数差异都归因于分块。128 预测脸本身在旧策略下已只需一块。

## 实际流程验收

`tools/verify_face_restoration.py` 的最终证据在 `workspace/.vision-evaluation/merge-quality-20261007-v1/restoration-acceptance-v3/acceptance.json`。两模型各完成完整八图独立 aligned 副本，仅修复选定第二张；未选文件与 sidecar 独立复制，选中像素审查记录失效并需重新复核。每模型另完成八帧真实 ME 预测、原生合成与八张遮罩，audit 全部 complete。所有原 aligned、原帧和隔离检查点 SHA 不变。

最终合成/修复/完整副本回归 48 passed，包括有界旧会话迁移；旧模型 SR>0 会话要求明确选择新模型，SR=0 可保留其余配置。上轮 v2 接口验证及诊断日志保留。此次没有改变训练桥，也没有启动 ME 训练。
