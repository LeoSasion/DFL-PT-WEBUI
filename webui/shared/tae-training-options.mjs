// TAEQI2.1 configuration shared by the API and WebUI.
export const TAE_RESOLUTIONS = Object.freeze([256, 448, 512, 640, 832]);
export const TAE_DEFAULT_CONFIG = Object.freeze({
  resolution: 256, face_type: "wf", inter_dims: 256, hidden_dims: 96,
  decoder_mode: "frozen", decoder_lr_multiplier: .1, precision: "bf16",
  batch_size: 1, learning_rate: 1e-4, warmup_steps: 50, weight_decay: 1e-4,
  grad_clip: 1, random_warp: true, random_flip: true, ct_mode: "none",
  half_resolution_probability: 0, source_balanced: true, training_mask_mode: "face",
  rgb_weight: 1, dssim_weight: 1, edge_weight: .25, eyes_mouth_weight: 1, mask_weight: .1,
});

const number = (configKey, label, group, min, max, integer = false, id = configKey) => ({
  id, configKey, label, group, type: "number", default: TAE_DEFAULT_CONFIG[configKey],
  min, ...(max === null ? {} : { max }), integer, structural: group === "architecture",
});
const boolean = (configKey, label) => ({ id: configKey, configKey, label, group: "augmentation",
  type: "boolean", default: TAE_DEFAULT_CONFIG[configKey] });

export const TAE_CONFIG_PARAMETERS = [
  { id: "resolution", configKey: "resolution", label: "训练分辨率", group: "architecture", structural: true,
    type: "select", default: TAE_DEFAULT_CONFIG.resolution,
    options: TAE_RESOLUTIONS.map(value => ({ value, label: `${value} × ${value}` })),
    help: "目标人脸须有足够的原生像素。分辨率决定 Inter 空间网格，创建后不可更改。" },
  { id: "face_type", configKey: "face_type", label: "人脸类型", group: "architecture", structural: true,
    type: "select", default: "wf", options: [
      { value: "h", label: "半脸" }, { value: "mf", label: "中全脸" }, { value: "f", label: "全脸" },
      { value: "wf", label: "whole face" }, { value: "head", label: "头部" }] },
  number("inter_dims", "Inter 瓶颈维度", "architecture", 4, 1024, true),
  number("hidden_dims", "Inter 特征宽度", "architecture", 4, 1024, true),
  { ...number("batch_size", "Batch size", "optimization", 1, 64, true, "batchSize"), resumeMutable: true,
    help: "续训改变 Batch size 会保留权重、总迭代和采样状态，重置 Adam 与预热并记录新阶段。" },
  { id: "precision", configKey: "precision", label: "训练精度", group: "optimization",
    type: "select", default: "bf16", resumeMutable: true, options: [
      { value: "fp32", label: "全精度 FP32" }, { value: "bf16", label: "BF16 混合精度" }],
    help: "主权重与 Adam 保持 FP32；CPU 使用 FP32。续训切换精度会重置 Adam 与预热，记录新阶段。导出精度单独选择。" },
  { id: "decoder_mode", configKey: "decoder_mode", label: "Decoder 训练范围", group: "optimization",
    type: "select", default: "frozen", resumeMutable: true, options: [
      { value: "frozen", label: "冻结（默认）" }, { value: "tail", label: "仅训练末段" }, { value: "full", label: "训练全部" }],
    help: "Encoder 始终冻结。改变训练范围会保留权重与总迭代，重置 Adam 和预热，开始新阶段。" },
  { ...number("decoder_lr_multiplier", "Decoder 学习率倍率", "optimization", 1e-12, 1), resumeMutable: true,
    help: "相对于 Inter 学习率，默认 0.1；冻结时不更新 decoder。续训修改倍率也会开始新阶段。" },
  { ...number("learning_rate", "学习率", "optimization", 1e-12, null), resumeMutable: true,
    help: "停止后续训可调整基础学习率；保留权重、迭代和采样状态，重置 AdamW 与预热并记录新阶段。" },
  number("warmup_steps", "学习率预热步数", "optimization", 0, null, true),
  number("weight_decay", "权重衰减", "optimization", 0, null),
  number("grad_clip", "梯度裁剪", "optimization", 1e-12, null),
  { ...boolean("random_warp", "随机形变"), resumeMutable: true,
    help: "续训可切换；保留权重、总迭代和采样进度，重置优化器与预热并记录新阶段。" },
  boolean("random_flip", "随机翻转"),
  { id: "ct_mode", configKey: "ct_mode", label: "色彩迁移", group: "augmentation", type: "select",
    default: "none", resumeMutable: true, options: [{ value: "none", label: "关闭" }, { value: "rct", label: "RCT" }, { value: "lct", label: "LCT" }],
    help: "RCT 或 LCT 将 SRC 输入和重建目标的颜色匹配至随机 DST 样本；通用预训练的两路使用同一素材池。续训切换会记录新阶段。" },
  number("half_resolution_probability", "半尺寸输入概率", "augmentation", 0, 1),
  boolean("source_balanced", "按来源均衡采样"),
  { id: "training_mask_mode", configKey: "training_mask_mode", label: "遮罩训练范围", group: "loss", type: "select",
    default: "face", resumeMutable: true, options: [
      { value: "face", label: "开启：原始人脸遮罩" },
      { value: "expanded_feather", label: "外扩羽化：增加脸周区域" },
      { value: "full", label: "关闭：全图重建" },
    ], help: "控制 RGB、结构和边缘重建范围；遮罩头仍学习原人脸区域。续训切换会保留权重与采样状态，并重置优化器和预热。" },
  number("rgb_weight", "RGB 重建权重", "loss", 0, null),
  number("dssim_weight", "结构重建权重", "loss", 0, null),
  number("edge_weight", "边缘权重", "loss", 0, null),
  { ...number("eyes_mouth_weight", "眼嘴区域权重", "loss", 0, null), resumeMutable: true },
  number("mask_weight", "遮罩头监督权重", "loss", 1e-12, null),
];

export function configFromTaeParameters(parameters = {}, base = TAE_DEFAULT_CONFIG) {
  const config = { ...TAE_DEFAULT_CONFIG, ...base };
  for (const item of TAE_CONFIG_PARAMETERS) {
    if (Object.hasOwn(parameters, item.id)) config[item.configKey] = parameters[item.id];
  }
  return config;
}

export function parametersFromTaeConfig(config) {
  return Object.fromEntries(TAE_CONFIG_PARAMETERS.filter(item => Object.hasOwn(config, item.configKey))
    .map(item => [item.id, config[item.configKey]]));
}

export function validateTaeConfig(config) {
  if (!config || typeof config !== "object" || Array.isArray(config)) throw new Error("TAE 配置必须为对象");
  config = { ...TAE_DEFAULT_CONFIG, ...config };
  for (const key of Object.keys(config)) if (!Object.hasOwn(TAE_DEFAULT_CONFIG, key)) throw new Error(`不支持的 TAE 配置：${key}`);
  for (const item of TAE_CONFIG_PARAMETERS) {
    const value = config[item.configKey];
    if (item.type === "boolean" ? typeof value !== "boolean" : item.type === "select"
      ? !item.options.some(option => option.value === value)
      : typeof value !== "number" || !Number.isFinite(value) || value < item.min
        || (item.max !== undefined && value > item.max) || (item.integer && !Number.isSafeInteger(value))) {
      throw new Error(`TAE ${item.label}无效`);
    }
  }
  if (config.rgb_weight + config.dssim_weight + config.edge_weight <= 0) throw new Error("TAE 图像重建监督不能全部关闭");
  return config;
}
