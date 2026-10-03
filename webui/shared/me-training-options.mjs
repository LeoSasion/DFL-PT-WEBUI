// Keep these public form descriptors aligned with me_backend.config.MEConfig.
export const ME_CONFIG_GROUPS = [
  { id: "architecture", label: "网络结构", help: "保存后不可更改；续训沿用检查点结构。" },
  { id: "optimization", label: "优化与训练", help: "续训修改配置需要明确启用允许配置变更。" },
  { id: "memory", label: "显存与精度", help: "RG 用计算换显存；FP16 需要 CUDA。" },
  { id: "loss", label: "重建与风格损失" },
  { id: "adversarial", label: "GAN 与 TrueFace", help: "TrueFace 仅支持 DF；预训练阶段关闭对抗和风格损失。" },
  { id: "data", label: "采样与增强" },
  { id: "replay", label: "困难样本回放" },
];

const number = (id, label, group, value, min, max, extras = {}) => ({
  id, configKey: id, label, group, type: "number", default: value, min, max, advanced: true, ...extras,
});
const boolean = (id, label, group, value = false, help = "") => ({
  id, configKey: id, label, group, type: "boolean", default: value, advanced: true, help,
});
const select = (id, label, group, value, options, extras = {}) => ({
  id, configKey: id, label, group, type: "select", default: value,
  options: options.map(option => typeof option === "object" ? option : { value: option, label: String(option) }),
  advanced: true, ...extras,
});

// Architecture suffixes are explicit options, not separate model types.
const architectureOptions = ["liae", "df"].flatMap(family => Array.from({ length: 16 }, (_, bits) => {
  const suffix = ["u", "d", "t", "c"].filter((_, index) => bits & (1 << index)).join("");
  return suffix ? `${family}-${suffix}` : family;
}));

export const ME_CONFIG_PARAMETERS = [
  select("archi", "架构", "architecture", "liae-ud", architectureOptions, { structural: true }),
  select("resolution", "模型分辨率", "architecture", 128,
    Array.from({ length: 19 }, (_, index) => ({ value: 64 + 32 * index, label: `${64 + 32 * index} px` })),
    { structural: true, advanced: false }),
  number("ae_dims", "编码维度", "architecture", 256, 32, 1024, { integer: true, structural: true }),
  number("e_dims", "编码器维度", "architecture", 64, 16, 256, { integer: true, structural: true }),
  number("d_dims", "解码器维度", "architecture", 64, 16, 256, { integer: true, structural: true }),
  number("d_mask_dims", "遮罩维度", "architecture", 22, 16, 256, { integer: true, structural: true }),
  select("face_type", "人脸范围", "architecture", "f", [
    { value: "h", label: "半脸" }, { value: "mf", label: "中脸" }, { value: "f", label: "全脸" },
    { value: "wf", label: "整脸" }, { value: "head", label: "头部" },
  ], { structural: true }),
  number("batchSize", "Batch size", "optimization", 4, 1, 256,
    { integer: true, configKey: "batch_size", advanced: false }),
  number("lr", "学习率", "optimization", 5e-5, 1e-8, 1e-2, { step: 0.00001 }),
  boolean("adabelief", "使用 AdaBelief", "optimization", true, "关闭时使用 RMSprop；切换优化器需重置优化器。"),
  select("lr_dropout", "学习率 Dropout", "optimization", "n", [
    { value: "n", label: "关闭" }, { value: "y", label: "设备端" }, { value: "cpu", label: "CPU 端" },
  ]),
  boolean("clipgrad", "梯度裁剪", "optimization"),
  boolean("pretrain", "预训练阶段", "optimization", false, "使用专用预训练人脸集；进入或退出此阶段应重置数据状态。"),
  boolean("use_rg", "RG 梯度重计算", "memory", false, "显存不足时启用；受控单卡短基准节省约 14% 峰值显存，但单步慢约 32%。"),
  boolean("use_fp16", "FP16 混合精度", "memory", false, "仅 CUDA 可用；受控单卡短基准单步快约 15%、显存少约 9%，主权重和优化器仍为 FP32。"),
  boolean("optimizer_on_cpu", "优化器放在 CPU", "memory", false, "仅显存仍不足时启用；受控单卡短基准峰值显存少约 36%，但单步耗时约为 6.4 倍。"),
  boolean("masked_training", "遮罩内训练", "loss", true),
  boolean("eyes_prio", "眼部优先", "loss"),
  boolean("mouth_prio", "嘴部优先", "loss"),
  select("loss_function", "重建损失", "loss", "SSIM", ["SSIM", "MS-SSIM", "MS-SSIM+L1"]),
  number("background_power", "背景重建权重", "loss", 0, 0, 1, { step: 0.01 }),
  number("face_style_power", "人脸风格权重", "loss", 0, 0, 100, { step: 0.1 }),
  number("bg_style_power", "背景风格权重", "loss", 0, 0, 100, { step: 0.1 }),
  boolean("blur_out_mask", "模糊遮罩外区域", "loss"),
  number("gan_power", "GAN 权重", "adversarial", 0, 0, 10, { step: 0.01 }),
  number("gan_patch_size", "GAN Patch 尺寸", "adversarial", 16, 3, 640, { integer: true }),
  number("gan_dims", "GAN 维度", "adversarial", 16, 4, 128, { integer: true }),
  number("gan_smoothing", "GAN 标签平滑", "adversarial", 0.1, 0, 0.5, { step: 0.01 }),
  number("gan_noise", "GAN 标签噪声", "adversarial", 0, 0, 0.5, { step: 0.01 }),
  number("true_face_power", "TrueFace 权重", "adversarial", 0, 0, 1, { step: 0.01 }),
  boolean("random_warp", "随机变形", "data", true, "LIAE 关闭随机变形时冻结 inter_AB，适合后期微调。"),
  boolean("random_src_flip", "SRC 随机翻转", "data"),
  boolean("random_dst_flip", "DST 随机翻转", "data", true),
  number("random_hsv_power", "HSV 扰动强度", "data", 0, 0, 0.3, { step: 0.01 }),
  boolean("uniform_yaw", "均匀姿态采样", "data"),
  boolean("random_downsample", "随机降采样", "data"),
  boolean("random_noise", "随机噪声", "data"),
  boolean("random_blur", "随机模糊", "data"),
  boolean("random_jpeg", "随机 JPEG 压缩", "data"),
  boolean("random_color", "随机颜色增强", "data"),
  select("ct_mode", "颜色迁移", "data", "none", ["none", "rct", "lct", "mkl", "idt", "sot", "mix", "fs-aug", "cc-aug"]),
  number("data_workers", "每侧采样进程数", "data", 0, 0, 32, { integer: true,
    help: "0 在训练进程采样；SRC、DST 各使用此数量的工作进程。" }),
  boolean("retraining_samples", "困难样本回放", "replay"),
  number("retraining_capacity", "回放容量", "replay", 128, 1, 2048, { integer: true }),
  number("retraining_every", "回放间隔", "replay", 16, 1, 10000, { integer: true,
    help: "每隔若干正常迭代执行一次额外优化器更新。" }),
];

export const ME_DEFAULT_CONFIG = Object.freeze(Object.fromEntries(ME_CONFIG_PARAMETERS.map(item => [item.configKey, item.default])));
export function configFromParameters(parameters = {}, base = ME_DEFAULT_CONFIG) {
  const config = { ...ME_DEFAULT_CONFIG, ...base };
  for (const parameter of ME_CONFIG_PARAMETERS) {
    if (Object.hasOwn(parameters, parameter.id)) config[parameter.configKey] = parameters[parameter.id];
  }
  return config;
}
export function parametersFromConfig(config = {}) {
  const complete = { ...ME_DEFAULT_CONFIG, ...config };
  return Object.fromEntries(ME_CONFIG_PARAMETERS.map(item => [item.id, complete[item.configKey]]));
}

export function validateMeConfig(config) {
  const unknown = Object.keys(config).filter(key => !Object.hasOwn(ME_DEFAULT_CONFIG, key));
  if (unknown.length) throw new Error(`不支持的 ME 配置：${unknown.join(", ")}`);
  for (const parameter of ME_CONFIG_PARAMETERS) {
    const value = config[parameter.configKey];
    if (parameter.type === "boolean" && typeof value !== "boolean") throw new Error(`${parameter.label} 必须是布尔值`);
    if (parameter.type === "number" && (typeof value !== "number" || !Number.isFinite(value)
      || (parameter.integer && !Number.isInteger(value)) || value < parameter.min || value > parameter.max)) {
      throw new Error(`${parameter.label} 超出范围或格式不合法`);
    }
    if (parameter.type === "select" && !parameter.options.some(option => option.value === value)
      && !(parameter.configKey === "archi" && /^(df|liae)(-[udtc]+)?$/.test(value)
        && new Set(value.split("-")[1] ?? "").size === (value.split("-")[1] ?? "").length)) {
      throw new Error(`${parameter.label} 的选项不受支持`);
    }
  }
  if (config.true_face_power > 0 && !config.archi.startsWith("df")) throw new Error("TrueFace 仅支持 DF 架构");
  if (config.gan_patch_size > config.resolution) throw new Error("GAN Patch 尺寸不能大于模型分辨率");
  if (config.retraining_samples && config.retraining_capacity < config.batch_size) throw new Error("回放容量必须至少容纳一个 batch");
  return config;
}

export function getMePreset(name) {
  const presets = {
    default: { ...ME_DEFAULT_CONFIG },
    "rg-fp16": { use_rg: true, use_fp16: true, optimizer_on_cpu: false, data_workers: 2 },
    pretrain: { pretrain: true, uniform_yaw: true, use_rg: true, use_fp16: true, optimizer_on_cpu: false, data_workers: 2 },
    finetune: { pretrain: false, lr: 1e-5, random_warp: false, eyes_prio: true, mouth_prio: true,
      gan_power: 0.1, optimizer_on_cpu: false, lr_dropout: "cpu", retraining_samples: true },
  };
  const config = presets[name];
  if (!config) throw new Error(`未知 ME 预设：${name}`);
  return Object.fromEntries(ME_CONFIG_PARAMETERS.filter(item => Object.hasOwn(config, item.configKey)).map(item => [item.id, config[item.configKey]]));
}
