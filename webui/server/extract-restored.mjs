import { spawn } from "node:child_process";
import { lstat, mkdir, readFile, realpath, rename, writeFile } from "node:fs/promises";
import { randomBytes } from "node:crypto";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { PATHS, assertWithin } from "./paths.mjs";
import { RestorationManager } from "./restoration-manager.mjs";

const VALIDATE_FACES = String.raw`
import hashlib,json,sys
from pathlib import Path
import numpy as np
import cv2
directory,manifest,dflroot=map(Path,sys.argv[1:4])
require98=sys.argv[4]=='tufa'
sys.path.insert(0,str(dflroot))
from DFLIMG import DFLJPG
record=json.loads(manifest.read_text(encoding='utf-8'))
mapping={item['sourceName']:item for item in record['outputs']}
files=sorted(directory.glob('*.jpg'))
if not files: raise ValueError('No actual DFL face JPEG produced; empty extraction cannot publish')
results=[]
def points(value,count,label):
    array=np.asarray(value,dtype=np.float64)
    if array.shape!=(count,2) or not np.isfinite(array).all(): raise ValueError('Invalid '+label)
    return array
for file in files:
    if file.is_symlink(): raise ValueError('Linked face output refused')
    image=cv2.imread(str(file))
    face=DFLJPG.load(file)
    if face is None or not face.has_data() or image is None: raise ValueError('Not an actual DFL JPEG: '+file.name)
    legacy=points(face.get_landmarks(),68,'legacy68 aligned points')
    source68=points(face.get_source_landmarks(),68,'source68 points')
    source=face.get_source_filename()
    if source not in mapping: raise ValueError('Extracted source name not in restoration mapping')
    entry=mapping[source]
    metadata=face.get_dict()
    restoration=metadata.get('restoration_provenance') or {}
    if (restoration.get('sourceFilename')!=source or restoration.get('processedFilename')!=entry['name']
        or restoration.get('processedSha256')!=entry['outputSha256']
        or restoration.get('declaredInputSha256')!=entry['inputSha256']):
        raise ValueError('Restoration provenance/hash mismatch')
    provenance=metadata.get('landmark_provenance') or {}
    if provenance.get('processedSourceSha256')!=entry['outputSha256'] or provenance.get('processedCanvasWH')!=[entry['width'],entry['height']]:
        raise ValueError('Processed source geometry/hash mismatch')
    audit=metadata.get('native_landmarks98')
    if not require98:
        if provenance.get('alignmentModel') not in ('fan68','fan3d'): raise ValueError('Expected selected FAN alignment model')
        results.append({'name':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'sourceName':source,
            'inputSha256':entry['inputSha256'],'processedSourceSha256':entry['outputSha256'],'legacyPointCount':68,'nativePointCount':None,
            'alignedSize':[image.shape[1],image.shape[0]]})
        continue
    if not audit or audit.get('model_id')!='tufa98' or audit.get('point_definition',{}).get('name')!='WFLW98':
        raise ValueError('Native TUFA98 audit required before restored-face publication')
    original98=points(audit.get('points_original'),98,'native98 original points')
    aligned98=points(audit.get('points_aligned'),98,'native98 aligned points')
    affine=np.asarray(audit.get('source_to_aligned_affine'),dtype=np.float64)
    if affine.shape!=(2,3) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:,:2]))<1e-12:
        raise ValueError('Invalid native98 affine')
    mapped=original98@affine[:,:2].T+affine[:,2]
    delta=float(np.max(np.abs(mapped-aligned98)))
    if delta>1e-3: raise ValueError('Native98 affine coordinates do not roundtrip')
    if audit.get('aligned_canvas_wh')!=[image.shape[1],image.shape[0]]: raise ValueError('Aligned canvas mismatch')
    legacy_affine=face.get_image_to_face_mat()
    if legacy_affine is not None and not np.allclose(legacy_affine,affine,rtol=0,atol=1e-7):
        raise ValueError('Native98 affine differs from DFL legacy geometry')
    sidecar=Path(str(file)+'.landmarks.json')
    if not sidecar.is_file() or sidecar.is_symlink(): raise ValueError('Native98 sidecar missing')
    paired=json.loads(sidecar.read_text(encoding='utf-8'))
    digest=hashlib.sha256(file.read_bytes()).hexdigest()
    if (paired.get('schemaVersion')!=1 or paired.get('aligned_filename')!=file.name
        or paired.get('aligned_sha256')!=digest or paired.get('source_filename')!=source
        or paired.get('native_landmarks98')!=audit or paired.get('restoration_provenance')!=restoration
        or paired.get('landmark_provenance')!=provenance): raise ValueError('Native98 sidecar not bound to JPEG metadata')
    results.append({'name':file.name,'sha256':digest,'sourceName':source,'inputSha256':entry['inputSha256'],
        'processedSourceSha256':entry['outputSha256'],'legacyPointCount':68,'nativePointCount':98,
        'affineRoundtripMaxPx':delta,'alignedSize':[image.shape[1],image.shape[0]]})
if require98 and len(list(directory.glob('*.jpg.landmarks.json')))!=len(files): raise ValueError('Orphan landmark sidecars')
print(json.dumps({'validatedFaces':len(results),'records':results},allow_nan=False))
`;

function runPython(executable, args, paths, capture = false) {
  return new Promise((resolve, reject) => {
    let stdout = "", stderr = "";
    const child = spawn(executable, args, { cwd: paths.repositoryRoot, env: process.env, stdio: capture ? ["ignore", "pipe", "pipe"] : "inherit", windowsHide: true });
    if (capture) {
      child.stdout.on("data", chunk => { stdout += chunk; if (stdout.length > 2 * 1024 * 1024) child.kill(); });
      child.stderr.on("data", chunk => { stderr = (stderr + chunk).slice(-16000); });
    }
    child.once("error", reject);
    child.once("close", (code, signal) => code === 0 ? resolve(stdout) : reject(new Error(`${capture ? stderr.trim() : "提取未完成"}：${signal ?? code}`)));
  });
}

export async function validateRestoredFaces(staging, sourceMap, paths = PATHS, landmarkModel = "tufa") {
  return JSON.parse(await runPython(paths.python, ["-B", "-c", VALIDATE_FACES, staging, sourceMap, paths.currentDflRoot, landmarkModel], paths, true));
}

export async function extractRestored(side, parameters, { paths = PATHS, manager = new RestorationManager({
  workspaceRoot: paths.workspaceRoot, runtimeRoot: paths.runtimeRoot, projectRoot: paths.repositoryRoot, pythonPath: paths.python,
}), runExtractor = args => runPython(paths.python, args, paths), validateFaces = validateRestoredFaces } = {}) {
  parameters = { detector: "yolo26s-face", landmarkModel: "tufa", faceType: "whole_face", imageSize: 512,
    maxFaces: 1, jpegQuality: 90, gpuIndexes: "0", ...parameters };
  const taskId = parameters.restorationTaskId;
  if (!["src", "dst"].includes(side) || !/^rst-[a-f0-9]{32}$/.test(taskId)) throw new Error("修复副本提取参数无效");
  if (!["tufa", "fan"].includes(parameters.landmarkModel)) throw new Error("修复人脸提取关键点模型无效");
  const task = JSON.parse(await readFile(await manager.safePath(path.join(manager.root, "tasks", `${taskId}.json`)), "utf8"));
  if (task.taskId !== taskId || task.side !== side) throw new Error("修复记录与当前素材侧不一致");
  manager.tasks.set(taskId, task);
  const input = await manager.resolveOutputDirectory(taskId);
  const sourceMap = await manager.safePath(path.join(manager.root, "outputs", taskId, "manifest.json"));
  const workspaceReal = await realpath(paths.workspaceRoot);
  const data = path.join(paths.workspaceRoot, `data_${side}`);
  const info = await lstat(data);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("源帧目录不安全");
  assertWithin(workspaceReal, await realpath(data));
  const parent = path.join(data, "aligned_restored");
  await manager.safePath(parent, { directory: true, create: true });
  const parentInfo = await lstat(parent);
  if (!parentInfo.isDirectory() || parentInfo.isSymbolicLink()) throw new Error("副本人脸目录不安全");
  assertWithin(workspaceReal, await realpath(parent));
  const destination = path.join(parent, taskId);
  if (await lstat(destination).catch(error => { if (error.code === "ENOENT") return null; throw error; })) throw new Error("修复记录已提取，不覆盖已有副本");
  const staging = path.join(parent, `.pending-${taskId}-${randomBytes(6).toString("hex")}`);
  await mkdir(staging);
  const args = [paths.currentMain, "extract", "--input-dir", input, "--output-dir", staging, "--source-map", sourceMap, "--no-output-debug"];
  for (const [key, flag] of [["detector", "--detector"], ["landmarkModel", "--landmark-model"], ["faceType", "--face-type"], ["imageSize", "--image-size"], ["maxFaces", "--max-faces-from-image"], ["jpegQuality", "--jpeg-quality"], ["gpuIndexes", "--force-gpu-idxs"]])
    if (parameters[key] !== "" && parameters[key] != null) args.push(flag, String(parameters[key]));
  if (parameters.cpuOnly) args.push("--cpu-only");
  process.stdout.write("[WEB] 从修复 PNG 重新生成独立人脸数据集；原 aligned 保留。\n");
  try {
  await runExtractor(args);
  await manager.resolveOutputDirectory(taskId);
  await manager.safePath(staging, { directory: true });
  const validation = await validateFaces(staging, sourceMap, paths, parameters.landmarkModel);
  if (!Number.isInteger(validation.validatedFaces) || validation.validatedFaces < 1) throw new Error("未生成通过关键点与溯源检查的真实 DFL 人脸；保留暂存诊断，不发布空数据集");
  await writeFile(path.join(staging, "restoration-source.json"), `${JSON.stringify({ taskId, side, modelId: task.modelId, sourceMap: task.report?.outputs ?? task.outputs, validation, originalAlignedPreserved: true }, null, 2)}\n`, { flag: "wx" });
  await manager.safePath(parent, { directory: true });
  if (await lstat(destination).catch(error => { if (error.code === "ENOENT") return null; throw error; })) throw new Error("修复记录已提取，不覆盖已有副本");
  await rename(staging, destination);
  process.stdout.write(`[WEB] 修复人脸副本已保存：data_${side}/aligned_restored/${taskId}\n`);
  return { destination, validation };
  } catch (error) {
    await writeFile(path.join(staging, "extraction-failure.json"), `${JSON.stringify({ taskId, side, error: error.message, published: false }, null, 2)}\n`, { flag: "wx" }).catch(() => {});
    throw error;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try { await extractRestored(process.argv[2], JSON.parse(process.argv[3] ?? "{}")); }
  catch (error) { process.stderr.write(`[WEB] ${error.message}\n`); process.exitCode = 1; }
}
