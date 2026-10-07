import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, writeFile, copyFile, rm } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { PATHS } from '../server/paths.mjs';
import { RestorationManager } from '../server/restoration-manager.mjs';
import { extractRestored, validateRestoredFaces } from '../server/extract-restored.mjs';

// A real JPEG serialized by DFLJPG with controlled geometry, not a fake validator.
const FIXTURE = String.raw`
import cv2,numpy as np,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
from DFLIMG import DFLJPG
root=Path(sys.argv[1]); file=root/'face.jpg'
cv2.imwrite(str(file),np.full((64,64,3),127,np.uint8))
face=DFLJPG.load(file); p68=np.arange(136).reshape(68,2).astype(float)/4
p98=np.arange(196).reshape(98,2).astype(float)/4; affine=[[1.,0.,2.],[0.,1.,3.]]
native={'model_id':'tufa98','point_definition':{'name':'WFLW98'},'points_original':p98.tolist(),
'points_aligned':(p98+[2.,3.]).tolist(),'source_to_aligned_affine':affine,'aligned_canvas_wh':[64,64]}
restoration={'sourceFilename':'original.jpeg','processedFilename':'restored.png','processedSha256':'b'*64,'declaredInputSha256':'a'*64}
provenance={'alignmentModel':'tufa68','processedSourceSha256':'b'*64,'processedCanvasWH':[128,96]}
face.set_dict({'landmarks':p68.tolist(),'source_landmarks':p68.tolist(),'source_filename':'original.jpeg',
'image_to_face_mat':affine,'native_landmarks98':native,'restoration_provenance':restoration,'landmark_provenance':provenance})
face.save()
paired={'schemaVersion':1,'aligned_filename':file.name,'aligned_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
'source_filename':'original.jpeg','native_landmarks98':native,'restoration_provenance':restoration,'landmark_provenance':provenance}
Path(str(file)+'.landmarks.json').write_text(json.dumps(paired),encoding='utf-8')
`;
async function python(args) {
  return new Promise((resolve,reject)=>{
    let err=''; const child=spawn(PATHS.python,args,{cwd:PATHS.repositoryRoot,windowsHide:true,stdio:['ignore','ignore','pipe']});
    child.stderr.on('data',c=>err+=c); child.on('error',reject); child.on('close',code=>code===0?resolve():reject(Error(err)));
  });
}
async function fixture(callback) {
  const root=await mkdtemp(path.join(os.tmpdir(),'dfl-restored-extraction-'));
  const workspace=path.join(root,'workspace'), runtime=path.join(workspace,'.webui'), id='rst-'+'1'.repeat(32);
  const manager=new RestorationManager({workspaceRoot:workspace,runtimeRoot:runtime});
  await mkdir(path.join(workspace,'data_src','aligned'),{recursive:true}); await manager.initialize();
  const sentinel=path.join(workspace,'data_src','aligned','original.jpg'); await writeFile(sentinel,'untouched');
  const output=path.join(manager.root,'outputs',id), input=path.join(output,'images'), faces=path.join(root,'fixture-faces');
  await mkdir(input,{recursive:true}); await mkdir(faces);
  const entry={sourceName:'original.jpeg',name:'restored.png',inputSha256:'a'.repeat(64),outputSha256:'b'.repeat(64),width:128,height:96};
  await writeFile(path.join(output,'manifest.json'),JSON.stringify({outputs:[entry]}));
  await writeFile(path.join(manager.root,'tasks',id+'.json'),JSON.stringify({taskId:id,side:'src',status:'completed',modelId:'swinir-psnr',outputs:[entry]}));
  // Completion verification has separate manager tests; here isolate the extraction/publication contract.
  manager.resolveOutputDirectory=async()=>input;
  const paths={...PATHS,workspaceRoot:workspace,runtimeRoot:runtime};
  const destination=path.join(workspace,'data_src','aligned_restored',id);
  const produce=async args=>{
    const staging=args[args.indexOf('--output-dir')+1]; await python(['-B','-c',FIXTURE,faces,PATHS.currentDflRoot]);
    await copyFile(path.join(faces,'face.jpg'),path.join(staging,'face.jpg'));
    await copyFile(path.join(faces,'face.jpg.landmarks.json'),path.join(staging,'face.jpg.landmarks.json'));
  };
  try { await callback({manager,paths,id,destination,produce,faces,sentinel,output}); }
  finally { await manager.close(); await rm(root,{recursive:true,force:true}); }
}

test('exit-zero with no actual faces fails atomically and retains diagnostic staging',async()=>fixture(async f=>{
  await assert.rejects(extractRestored('src',{restorationTaskId:f.id},{...f,runExtractor:async()=>{}}),/empty extraction cannot publish/);
  await assert.rejects(readFile(path.join(f.destination,'restoration-source.json')),e=>e.code==='ENOENT');
  const parent=path.dirname(f.destination), pending=(await readdir(parent)).filter(n=>n.startsWith('.pending-'));
  assert.equal(pending.length,1);
  assert.equal(JSON.parse(await readFile(path.join(parent,pending[0],'extraction-failure.json'),'utf8')).published,false);
  assert.equal(await readFile(f.sentinel,'utf8'),'untouched');
}));

test('real DFL68 plus SHA-bound TUFA98 sidecar publish once with original aligned preserved',async()=>fixture(async f=>{
  const result=await extractRestored('src',{restorationTaskId:f.id},{...f,runExtractor:f.produce});
  assert.equal(result.validation.validatedFaces,1);
  const record=result.validation.records[0]; assert.equal(record.legacyPointCount,68); assert.equal(record.nativePointCount,98);
  assert.equal(record.sourceName,'original.jpeg'); assert.equal(record.inputSha256,'a'.repeat(64));
  assert.equal(record.affineRoundtripMaxPx,0); assert.equal(await readFile(f.sentinel,'utf8'),'untouched');
  assert.equal((await readdir(path.dirname(f.destination))).some(n=>n.startsWith('.pending-')),false);
  await assert.rejects(extractRestored('src',{restorationTaskId:f.id},{...f,runExtractor:f.produce}),/不覆盖/);
}));

test('one valid face plus a broken second face cannot publish a partial batch',async()=>fixture(async f=>{
  await assert.rejects(extractRestored('src',{restorationTaskId:f.id},{...f,runExtractor:async args=>{
    assert.ok(args.includes('--no-output-debug'),'Background extraction must not prompt for debug output');
    await f.produce(args); const staging=args[args.indexOf('--output-dir')+1];
    await copyFile(path.join(staging,'face.jpg'),path.join(staging,'second.jpg'));
  }}),/sidecar missing/);
  await assert.rejects(readFile(path.join(f.destination,'face.jpg')),e=>e.code==='ENOENT');
  const pending=(await readdir(path.dirname(f.destination))).find(n=>n.startsWith('.pending-'));
  assert.ok(await readFile(path.join(path.dirname(f.destination),pending,'face.jpg')));
  assert.ok(await readFile(path.join(path.dirname(f.destination),pending,'second.jpg')));
}));

test('missing sidecar, hash tamper and affine mismatch cannot become published faces',async()=>fixture(async f=>{
  await f.produce([PATHS.currentMain,'--output-dir',f.faces]);
  const sidecar=path.join(f.faces,'face.jpg.landmarks.json'), original=await readFile(sidecar,'utf8');
  await rm(sidecar); await assert.rejects(validateRestoredFaces(f.faces,path.join(f.output,'manifest.json')),/sidecar missing/);
  const data=JSON.parse(original); data.aligned_sha256='0'.repeat(64); await writeFile(sidecar,JSON.stringify(data));
  await assert.rejects(validateRestoredFaces(f.faces,path.join(f.output,'manifest.json')),/not bound/);
  await writeFile(sidecar,original);
  await python(['-B','-c',"import sys;sys.path.insert(0,sys.argv[2]);from DFLIMG import DFLJPG;f=DFLJPG.load(sys.argv[1]);f.get_dict()['native_landmarks98']['points_aligned'][0][0]+=10;f.save()",path.join(f.faces,'face.jpg'),PATHS.currentDflRoot]);
  await assert.rejects(validateRestoredFaces(f.faces,path.join(f.output,'manifest.json')),/do not roundtrip/);
}));

test('explicit FAN extraction requires real legacy metadata but does not claim TUFA98',async()=>fixture(async f=>{
  await f.produce([PATHS.currentMain,'--output-dir',f.faces]);
  await python(['-B','-c',"import sys;sys.path.insert(0,sys.argv[2]);from DFLIMG import DFLJPG;f=DFLJPG.load(sys.argv[1]);m=f.get_dict();m.pop('native_landmarks98');m['landmark_provenance']['alignmentModel']='fan68';f.save()",path.join(f.faces,'face.jpg'),PATHS.currentDflRoot]);
  await rm(path.join(f.faces,'face.jpg.landmarks.json'));
  const result=await validateRestoredFaces(f.faces,path.join(f.output,'manifest.json'),PATHS,'fan');
  assert.equal(result.validatedFaces,1); assert.equal(result.records[0].nativePointCount,null);
  await assert.rejects(validateRestoredFaces(f.faces,path.join(f.output,'manifest.json')),/TUFA98 audit required/);
}));
