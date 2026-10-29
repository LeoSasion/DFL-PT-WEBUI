import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { isDfmPublicationPending, verifyDfmReceipt, verifyExtremeExport } from '../server/dfm-export.mjs';
import { getCommandDefinition, validateCommandParameters, buildCommand } from '../server/command-registry.mjs';
import { validateTaeConfig, TAE_DEFAULT_CONFIG } from '../shared/tae-training-options.mjs';
import { validateMeConfig, ME_DEFAULT_CONFIG } from '../shared/me-training-options.mjs';

test('DFM receipt binds job, sidecars, bytes and hashes', async t => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'dfm-receipt-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const files = [{ name: 'model.dfm', data: 'graph' }, { name: 'model.dfm.uuid.data', data: 'weights' }];
  for (const file of files) await writeFile(path.join(dir, file.name), file.data);
  const receipt = { version: 1, family: 'TAE', mode: 'normal', export_id: 'job-a',
    files: files.map(file => ({ name: file.name, bytes: Buffer.byteLength(file.data), sha256: createHash('sha256').update(file.data).digest('hex') })) };
  await writeFile(path.join(dir, 'model.dfm.export.json'), JSON.stringify(receipt));
  const expected = { family: 'TAE', mode: 'normal', jobId: 'job-a' };
  await verifyDfmReceipt(dir, expected);
  await assert.rejects(verifyDfmReceipt(dir, { ...expected, jobId: 'job-b' }), /不匹配/);
  await writeFile(path.join(dir, files[1].name), 'changed');
  await assert.rejects(verifyDfmReceipt(dir, expected), /校验失败/);
  await rm(path.join(dir, files[1].name));
  await assert.rejects(verifyDfmReceipt(dir, expected));
});

test('DFM pending publication cannot pass postflight even with matching receipt and hashes', async t => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'dfm-pending-receipt-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  for (const mode of ['normal', 'extreme']) {
    const name = mode === 'normal' ? 'model.dfm' : 'model_extreme.dfm';
    const graph = Buffer.from('matching graph');
    const report = { version: 1, family: 'ME', mode, export_id: 'pending-job', files: [
      { name, bytes: graph.length, sha256: createHash('sha256').update(graph).digest('hex') },
    ] };
    await writeFile(path.join(dir, name), graph);
    await writeFile(path.join(dir, name + '.export.json'), JSON.stringify(report));
    const expected = { family: 'ME', mode, jobId: 'pending-job' };
    assert.equal(await isDfmPublicationPending(dir, name), false);
    await verifyDfmReceipt(dir, expected);
    await writeFile(path.join(dir, name + '.publish.json'), '{"phase":"publishing"}');
    assert.equal(await isDfmPublicationPending(dir, name), true);
    await assert.rejects(verifyDfmReceipt(dir, expected), /发布尚未完成/);
    // An unreadable or malformed transaction is also pending, never a success.
    await writeFile(path.join(dir, name + '.publish.json'), 'incomplete journal');
    await assert.rejects(verifyDfmReceipt(dir, expected), /发布尚未完成/);
    await rm(path.join(dir, name + '.publish.json'));
    // OS lock files survive release and are not pending transaction markers.
    await writeFile(path.join(dir, name + '.publish.lock'), '');
    await verifyDfmReceipt(dir, expected);
  }
});

test('extreme preflight rejects absent runtime and unsupported GPU', async () => {
  const run = payload => async () => ({ code: 0, stdout: JSON.stringify(payload) });
  await verifyExtremeExport({ run: run({ missing: [], capability: [12, 0] }) });
  await assert.rejects(verifyExtremeExport({ run: run({ missing: ['modelopt'], capability: [12, 0] }) }), /requirements-export/);
  await assert.rejects(verifyExtremeExport({ run: run({ missing: [], capability: [8, 9] }) }), /Blackwell/);
});

test('both families expose normal/extreme mode and bind resolved calibration and export ID', () => {
  for (const family of ['me', 'tae']) {
    const id = `export.dfm_${family}`;
    const definition = getCommandDefinition(id);
    const parameters = validateCommandParameters(id, { forceModelName: 'example', exportMode: 'extreme' }, 'guided');
    const launch = buildCommand(definition, { parameters, jobId: 'job-a', preflight: { calibrationDir: 'X:/resolved/aligned', vae_weights: 'X:/verified.safetensors' } }).launch;
    assert.equal(launch.args[launch.args.indexOf('--calibration-dir') + 1], 'X:/resolved/aligned');
    assert.equal(launch.args[launch.args.indexOf('--export-id') + 1], 'job-a');
    assert.equal(launch.args[launch.args.indexOf('--export-mode') + 1], 'extreme');
    if (family === 'tae') assert.equal(launch.args[launch.args.indexOf('--vae-weights') + 1], 'X:/verified.safetensors');
    assert.throws(() => validateCommandParameters(id, { exportMode: 'fake' }, 'guided'));
  }
});

test('legacy training metadata normalizes precision without inventing resume changes', () => {
  const { precision, ...tae } = TAE_DEFAULT_CONFIG;
  assert.equal(validateTaeConfig(tae).precision, 'bf16');
  const { use_bf16, ...me } = ME_DEFAULT_CONFIG;
  assert.equal(validateMeConfig(me).use_bf16, false);
  assert.throws(() => validateMeConfig({ ...ME_DEFAULT_CONFIG, use_fp16: true, use_bf16: true }), /不能同时/);
});
