import assert from "node:assert/strict";
import test from "node:test";
import { Readable, Writable } from "node:stream";
import { RuntimeServer } from "../server/app-server.mjs";

async function call(server, method, url, body, authenticated = true) {
  const request = Readable.from(body ? [Buffer.from(JSON.stringify(body))] : []);
  Object.assign(request, { method, url, headers: { host: "127.0.0.1:4174", ...(authenticated ? {cookie:`dfl_web_session=${server.sessionToken}`} : {}) } });
  const chunks = [];
  const response = new Writable({ write(chunk, _, done) { chunks.push(Buffer.from(chunk)); done(); } });
  response.writeHead = (status, headers) => { response.status = status; response.headers = headers; };
  const finished = new Promise(resolve => response.on("finish", resolve));
  await server.handleRequest(request, response); await finished;
  return { status: response.status, payload: JSON.parse(Buffer.concat(chunks).toString()) };
}

test("feedback API requires a local session and filters diagnostics before returning a preview", async () => {
  const server = new RuntimeServer({jobManager:{list:()=>[]},operationManager:{list:()=>[]}});
  server.buildSystemDiagnostic = async () => ({ product:{node:"24.19.0"}, telemetry:{error:"C:\\private\\sk-secret"}, workspace:{projectId:"private-person",modelCount:2} });
  assert.equal((await call(server,"POST","/api/system/feedback",{failureStep:"image"},false)).status,403);
  const result = await call(server,"POST","/api/system/feedback",{failureStep:"image",rawLog:"sk-secret"});
  assert.equal(result.status,200); assert.equal(result.payload.data.failureStep,"image");
  assert.doesNotMatch(JSON.stringify(result.payload),/private-person|sk-secret|C:\\\\/);
  assert.match(result.payload.data.preview,/图像编辑/);
  server.buildSystemDiagnostic = async () => { throw new Error("C:\\private\\unavailable"); };
  const degraded = await call(server,"POST","/api/system/feedback",{failureStep:"install"});
  assert.equal(degraded.status,200); assert.equal(degraded.payload.data.diagnostics.runtime.pythonAvailable,null);
});

test("release API distinguishes fixed product identity, resource presence and untested training",async()=>{
  const server = new RuntimeServer({jobManager:{list:()=>[]},operationManager:{list:()=>[]}});
  const result = await call(server,"GET","/api/system/release");
  assert.equal(result.status,200); assert.equal(result.payload.data.product,"DFL-PT-WEBUI");
  assert.equal(result.payload.data.readiness.trainingReady,null);
  assert.equal(result.payload.data.readiness.trainingStatus,"requires-command-preflight");
  assert.doesNotMatch(JSON.stringify(result.payload),/E:\\\\|Users\\\\/);
});
