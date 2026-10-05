import assert from "node:assert/strict";
import { Readable, Writable } from "node:stream";
import test from "node:test";
import { RuntimeServer } from "../server/app-server.mjs";

function runtime() {
  return new RuntimeServer({ jobManager: { list: () => [] }, operationManager: { list: () => [] } });
}

async function call(server, side = "src", method = "GET") {
  const request = Readable.from([]);
  Object.assign(request, { method, url: `/api/assets/${side}/similarity-review-state`,
    headers: { host: "127.0.0.1:4174", cookie: `dfl_web_session=${server.sessionToken}` } });
  const chunks = [];
  const response = new Writable({ write(chunk, _encoding, done) { chunks.push(Buffer.from(chunk)); done(); } });
  response.writeHead = (status, headers) => { response.status = status; response.headers = headers; };
  const finished = new Promise(resolve => response.on("finish", resolve));
  await server.handleRequest(request, response);
  await finished;
  return { status: response.status, body: JSON.parse(Buffer.concat(chunks).toString("utf8")) };
}

// The production read method is replaced before a request, so these route tests
// never read the workstation's workspace or registry and never spawn Python.
test("similarity recovery reads return the reviewed side, workspace and current inventory", async () => {
  const server = runtime();
  const reads = [];
  server.readSimilarityReviewState = async side => {
    reads.push(side);
    return { side, workspaceKey: "private-offline-fixture", names: ["remaining.png"], fingerprint: "fixture-fingerprint" };
  };
  for (const side of ["src", "dst"]) {
    const result = await call(server, side);
    assert.equal(result.status, 200);
    assert.deepEqual(result.body.data, { side, workspaceKey: "private-offline-fixture",
      names: ["remaining.png"], fingerprint: "fixture-fingerprint" });
  }
  assert.deepEqual(reads, ["src", "dst"]);
  assert.equal((await call(server, "src", "POST")).status, 404);
  assert.equal((await call(server, "other")).status, 404);
  assert.deepEqual(reads, ["src", "dst"]);
});

test("a quarantine still moving files refuses the read before inventory is requested", async () => {
  const server = runtime();
  let reads = 0;
  server.readSimilarityReviewState = async () => { reads++; throw new Error("must not read while moving"); };
  server.workspaceMutation = "批量隔离 aligned 图片";
  const moving = await call(server);
  assert.equal(moving.status, 409);
  assert.equal(moving.body.error.code, "WORKSPACE_MUTATION_BUSY");
  assert.equal(reads, 0);
  server.workspaceMutation = null;
  server.projectRestartPending = true;
  const switching = await call(server);
  assert.equal(switching.status, 409);
  assert.equal(switching.body.error.code, "PROJECT_RESTART_PENDING");
  assert.equal(reads, 0);
});

test("an unreadable inventory stays an error and a concurrent write cannot appear settled", async () => {
  const server = runtime();
  server.readSimilarityReviewState = async () => { throw Object.assign(new Error("offline unreadable inventory"),
    { code: "FIXTURE_INVENTORY_UNREADABLE", status: 422 }); };
  const unreadable = await call(server);
  assert.equal(unreadable.status, 422);
  assert.equal(unreadable.body.error.code, "FIXTURE_INVENTORY_UNREADABLE");
  assert.equal(unreadable.body.data, undefined);
  server.readSimilarityReviewState = async side => {
    server.workspaceMutation = "其他素材操作";
    return { side, workspaceKey: "private-offline-fixture", names: [], fingerprint: "fixture-fingerprint" };
  };
  const changed = await call(server);
  assert.equal(changed.status, 409);
  assert.equal(changed.body.error.code, "WORKSPACE_MUTATION_BUSY");
  assert.equal(changed.body.data, undefined);
});
