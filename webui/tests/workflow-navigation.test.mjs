import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  getNavigationWorkflowGroup,
  getWorkflowNavigation,
  navigationWorkflowStages,
  workflowStageDestinations,
  workflowStages,
} from "../src/data/dashboard.js";

test("every workflow stage has a concrete navigation destination", () => {
  assert.deepEqual(
    Object.keys(workflowStageDestinations),
    workflowStages.map((stage) => stage.id),
  );

  for (const stage of workflowStages) {
    assert.match(workflowStageDestinations[stage.id].nav, /\S/);
  }
});

test("workflow destinations cover the dedicated command and product pages", () => {
  assert.deepEqual(workflowStageDestinations, {
    material: { nav: "video" },
    frames: { nav: "workflow.frames", task: "extract" },
    faces: { nav: "workflow.faces", task: "src" },
    clean: { nav: "workflow.clean", task: "sort" },
    mask: { nav: "xseg", task: "xseg" },
    train: { nav: "training", task: "me" },
    diagnose: { nav: "diagnostics", task: "diagnose" },
    merge: { nav: "merge", task: "merge" },
    encode: { nav: "export", task: "export" },
  });
});

test("primary navigation keeps the workflow highlight synchronized", () => {
  assert.equal(navigationWorkflowStages.video, "material");
  assert.equal(navigationWorkflowStages.xseg, "mask");
  assert.equal(navigationWorkflowStages.training, "train");
  assert.equal(navigationWorkflowStages.diagnostics, "diagnose");
  assert.equal(navigationWorkflowStages.merge, "merge");
  assert.equal(navigationWorkflowStages.export, "encode");
});

test("preparation pages expose their steps and a direct training exit", () => {
  for (const page of ["video", "src", "dst", "xseg", "workflow.frames", "workflow.faces", "workflow.clean", "workflow.roles"]) {
    assert.equal(getNavigationWorkflowGroup(page), "preprocess");
    assert.deepEqual(getWorkflowNavigation(page).stages.map(stage => stage.id), ["material", "frames", "faces", "clean", "mask", "train"]);
  }
});

test("all training pages show only three independent workflow groups", () => {
  for (const page of ["overview", "training", "diagnostics"]) {
    const navigation = getWorkflowNavigation(page);
    assert.equal(navigation.group, "training");
    assert.equal(navigation.kind, "groups");
    assert.deepEqual(navigation.stages.map(group => group.label), ["预处理", "训练", "后处理"]);
    for (const group of navigation.stages) assert.ok(workflowStageDestinations[group.stage]);
  }
});

test("postprocessing starts from a training return link and exposes merge and encode", () => {
  for (const page of ["merge", "export"]) {
    assert.equal(getNavigationWorkflowGroup(page), "postprocess");
    assert.deepEqual(getWorkflowNavigation(page).stages.map(stage => stage.id), ["train", "merge", "encode"]);
  }
  assert.equal(getNavigationWorkflowGroup("tools"), null);
  assert.equal(getNavigationWorkflowGroup("settings"), null);
});

test("terminal safe stop targets the selected training session", async () => {
  const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(source, /const safeStopSelectedJob = useCallback\(\(\) => \{/);
  assert.match(source, /if \(selectedJob\?\.id\) setStopTargetJobId\(selectedJob\.id\);/);
  assert.match(source, /onSafeStop=\{safeStopSelectedJob\}/);
  assert.match(source, /control\("close", targetJobId\)/);
});
