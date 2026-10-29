// Browser regression harness, loaded only by the explicit TAE UI smoke test.
import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { LanguageProvider } from "../i18n.jsx";
import { ProgressFeedbackProvider } from "../components/ProgressFeedback.jsx";
import { NewTaskDialog } from "../components/NewTaskDialog.jsx";
import { TrainingWorkspace } from "../components/TrainingView.jsx";
import "../styles.css";
import "../components/project-workflow.css";

if (!import.meta.env.DEV) throw new Error("This fixture is available only in the test development server");

function Fixture() {
  const fixture = window.taeFixture;
  const [taskType, setTaskType] = useState("train.tae");
  if (fixture.mode === "controls") return <TrainingWorkspace iteration={2} trainingState="running"
    trainingJob={{ id: "h3-test", commandId: "train.tae", state: "running", controls: ["save", "preview", "close"] }}
    pendingAction={null} savedModelState="empty" previewVersion={null} manualPreviewRefresh={0} />;
  return <NewTaskDialog open taskType={taskType} onTaskType={setTaskType} workspacePath="E:\\fixture\\workspace"
    serviceOnline commands={fixture.commands} workspace={{ models: fixture.models }}
    telemetry={{ gpus: [{ index: 0, name: "Fixture GPU", memoryTotalMiB: 24000 }] }}
    initialStep="parameters" onClose={() => {}} onCreate={() => { throw new Error("Fixture cannot launch training"); }}
    onPreflight={async (commandId, options) => {
      window.taePreflight = { commandId, options };
      return { resources: null };
    }} />;
}

createRoot(document.getElementById("root")).render(<LanguageProvider><ProgressFeedbackProvider><Fixture /></ProgressFeedbackProvider></LanguageProvider>);
