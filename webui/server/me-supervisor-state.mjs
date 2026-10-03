import { rename, writeFile } from "node:fs/promises";

const RETRY_DELAYS_MS = [20, 50, 100, 250, 500, 1000, 1500];
const RETRYABLE_RENAME_ERRORS = new Set(["EPERM", "EBUSY", "EACCES"]);
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// A failed Windows rename must not poison the supervisor's later heartbeats.
export function createMeSupervisorStateWriter(stateFile, {
  writeFileImpl = writeFile,
  renameImpl = rename,
  waitImpl = wait,
  retryDelaysMs = RETRY_DELAYS_MS,
} = {}) {
  let pending = Promise.resolve();
  return (value) => {
    const snapshot = `${JSON.stringify(value)}\n`;
    const operation = pending.catch(() => {}).then(async () => {
      const temporary = `${stateFile}.${process.pid}.tmp`;
      await writeFileImpl(temporary, snapshot, "utf8");
      for (let attempt = 0; ; attempt += 1) {
        try {
          await renameImpl(temporary, stateFile);
          return;
        } catch (error) {
          if (!RETRYABLE_RENAME_ERRORS.has(error.code) || attempt >= retryDelaysMs.length) {
            throw error;
          }
          await waitImpl(retryDelaysMs[attempt]);
        }
      }
    });
    pending = operation;
    return operation;
  };
}
