import { useCallback, useSyncExternalStore } from "react";
import { EMPTY_TERMINAL_SNAPSHOT } from "./terminal-event-store.js";

export function useTerminalEvents(store, jobId) {
  const subscribe = useCallback(callback => store?.subscribe(jobId, callback) ?? (() => {}), [store, jobId]);
  const snapshot = useCallback(() => store?.getSnapshot(jobId) ?? EMPTY_TERMINAL_SNAPSHOT, [store, jobId]);
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}
