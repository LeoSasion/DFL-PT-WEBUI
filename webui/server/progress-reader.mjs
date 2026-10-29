import { StringDecoder } from "node:string_decoder";

// Progress is advisory; malformed or oversized diagnostic lines have no
// protocol authority. Discard an entire oversized line, never parse its tail.
export function createJsonProgressReader(onProgress, { prefix = "DFL_PROGRESS ", maxLineLength = 64_000 } = {}) {
  if (!Number.isSafeInteger(maxLineLength) || maxLineLength < 1) throw new RangeError("Progress line limit must be positive");
  const decoder = new StringDecoder("utf8");
  let pending = "", discarded = false, ended = false;
  const append = text => {
    if (discarded) return;
    if (pending.length + text.length > maxLineLength) { pending = ""; discarded = true; }
    else pending += text;
  };
  const emit = () => {
    if (!discarded && pending.startsWith(prefix)) {
      try { onProgress?.(JSON.parse(pending.slice(prefix.length))); } catch { /* Diagnostics are not a result. */ }
    }
    pending = ""; discarded = false;
  };
  const consume = text => {
    let offset = 0, end;
    while ((end = text.indexOf("\n", offset)) !== -1) {
      append(text.slice(offset, end)); emit(); offset = end + 1;
    }
    append(text.slice(offset));
  };
  return {
    write(chunk) { if (!ended) consume(decoder.write(typeof chunk === "string" ? Buffer.from(chunk) : chunk)); },
    end() { if (!ended) { ended = true; consume(decoder.end()); emit(); } },
  };
}
