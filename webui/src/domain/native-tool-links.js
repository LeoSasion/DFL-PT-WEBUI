// Fixed local-tool destinations only; hashes never carry paths or commands.
export function nativeToolDestination(hash) {
  const match = /^#\/(aligned|similarity)\/(src|dst)$/.exec(hash ?? "");
  if (!match) return null;
  return match[1] === "aligned" ? { nav: match[2] }
    : { nav: "tools", focus: { tab: "audit", mode: "similarity", side: match[2] } };
}
