#!/usr/bin/env node
import { copyFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const index = path.join(dist, "client", "index.html");
const worker = path.join(root, "worker", "index.js");
const hosting = path.join(root, ".openai", "hosting.json");

const sitesBuild = process.argv.includes("--sites");
for (const file of [index, worker, ...(sitesBuild ? [hosting] : [])]) {
  if (!existsSync(file)) throw new Error("Missing Sites build input: " + file);
}

mkdirSync(path.join(dist, "server"), { recursive: true });
copyFileSync(worker, path.join(dist, "server", "index.js"));
const outputHosting = path.join(dist, ".openai", "hosting.json");
if (sitesBuild) {
  mkdirSync(path.dirname(outputHosting), { recursive: true });
  copyFileSync(hosting, outputHosting);
  console.log("Prepared Sites build with the local deployment binding.");
} else {
  rmSync(outputHosting, { force: true });
  console.log("Prepared local web build without a Sites deployment binding.");
}
