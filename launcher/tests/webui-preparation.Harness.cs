using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Threading;

namespace DflPtWebUi.Launcher
{
    internal static class WebUiPreparationHarness
    {
        private static string qa;
        private static string fakeNode;
        private static int passed;
        private static int Main(string[] args)
        {
            if (args.Length == 2 && args[0].StartsWith("crash", StringComparison.Ordinal)) return Crash(args[0], args[1]);
            qa = Path.GetFullPath(args[0]); fakeNode = Path.GetFullPath(args[1]);
            Directory.CreateDirectory(qa);
            try
            {
                Check("dependency inputs invalidate package, lock and Node identities", Inputs);
                Check("failed install restores healthy modules and existing dist", InstallFailure);
                Check("failed build restores both validated dependencies and dist", BuildFailure);
                Check("failed native probe rolls back a nominally successful install", ProbeFailure);
                Check("successful preparation commits receipt and removes only old owned backups", Success);
                Check("build-only repair preserves modules without reinstalling", BuildOnly);
                Check("cold recovery restores interrupted preparation in a fresh process", ColdRecovery);
                Check("cold recovery is idempotent after one original was already restored", HalfRollback);
                Check("unknown backup writes are preserved and block destructive recovery", ChangedBackup);
                Check("a corrupt receipt backup is detected before restoring either tree", ChangedReceipt);
                Check("commit cleanup can resume after a locked file interrupted deletion", CleanupRetry);
                Check("normal pnpm junctions are moved and unlinked without touching targets", PnpmJunction);
                Check("internal pnpm junctions remain usable after a failed repair", PnpmRollback);
                Check("a top-level modules junction is rejected without touching its target", RootJunction);
                Check("another process cannot recover or replace a live preparation", ConcurrentLease);
                Check("corrupt and foreign recovery tokens cannot select arbitrary paths", CorruptJournal);
                Check("drive roots cannot become prepared runtime stores", DriveRoot);
                Check("source changes during preparation cannot publish a stale dependency receipt", ChangedInputs);
                Check("atomic replacement failure preserves its destination and cleans its own temporary file", AtomicFailure);
                Check("failed commit-journal replacement still rolls back the prepared trees", CommitJournalFailure);
                Console.WriteLine("TOTAL " + passed); return 0;
            }
            catch (Exception error) { Console.Error.WriteLine(error); return 1; }
        }

        private static void Check(string label, Action test) { test(); passed++; Console.WriteLine("PASS " + label); }
        private static void Assert(bool condition, string message) { if (!condition) throw new Exception(message); }
        private static void Throws(Action action) { bool threw = false; try { action(); } catch { threw = true; } Assert(threw, "Expected operation to refuse unsafe/incomplete state."); }
        private static string NewRoot(string name)
        {
            string root = Path.Combine(qa, name + " 中文 fixture");
            Directory.CreateDirectory(Path.Combine(root, "webui", "node_modules", "vite", "bin"));
            Directory.CreateDirectory(Path.Combine(root, "webui", "node_modules", "node-pty"));
            Directory.CreateDirectory(Path.Combine(root, "webui", "dist", "client"));
            Directory.CreateDirectory(Path.Combine(root, "_internal", "node", "bin"));
            File.Copy(fakeNode, Path.Combine(root, "_internal", "node", "bin", "node.exe"));
            Write(root, "webui/package.json", "{\"packageManager\":\"pnpm@11.19.0\"}");
            Write(root, "webui/pnpm-lock.yaml", "lock-v1");
            Write(root, "webui/node_modules/vite/bin/vite.js", "old-vite");
            Write(root, "webui/node_modules/node-pty/package.json", "{}");
            Write(root, "webui/node_modules/node-pty/index.js", "ready-old");
            Write(root, "webui/dist/client/index.html", "old usable build");
            Write(root, "workspace/model.bin", "MODEL MUST STAY");
            Write(root, "workspace/media.bin", "MEDIA MUST STAY");
            return root;
        }
        private static void Write(string root, string relative, string text)
        {
            string path = Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(path)); File.WriteAllText(path, text);
        }
        private static string Read(string root, string relative) { return File.ReadAllText(Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar))); }
        private static void Originals(string root)
        {
            Assert(Read(root, "webui/node_modules/node-pty/index.js") == "ready-old", "Original dependencies were not restored.");
            Assert(Read(root, "webui/dist/client/index.html") == "old usable build", "Original dist was not restored.");
            Protected(root);
        }
        private static void Protected(string root)
        {
            Assert(Read(root, "workspace/model.bin") == "MODEL MUST STAY", "Model changed.");
            Assert(Read(root, "workspace/media.bin") == "MEDIA MUST STAY", "Media changed.");
        }
        private static int Node(string root, string command)
        {
            ProcessStartInfo start = new ProcessStartInfo(Path.Combine(root, "_internal", "node", "bin", "node.exe"), command);
            start.WorkingDirectory = Path.Combine(root, "webui"); start.UseShellExecute = false; start.CreateNoWindow = true;
            using (Process process = Process.Start(start)) { if (!process.WaitForExit(10000)) { process.Kill(); throw new Exception("Fake Node timeout."); } return process.ExitCode; }
        }
        private static string Managed(string root) { return Path.Combine(root, ".launcher-install", "webui-preparation"); }
        private static string CurrentWork(string root) { return Path.Combine(Managed(root), File.ReadAllText(Path.Combine(Managed(root), "current.txt")).Trim()); }
        private static void Inputs()
        {
            string root = NewRoot("inputs");
            Assert(!WebUiDependencies.DependencyInputsMatch(root), "Missing receipt must not be accepted.");
            WebUiDependencies.RecordDependencyInputs(root); Assert(WebUiDependencies.DependencyInputsMatch(root), "Fresh receipt rejected.");
            Write(root, "webui/package.json", "changed-package"); Assert(!WebUiDependencies.DependencyInputsMatch(root), "Changed package was missed.");
            WebUiDependencies.RecordDependencyInputs(root); Write(root, "webui/pnpm-lock.yaml", "lock-v2");
            Assert(!WebUiDependencies.DependencyInputsMatch(root), "Changed lock was missed.");
            WebUiDependencies.RecordDependencyInputs(root);
            string node = Path.Combine(root, "_internal", "node", "bin", "node.exe");
            File.SetLastWriteTimeUtc(node, File.GetLastWriteTimeUtc(node).AddSeconds(3));
            Assert(!WebUiDependencies.DependencyInputsMatch(root), "Changed Node identity was missed.");
        }
        private static void InstallFailure()
        {
            string root = NewRoot("install-failure"); WebUiDependencies.RecordDependencyInputs(root);
            using (WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true))
            { Assert(Node(root, "install-fail") == 17, "Synthetic install failure was not observed."); }
            Originals(root); Assert(WebUiDependencies.DependencyInputsMatch(root), "Old input receipt was not retained.");
        }
        private static void BuildFailure()
        {
            string root = NewRoot("build-failure"); WebUiDependencies.RecordDependencyInputs(root);
            using (WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true))
            { Assert(Node(root, "install-ok") == 0, "Install failed."); Assert(Node(root, "build-fail") == 19, "Build failure missing."); WebUiDependencies.RecordDependencyInputs(root); }
            Originals(root);
        }
        private static void ProbeFailure()
        {
            string root = NewRoot("probe-failure");
            using (WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true))
            { Node(root, "install-broken"); Assert(!WebUiDependencies.Inspect(root, null).Ready, "Broken native dependencies passed probe."); }
            Originals(root);
        }
        private static void Success()
        {
            string root = NewRoot("success");
            using (WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true))
            { Node(root, "install-ok"); Assert(WebUiDependencies.Inspect(root, null).Ready, "Fresh modules failed."); Node(root, "build-ok"); WebUiDependencies.RecordDependencyInputs(root); tx.Commit(); Assert(tx.CleanupWarning == null, "Unexpected cleanup warning."); }
            Assert(WebUiDependencies.DependencyInputsMatch(root), "Committed receipt missing.");
            Assert(Read(root, "webui/dist/client/index.html") == "new synthetic build", "New build not committed.");
            Assert(!File.Exists(Path.Combine(Managed(root), "current.txt")), "Committed journal pointer remains."); Protected(root);
        }
        private static void BuildOnly()
        {
            string root = NewRoot("build-only"); WebUiDependencies.RecordDependencyInputs(root);
            using (WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, false, true))
            { Assert(Read(root, "webui/node_modules/node-pty/index.js") == "ready-old", "Build-only repair moved dependencies."); Node(root, "build-ok"); tx.Commit(); }
            Assert(Read(root, "webui/node_modules/node-pty/index.js") == "ready-old", "Build-only repair replaced dependencies.");
        }
        private static int Child(string operation, string root)
        {
            ProcessStartInfo start = new ProcessStartInfo(Assembly.GetExecutingAssembly().Location, operation + " \"" + root + "\"");
            start.UseShellExecute = false; start.CreateNoWindow = true;
            using (Process process = Process.Start(start)) { if (!process.WaitForExit(10000)) { process.Kill(); throw new Exception("Child fixture timeout."); } return process.ExitCode; }
        }
        private static int Crash(string operation, string root)
        {
            if (operation == "crash-contender") { try { WebUiDependencies.RecoverPreparation(root); return 9; } catch (IOException) { return 23; } }
            WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true);
            Node(root, "install-ok"); Node(root, "build-fail");
            if (operation == "crash-half")
            {
                Directory.Move(Path.Combine(root, "webui", "node_modules"), Path.Combine(tx.RecoveryPath, "failed-modules"));
                Directory.Move(Path.Combine(tx.RecoveryPath, "original-modules"), Path.Combine(root, "webui", "node_modules"));
            }
            Environment.Exit(37); return 37;
        }
        private static void ColdRecovery() { string root = NewRoot("cold"); Assert(Child("crash", root) == 37, "Child did not stop abruptly."); WebUiDependencies.RecoverPreparation(root); Originals(root); WebUiDependencies.RecoverPreparation(root); }
        private static void HalfRollback() { string root = NewRoot("half"); Assert(Child("crash-half", root) == 37, "Child did not stop in rollback window."); WebUiDependencies.RecoverPreparation(root); Originals(root); }
        private static void ChangedBackup()
        {
            string root = NewRoot("changed-backup"); WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true);
            Node(root, "install-ok"); Node(root, "build-ok");
            string extra = Path.Combine(tx.RecoveryPath, "original-modules", "user-note.txt"); File.WriteAllText(extra, "DO NOT DELETE");
            WebUiDependencies.RecordDependencyInputs(root); tx.Commit(); Assert(tx.CleanupWarning != null, "Unexpected backup write was not preserved.");
            Throws(delegate { WebUiDependencies.RecoverPreparation(root); }); Assert(File.ReadAllText(extra) == "DO NOT DELETE", "Unknown backup write was deleted."); Protected(root);
        }
        private static void ChangedReceipt()
        {
            string root = NewRoot("changed-receipt"); WebUiDependencies.RecordDependencyInputs(root);
            WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true); Node(root, "install-ok"); Node(root, "build-ok");
            File.WriteAllText(Path.Combine(tx.RecoveryPath, "receipt.txt"), "unknown edit");
            Throws(delegate { tx.Dispose(); });
            Assert(Read(root, "webui/node_modules/node-pty/index.js") == "ready-new", "Receipt preflight restored one tree before detecting corruption."); Protected(root);
        }
        private static void CleanupRetry()
        {
            string root = NewRoot("cleanup-retry"); WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true);
            Node(root, "install-ok"); Node(root, "build-ok"); WebUiDependencies.RecordDependencyInputs(root);
            using (FileStream locked = new FileStream(Path.Combine(tx.RecoveryPath, "original-modules", "node-pty", "index.js"), FileMode.Open, FileAccess.Read, FileShare.Read))
            { tx.Commit(); Assert(tx.CleanupWarning != null, "Locked backup did not preserve a cleanup warning."); }
            WebUiDependencies.RecoverPreparation(root);
            Assert(!File.Exists(Path.Combine(Managed(root), "current.txt")), "Interrupted cleanup did not finish.");
            Assert(Read(root, "webui/node_modules/node-pty/index.js") == "ready-new", "Cleanup retry rolled back a committed runtime."); Protected(root);
        }
        private static void Junction(string link, string target)
        {
            ProcessStartInfo start = new ProcessStartInfo("cmd.exe", "/d /c mklink /J \"" + link + "\" \"" + target + "\"");
            start.UseShellExecute = false; start.CreateNoWindow = true; start.RedirectStandardOutput = true; start.RedirectStandardError = true;
            using (Process process = Process.Start(start)) { process.WaitForExit(5000); Assert(process.HasExited && process.ExitCode == 0, "Cannot create isolated junction: " + process.StandardError.ReadToEnd() + process.StandardOutput.ReadToEnd()); }
        }
        private static void PnpmJunction()
        {
            string root = NewRoot("pnpm-link"), target = Path.Combine(root, "external-package");
            Directory.CreateDirectory(target); File.WriteAllText(Path.Combine(target, "keep.txt"), "EXTERNAL KEEP");
            Junction(Path.Combine(root, "webui", "node_modules", "linked-package"), target);
            using (WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true)) { Node(root, "install-ok"); Node(root, "build-ok"); tx.Commit(); Assert(tx.CleanupWarning == null, "Normal pnpm link blocked cleanup."); }
            Assert(File.ReadAllText(Path.Combine(target, "keep.txt")) == "EXTERNAL KEEP", "Link cleanup touched external target.");
        }
        private static void RootJunction()
        {
            string root = NewRoot("root-link"), target = Path.Combine(root, "other-modules");
            Directory.Move(Path.Combine(root, "webui", "node_modules"), target); Junction(Path.Combine(root, "webui", "node_modules"), target);
            Throws(delegate { WebUiDependencies.BeginPreparation(root, true, true); });
            Assert(File.ReadAllText(Path.Combine(target, "node-pty", "index.js")) == "ready-old", "Root link target changed.");
            Directory.Delete(Path.Combine(root, "webui", "node_modules"), false);
        }
        private static void PnpmRollback()
        {
            string root = NewRoot("pnpm-rollback");
            string target = Path.Combine(root, "webui", "node_modules", ".pnpm", "package");
            Directory.CreateDirectory(target); File.WriteAllText(Path.Combine(target, "index.js"), "PNPM ORIGINAL");
            string link = Path.Combine(root, "webui", "node_modules", "linked-package");
            Junction(link, target);
            using (WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true)) { Node(root, "install-fail"); }
            Assert(File.ReadAllText(Path.Combine(link, "index.js")) == "PNPM ORIGINAL", "Restored pnpm link is broken.");
            Directory.Delete(link, false); Originals(root);
        }
        private static void ConcurrentLease()
        {
            string root = NewRoot("lease"); using (WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true))
                Assert(Child("crash-contender", root) == 23, "Another process recovered a live transaction.");
            Originals(root);
        }
        private static void CorruptJournal()
        {
            string root = NewRoot("corrupt"); Directory.CreateDirectory(Managed(root));
            File.WriteAllText(Path.Combine(Managed(root), "current.txt"), "../../outside");
            Throws(delegate { WebUiDependencies.RecoverPreparation(root); }); Originals(root);
        }
        private static void DriveRoot() { Throws(delegate { PreparedRuntimeStore.GetRoot(Path.GetPathRoot(qa)); }); }
        private static void ChangedInputs()
        {
            string root = NewRoot("changed-inputs"); WebUiDependencies.RecordDependencyInputs(root);
            using (WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true))
            {
                Node(root, "install-ok"); Node(root, "build-ok");
                Write(root, "webui/pnpm-lock.yaml", "unexpected lock edit");
                Throws(delegate { WebUiDependencies.RecordDependencyInputs(root); });
                Throws(delegate { tx.Commit(); });
            }
            Originals(root); Assert(!WebUiDependencies.DependencyInputsMatch(root), "Changed input became falsely healthy.");
        }
        private static void AtomicFailure()
        {
            string root = NewRoot("atomic"); string path = Path.Combine(root, "atomic-receipt.txt");
            File.WriteAllText(path, "old receipt");
            using (FileStream locked = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
                Throws(delegate { WebUiDependencies.AtomicWrite(path, "replacement"); });
            Assert(File.ReadAllText(path) == "old receipt", "Atomic replacement altered a locked destination.");
            Assert(Directory.GetFiles(root, "atomic-receipt.txt.new-*").Length == 0, "Own temporary file leaked.");
        }
        private static void CommitJournalFailure()
        {
            string root = NewRoot("commit-journal"); WebUiDependencies.RecordDependencyInputs(root);
            using (WebUiPreparation tx = WebUiDependencies.BeginPreparation(root, true, true))
            {
                Node(root, "install-ok"); Node(root, "build-ok"); WebUiDependencies.RecordDependencyInputs(root);
                using (FileStream locked = new FileStream(Path.Combine(tx.RecoveryPath, "journal.txt"), FileMode.Open, FileAccess.Read, FileShare.Read))
                    Throws(delegate { tx.Commit(); });
            }
            Originals(root); Assert(!File.Exists(Path.Combine(Managed(root), "current.txt")), "Failed commit did not restore immediately.");
        }
    }
}
