using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace DflPtWebUi.Launcher
{
    internal sealed class WebUiDependencyHealth
    {
        public bool Ready { get; set; }
        public string Detail { get; set; }
    }

    internal static class WebUiDependencies
    {
        private const string PreparationDirectory = ".launcher-install/webui-preparation";

        public static void RecoverPreparation(string projectRoot)
        {
            WebUiPreparation.Recover(projectRoot);
        }

        public static WebUiPreparation BeginPreparation(string projectRoot, bool replaceModules, bool replaceDist)
        {
            return new WebUiPreparation(projectRoot, replaceModules, replaceDist);
        }

        public static bool DependencyInputsMatch(string projectRoot)
        {
            try
            {
                string receipt = SafePath(projectRoot, PreparationDirectory + "/dependency-inputs.txt");
                if (!File.Exists(receipt)) return false;
                string expected = DependencySignature(projectRoot);
                return File.ReadAllText(receipt, Encoding.UTF8) == "webui-inputs-v1\n" + expected + "\n";
            }
            catch { return false; }
        }

        public static void RecordDependencyInputs(string projectRoot)
        {
            string receipt = SafePath(projectRoot, PreparationDirectory + "/dependency-inputs.txt");
            string signature = DependencySignature(projectRoot);
            WebUiPreparation.ValidateInputs(projectRoot, signature);
            Directory.CreateDirectory(Path.GetDirectoryName(receipt));
            AtomicWrite(receipt, "webui-inputs-v1\n" + signature + "\n");
        }

        internal static string DependencySignature(string projectRoot)
        {
            string package = SafePath(projectRoot, "webui/package.json");
            string lockfile = SafePath(projectRoot, "webui/pnpm-lock.yaml");
            string node = SafePath(projectRoot, "_internal/node/bin/node.exe");
            FileInfo image = new FileInfo(node);
            if (!File.Exists(package) || !File.Exists(lockfile) || !image.Exists)
                throw new IOException("WebUI 依赖输入不完整，不能记录为已验证。");
            int machine;
            using (BinaryReader reader = new BinaryReader(File.OpenRead(node)))
            {
                if (reader.BaseStream.Length < 64 || reader.ReadUInt16() != 0x5a4d)
                    throw new IOException("项目 Node.js 不是有效的 Windows 可执行文件。");
                reader.BaseStream.Position = 0x3c;
                int header = reader.ReadInt32();
                if (header < 64 || header > reader.BaseStream.Length - 6)
                    throw new IOException("项目 Node.js 可执行文件头不完整。");
                reader.BaseStream.Position = header;
                if (reader.ReadUInt32() != 0x00004550) throw new IOException("项目 Node.js 文件头无效。");
                machine = reader.ReadUInt16();
            }
            string version = FileVersionInfo.GetVersionInfo(node).FileVersion ?? String.Empty;
            // The two small source manifests are hashed. Node's version, machine
            // and file identity catch ABI/runtime replacements without repeatedly
            // reading the complete portable executable on each repair check.
            return HashText(FileHash(package) + "\n" + FileHash(lockfile) + "\n" + version + "\n"
                + machine + "\n" + image.Length + "\n" + image.LastWriteTimeUtc.Ticks);
        }

        internal static string FileHash(string path)
        {
            using (SHA256 hash = SHA256.Create())
            using (Stream stream = File.OpenRead(path))
                return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", String.Empty);
        }

        internal static string HashText(string value)
        {
            using (SHA256 hash = SHA256.Create())
                return BitConverter.ToString(hash.ComputeHash(Encoding.UTF8.GetBytes(value))).Replace("-", String.Empty);
        }

        internal static string SafePath(string projectRoot, string relative)
        {
            string root = Path.GetFullPath(projectRoot).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (root.Length <= 2 || String.Equals(root, Path.GetPathRoot(root), StringComparison.OrdinalIgnoreCase))
                throw new IOException("WebUI 准备目录不能是磁盘根目录。");
            string path = Path.GetFullPath(Path.Combine(root, relative));
            if (!path.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                throw new IOException("WebUI 准备路径必须位于本项目内。");
            string current = path;
            while (current != null && current.Length >= root.Length)
            {
                if (Exists(current) && (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
                    throw new IOException("WebUI 准备路径不能经过目录链接：" + current);
                if (String.Equals(current, root, StringComparison.OrdinalIgnoreCase)) break;
                current = Path.GetDirectoryName(current);
            }
            return path;
        }

        internal static bool Exists(string path)
        {
            try { File.GetAttributes(path); return true; }
            catch (FileNotFoundException) { return false; }
            catch (DirectoryNotFoundException) { return false; }
        }

        internal static void AtomicWrite(string path, string value)
        {
            string temporary = path + ".new-" + Guid.NewGuid().ToString("N");
            try
            {
                File.WriteAllText(temporary, value, new UTF8Encoding(false));
                if (File.Exists(path)) File.Replace(temporary, path, null);
                else File.Move(temporary, path);
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
        }
        // esbuild belongs to Vite, so resolve it from Vite's dependency scope.
        // pnpm does not expose transitive dependencies at the project root.
        public const string ProbeScript =
            "let failed=false;"
            + "for(const name of ['node-pty','vite/esbuild']){try{"
            + "if(name==='node-pty'){require('node-pty')}else{"
            + "const r=require('node:module').createRequire(require.resolve('vite/package.json'));"
            + "r('esbuild').transformSync('let ready=true')}"
            + "console.log(name+': OK')"
            + "}catch(error){failed=true;console.error(name+': '+(error.stack||error))}}"
            + "if(failed)process.exitCode=1;";

        public static bool EntryPointsPresent(string projectRoot)
        {
            string modules = Path.Combine(projectRoot, "webui", "node_modules");
            return File.Exists(Path.Combine(modules, "vite", "bin", "vite.js"))
                && File.Exists(Path.Combine(modules, "node-pty", "package.json"));
        }

        // A file existing does not prove its native module can load. Keep this
        // read-only probe bounded so a broken installation cannot hang startup.
        public static WebUiDependencyHealth Inspect(string projectRoot,
            IDictionary<string, string> environment, int timeoutMilliseconds = 10000)
        {
            string node = Path.Combine(projectRoot, "_internal", "node", "bin", "node.exe");
            if (!File.Exists(node)) return Failed("缺少项目内的 Node.js，请修复依赖。");
            if (!EntryPointsPresent(projectRoot)) return Failed("WebUI 依赖入口缺失，请修复依赖。");
            try
            {
                ProcessStartInfo start = new ProcessStartInfo(node, "-e \"" + ProbeScript + "\"");
                start.WorkingDirectory = Path.Combine(projectRoot, "webui");
                start.UseShellExecute = false;
                start.CreateNoWindow = true;
                start.RedirectStandardOutput = true;
                start.RedirectStandardError = true;
                start.StandardOutputEncoding = Encoding.UTF8;
                start.StandardErrorEncoding = Encoding.UTF8;
                if (environment != null)
                    foreach (KeyValuePair<string, string> item in environment)
                        start.EnvironmentVariables[item.Key] = item.Value ?? String.Empty;
                using (Process process = Process.Start(start))
                {
                    Task<string> output = process.StandardOutput.ReadToEndAsync();
                    Task<string> error = process.StandardError.ReadToEndAsync();
                    if (!process.WaitForExit(Math.Max(1, timeoutMilliseconds)))
                    {
                        try { process.Kill(); } catch { }
                        process.WaitForExit(2000);
                        return Failed("WebUI 依赖加载检查超时，请查看日志并修复依赖。");
                    }
                    if (!Task.WaitAll(new Task[] { output, error }, 2000))
                        return Failed("WebUI 依赖检查输出未结束，请修复依赖。");
                    if (process.ExitCode != 0)
                        return Failed((error.Result + Environment.NewLine + output.Result).Trim());
                    return new WebUiDependencyHealth { Ready = true, Detail = "node-pty 与 Vite/esbuild 加载通过。" };
                }
            }
            catch (Exception error) { return Failed("WebUI 依赖检查失败：" + error.Message); }
        }

        private static WebUiDependencyHealth Failed(string detail)
        {
            return new WebUiDependencyHealth { Ready = false,
                Detail = detail.Length > 8000 ? detail.Substring(0, 8000) : detail };
        }
    }

    // Owns only node_modules, dist and their small dependency-input receipt.
    // npm/pnpm/Vite remain in HostController's command runner. Directory moves
    // preserve normal pnpm links without following them into another tree.
    internal sealed class WebUiPreparation : IDisposable
    {
        private const string Managed = ".launcher-install/webui-preparation";
        private readonly string root;
        private readonly string work;
        private readonly Dictionary<string, string> journal;
        private FileStream lease;
        private bool finished;
        public string CleanupWarning { get; private set; }
        public string RecoveryPath { get { return work; } }

        internal WebUiPreparation(string projectRoot, bool replaceModules, bool replaceDist)
        {
            root = Path.GetFullPath(projectRoot);
            lease = AcquireLease(root);
            try
            {
                RecoverCore(root);
                string pointer = WebUiDependencies.SafePath(root, Managed + "/current.txt");
                if (File.Exists(pointer)) throw new IOException("上次 WebUI 准备尚未恢复，请先处理恢复记录。");
                string token = Guid.NewGuid().ToString("N");
                work = WebUiDependencies.SafePath(root, Managed + "/" + token);
                journal = new Dictionary<string, string>(StringComparer.Ordinal);
                journal["schema"] = "1"; journal["token"] = token; journal["phase"] = "preparing";
                journal["inputs"] = WebUiDependencies.DependencySignature(root);
                journal["backupsReady"] = "0";
                Dictionary<string, string> inventories = new Dictionary<string, string>();
                foreach (string item in new[] { "modules", "dist" })
                {
                    bool replace = item == "modules" ? replaceModules : replaceDist;
                    string active = Active(item);
                    journal[item] = replace ? "1" : "0";
                    journal[item + "Had"] = replace && WebUiDependencies.Exists(active) ? "1" : "0";
                    string inventory = journal[item + "Had"] == "1" ? Inventory(active) : "";
                    journal[item + "Hash"] = inventory.Length > 0 ? WebUiDependencies.HashText(inventory) : "";
                    journal[item + "Deleting"] = "0";
                    if (inventory.Length > 0) inventories[item] = inventory;
                }
                string receipt = Receipt();
                journal["receiptHad"] = File.Exists(receipt) ? "1" : "0";
                journal["receiptHash"] = File.Exists(receipt) ? WebUiDependencies.FileHash(receipt) : "";
                Directory.CreateDirectory(work);
                foreach (KeyValuePair<string, string> entry in inventories)
                    File.WriteAllText(Path.Combine(work, entry.Key + ".inventory"), entry.Value, new UTF8Encoding(false));
                if (File.Exists(receipt)) File.Copy(receipt, Path.Combine(work, "receipt.txt"));
                Save();
                WebUiDependencies.AtomicWrite(pointer, token + "\n");
                try
                {
                    foreach (string item in new[] { "modules", "dist" })
                        if (journal[item + "Had"] == "1") Move(Active(item), Backup(item));
                    journal["backupsReady"] = "1";
                    Save();
                }
                catch { Rollback(); throw; }
            }
            catch { ReleaseLease(); throw; }
        }

        private WebUiPreparation(string projectRoot, string token)
        {
            root = Path.GetFullPath(projectRoot);
            work = WebUiDependencies.SafePath(root, Managed + "/" + token);
            journal = ReadJournal(Path.Combine(work, "journal.txt"));
            if (journal["schema"] != "1" || journal["token"] != token
                || (journal["phase"] != "preparing" && journal["phase"] != "committed" && journal["phase"] != "rolled-back"))
                throw new IOException("WebUI 恢复记录格式无效，已保留全部文件。");
        }

        internal static void Recover(string projectRoot)
        {
            using (FileStream operation = AcquireLease(projectRoot)) RecoverCore(projectRoot);
        }

        private static void RecoverCore(string projectRoot)
        {
            string pointer = WebUiDependencies.SafePath(projectRoot, Managed + "/current.txt");
            if (!File.Exists(pointer)) return;
            string token = File.ReadAllText(pointer, Encoding.UTF8).Trim();
            Guid parsed;
            if (token.Length != 32 || !Guid.TryParseExact(token, "N", out parsed))
                throw new IOException("WebUI 恢复标识无效，已保留全部文件。");
            WebUiPreparation previous = new WebUiPreparation(projectRoot, token);
            if (previous.journal["phase"] == "committed") previous.CleanupCommitted();
            else if (previous.journal["phase"] == "rolled-back") { previous.ClearPointer(); previous.finished = true; }
            else previous.Rollback();
        }

        public void Commit()
        {
            if (finished) return;
            if (WebUiDependencies.DependencySignature(root) != journal["inputs"])
                throw new IOException("WebUI 依赖输入在准备期间发生变化，原目录将恢复，请重新准备。");
            // The caller must finish installation, native loading and build
            // validation before this point. Only cleanup may follow commit.
            journal["phase"] = "committed";
            try { Save(); }
            catch { journal["phase"] = "preparing"; throw; }
            finished = true;
            try { CleanupCommitted(); }
            catch (Exception error) { CleanupWarning = "WebUI 已准备完成，旧备份暂时保留：" + error.Message; }
            finally { ReleaseLease(); }
        }

        public void Rollback()
        {
            if (finished) return;
            if (journal["phase"] == "committed") { finished = true; return; }
            // Preflight every original before moving any path. Unexpected
            // edits in a backup are preserved, never recursively discarded.
            string savedReceipt = Path.Combine(work, "receipt.txt");
            if (journal["receiptHad"] == "1" && (!File.Exists(savedReceipt)
                || WebUiDependencies.FileHash(savedReceipt) != journal["receiptHash"]))
                throw new IOException("WebUI 输入回执备份发生变化，已保留恢复记录。");
            foreach (string item in new[] { "modules", "dist" })
            {
                if (journal[item] != "1") continue;
                string backup = Backup(item);
                if (WebUiDependencies.Exists(backup) && WebUiDependencies.HashText(Inventory(backup)) != journal[item + "Hash"])
                    throw new IOException("WebUI 原备份发生变化，已保留现场：" + backup);
                if (journal[item + "Had"] == "1" && journal["backupsReady"] == "1" && !WebUiDependencies.Exists(backup)
                    && (!WebUiDependencies.Exists(Active(item)) || WebUiDependencies.HashText(Inventory(Active(item))) != journal[item + "Hash"]))
                    throw new IOException("WebUI 原备份缺失，已保留当前内容：" + backup);
            }
            foreach (string item in new[] { "modules", "dist" })
            {
                if (journal[item] != "1") continue;
                string active = Active(item), backup = Backup(item);
                bool moved = WebUiDependencies.Exists(backup);
                bool alreadyRestored = !moved && journal[item + "Had"] == "1";
                if (!alreadyRestored && WebUiDependencies.Exists(active) && (moved || journal["backupsReady"] == "1"))
                    Move(active, Path.Combine(work, "failed-" + item));
                if (moved) Move(backup, active);
            }
            string receipt = Receipt();
            if (journal["receiptHad"] == "1")
            {
                if (!File.Exists(receipt) || WebUiDependencies.FileHash(receipt) != journal["receiptHash"])
                {
                    if (File.Exists(receipt)) Move(receipt, Path.Combine(work, "failed-receipt.txt"));
                    File.Copy(savedReceipt, receipt);
                }
            }
            else if (File.Exists(receipt)) Move(receipt, Path.Combine(work, "failed-receipt.txt"));
            journal["phase"] = "rolled-back";
            Save();
            ClearPointer();
            finished = true;
            ReleaseLease();
            // Failed replacement trees stay quarantined for inspection. They
            // may contain an unexpected user write made during an async npm
            // command; rollback must not silently delete that content.
        }

        private void CleanupCommitted()
        {
            foreach (string item in new[] { "modules", "dist" })
            {
                string backup = Backup(item);
                if (!WebUiDependencies.Exists(backup)) continue;
                if (!BackupUnchanged(item, journal[item + "Deleting"] == "1"))
                    throw new IOException("旧 WebUI 备份发生变化，未自动清理：" + backup);
            }
            foreach (string item in new[] { "modules", "dist" })
                if (WebUiDependencies.Exists(Backup(item)))
                {
                    journal[item + "Deleting"] = "1";
                    Save();
                    DeleteOwnedTree(Backup(item));
                }
            if (File.Exists(Path.Combine(work, "receipt.txt"))) File.Delete(Path.Combine(work, "receipt.txt"));
            ClearPointer();
            if (File.Exists(Path.Combine(work, "journal.txt"))) File.Delete(Path.Combine(work, "journal.txt"));
            foreach (string item in new[] { "modules", "dist" })
                if (File.Exists(Path.Combine(work, item + ".inventory"))) File.Delete(Path.Combine(work, item + ".inventory"));
            if (Directory.GetFileSystemEntries(work).Length == 0) Directory.Delete(work);
            finished = true;
        }

        private void ClearPointer()
        {
            string pointer = WebUiDependencies.SafePath(root, Managed + "/current.txt");
            if (!File.Exists(pointer) || File.ReadAllText(pointer, Encoding.UTF8).Trim() != journal["token"])
                throw new IOException("WebUI 恢复标识已变化，未覆盖其他操作。");
            File.Delete(pointer);
        }

        private string Active(string item) { return WebUiDependencies.SafePath(root, item == "modules" ? "webui/node_modules" : "webui/dist"); }
        private string Backup(string item) { return WebUiDependencies.SafePath(root, Managed + "/" + journal["token"] + "/original-" + item); }
        private string Receipt() { return WebUiDependencies.SafePath(root, Managed + "/dependency-inputs.txt"); }
        private bool BackupUnchanged(string item, bool allowSubset)
        {
            string inventoryPath = Path.Combine(work, item + ".inventory");
            if (!File.Exists(inventoryPath) || WebUiDependencies.FileHash(inventoryPath) != journal[item + "Hash"]) return false;
            string actual = Inventory(Backup(item));
            if (!allowSubset) return WebUiDependencies.HashText(actual) == journal[item + "Hash"];
            HashSet<string> expected = new HashSet<string>(File.ReadAllLines(inventoryPath, Encoding.UTF8), StringComparer.Ordinal);
            foreach (string row in actual.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries))
                if (!expected.Contains(row)) return false;
            return true;
        }
        private void Save()
        {
            StringBuilder text = new StringBuilder();
            foreach (KeyValuePair<string, string> entry in journal) text.Append(entry.Key).Append('=').Append(entry.Value).Append('\n');
            WebUiDependencies.AtomicWrite(Path.Combine(work, "journal.txt"), text.ToString());
        }

        private static Dictionary<string, string> ReadJournal(string path)
        {
            FileInfo file = new FileInfo(path);
            if (!file.Exists || file.Length > 8192) throw new IOException("WebUI 恢复记录缺失或过大。");
            Dictionary<string, string> value = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (string line in File.ReadAllLines(path, Encoding.UTF8))
            {
                if (line.Length == 0) continue;
                int separator = line.IndexOf('=');
                if (separator <= 0 || value.ContainsKey(line.Substring(0, separator))) throw new IOException("WebUI 恢复记录无效。");
                value.Add(line.Substring(0, separator), line.Substring(separator + 1));
            }
            foreach (string key in new[] { "schema", "token", "phase", "inputs", "backupsReady", "modules", "modulesHad", "modulesHash", "modulesDeleting", "dist", "distHad", "distHash", "distDeleting", "receiptHad", "receiptHash" })
                if (!value.ContainsKey(key)) throw new IOException("WebUI 恢复记录缺少字段：" + key);
            foreach (string key in new[] { "backupsReady", "modules", "modulesHad", "modulesDeleting", "dist", "distHad", "distDeleting", "receiptHad" })
                if (value[key] != "0" && value[key] != "1") throw new IOException("WebUI 恢复标记无效。");
            return value;
        }

        internal static void ValidateInputs(string projectRoot, string signature)
        {
            string pointer = WebUiDependencies.SafePath(projectRoot, Managed + "/current.txt");
            if (!File.Exists(pointer)) return;
            string token = File.ReadAllText(pointer, Encoding.UTF8).Trim();
            Guid parsed;
            if (token.Length != 32 || !Guid.TryParseExact(token, "N", out parsed)) throw new IOException("WebUI 准备标识无效。");
            WebUiPreparation current = new WebUiPreparation(projectRoot, token);
            if (current.journal["phase"] != "preparing" || current.journal["inputs"] != signature)
                throw new IOException("WebUI 依赖输入在准备期间发生变化，不能记录为已验证。");
        }

        private static void Move(string source, string destination)
        {
            if (WebUiDependencies.Exists(destination)) throw new IOException("WebUI 恢复目标已存在，未覆盖：" + destination);
            if ((File.GetAttributes(source) & FileAttributes.Directory) != 0) Directory.Move(source, destination);
            else File.Move(source, destination);
        }

        private static string Inventory(string path)
        {
            StringBuilder value = new StringBuilder();
            AppendFingerprint(path, "", value);
            return value.ToString();
        }

        private static void AppendFingerprint(string path, string relative, StringBuilder value)
        {
            FileAttributes attributes = File.GetAttributes(path);
            bool directory = (attributes & FileAttributes.Directory) != 0;
            value.Append(Convert.ToBase64String(Encoding.UTF8.GetBytes(relative))).Append('|').Append((int)attributes).Append('|');
            if (!directory) value.Append(new FileInfo(path).Length);
            // Directory mtimes change on rename. Record file metadata and link
            // entries without dereferencing normal pnpm junction targets.
            if (!directory) value.Append('|').Append(File.GetLastWriteTimeUtc(path).Ticks);
            value.Append('\n');
            if (!directory || (attributes & FileAttributes.ReparsePoint) != 0) return;
            string[] children = Directory.GetFileSystemEntries(path);
            Array.Sort(children, StringComparer.OrdinalIgnoreCase);
            foreach (string child in children) AppendFingerprint(child, relative + "/" + Path.GetFileName(child), value);
        }

        private static void DeleteOwnedTree(string path)
        {
            FileAttributes attributes = File.GetAttributes(path);
            if ((attributes & FileAttributes.Directory) == 0) { File.Delete(path); return; }
            if ((attributes & FileAttributes.ReparsePoint) == 0)
                foreach (string child in Directory.GetFileSystemEntries(path)) DeleteOwnedTree(child);
            Directory.Delete(path, false);
        }

        private static FileStream AcquireLease(string projectRoot)
        {
            string path = WebUiDependencies.SafePath(projectRoot, Managed + "/operation.lock");
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            try { return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None); }
            catch (IOException error) { throw new IOException("另一个启动器正在准备 WebUI，请等待该操作结束。", error); }
        }

        private void ReleaseLease() { if (lease != null) { lease.Dispose(); lease = null; } }
        public void Dispose() { try { if (!finished) Rollback(); } finally { ReleaseLease(); } }
    }
}
