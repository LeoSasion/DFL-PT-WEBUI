using System;
using System.Collections.Generic;
using System.IO;
using System.Web.Script.Serialization;

namespace DflPtWebUi.Launcher
{
    internal static class ReleaseIdentity
    {
        private static Dictionary<string, object> Read(string path)
        {
            if (!File.Exists(path) || new FileInfo(path).Length > 65536) return new Dictionary<string, object>();
            try { return new JavaScriptSerializer().DeserializeObject(File.ReadAllText(path)) as Dictionary<string, object> ?? new Dictionary<string, object>(); }
            catch { return new Dictionary<string, object>(); }
        }
        public static object Get(string root)
        {
            Dictionary<string, object> version = Read(Path.Combine(root, "release", "version.json"));
            if (version.Count == 0) version = Read(Path.Combine(LauncherPayload.GetPath("bootstrap"), "version.json"));
            Dictionary<string, object> receipt = Read(Path.Combine(root, "release", "installation.json"));
            string source = receipt.ContainsKey("sourceCommit") ? Convert.ToString(receipt["sourceCommit"]) : "";
            bool verified = System.Text.RegularExpressions.Regex.IsMatch(source, "^[0-9a-f]{40}$") && receipt.ContainsKey("archiveSha256")
                && System.Text.RegularExpressions.Regex.IsMatch(Convert.ToString(receipt["archiveSha256"]), "^[0-9a-f]{64}$");
            return new Dictionary<string, object> {
                {"applicationVersion", version.ContainsKey("version") ? version["version"] : "unknown"},
                {"launcherVersion", "0.1.3-preview"},
                {"installationSource", verified && receipt.ContainsKey("installationSource") ? receipt["installationSource"] : "local-source / unverified"},
                {"sourceCommit", verified ? source : "unverified"},
                {"archiveSha256", verified ? receipt["archiveSha256"] : null}
            };
        }
        public static object UpgradeTarget()
        {
            Dictionary<string, object> pin = Read(Path.Combine(LauncherPayload.GetPath("bootstrap"), "source-pin.json"));
            return new Dictionary<string, object> {
                {"applicationVersion", pin.ContainsKey("applicationVersion") ? pin["applicationVersion"] : "unknown"},
                {"sourceCommit", pin.ContainsKey("sourceCommit") ? pin["sourceCommit"] : "unverified"},
                {"archiveSha256", pin.ContainsKey("archiveSha256") ? pin["archiveSha256"] : "unverified"}
            };
        }
    }
}
