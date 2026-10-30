using System;

namespace DflPtWebUi.Launcher
{
    internal static class LauncherConstants
    {
        public const string ProductName = "DFL-PT-WEBUI Launcher";
        // First installation uses the official source ZIP without requiring Git.
        // Source upgrades remain explicit; the independently pinned EXE can update itself.
        public const string ProjectHomeUrl = "https://github.com/LeoSasion/DFL-PT-WEBUI";
        public static readonly bool OnlineInstallationEnabled = true;
        public static readonly bool OnlineUpdatesEnabled = false;
        public static readonly bool LauncherSelfUpdatesEnabled = true;
        public const string GitRemote = "";
        public const string GitFallbackMirror = "";
        public const string GitBranch = "main";
        public const string LauncherUpdateManifestGitHub =
            "https://raw.githubusercontent.com/LeoSasion/DFL-PT-WEBUI/launcher-updates/launcher-update.json";
        public const string LauncherUpdateManifestGitee =
            "https://gitee.com/LeoSasion/DFL-PT-WEBUI/raw/launcher-updates/launcher-update.json";
        public const string VirtualHost = "launcher.local";
        public const string WebUiUrl = "http://127.0.0.1:4173/";
        public const string WebUiRuntimeHealthUrl = "http://127.0.0.1:4174/api/health";
        public const string RuntimeHealthUrl = "http://127.0.0.1:4174/api/health";
        public const string DefaultTerminalUrl = "ws://127.0.0.1:4185/terminal";
        public const string RequiredNodeVersion = "24.19.0";

        public static bool IsOfficialGitRemote(string value)
        {
            return OnlineUpdatesEnabled && !String.IsNullOrWhiteSpace(GitRemote)
                && String.Equals(NormalizeRemote(value), NormalizeRemote(GitRemote), StringComparison.OrdinalIgnoreCase);
        }

        private static string NormalizeRemote(string value)
        {
            string normalized = (value ?? String.Empty).Trim().TrimEnd('/');
            return normalized.EndsWith(".git", StringComparison.OrdinalIgnoreCase)
                ? normalized.Substring(0, normalized.Length - 4) : normalized;
        }

        // Keep this product's settings and payloads isolated from prior products.
        public static readonly string SettingsDirectory = ResolveSettingsDirectory();

        private static string ResolveSettingsDirectory()
        {
            // An explicit private profile also isolates payloads and WebView data.
            // Child update processes inherit it, so recovery stays in that profile.
            string value = Environment.GetEnvironmentVariable("DFL_PT_LAUNCHER_DATA_DIR");
            if (String.IsNullOrWhiteSpace(value)) return System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "DFL-PT-WEBUI", "Launcher");
            if (!System.IO.Path.IsPathRooted(value)) throw new ArgumentException("启动器独立配置目录必须是绝对路径。");
            string full = System.IO.Path.GetFullPath(value);
            if (String.Equals(full.TrimEnd('\\', '/'), System.IO.Path.GetPathRoot(full).TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("启动器独立配置目录不能是磁盘根目录。");
            return full;
        }
    }
}
