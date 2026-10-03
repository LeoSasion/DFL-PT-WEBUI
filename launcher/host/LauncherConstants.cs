using System;

namespace DflPtWebUi.Launcher
{
    internal static class LauncherConstants
    {
        public const string ProductName = "DFL-PT-WEBUI Launcher";
        // A new local repository has no published source or release channel.
        public static readonly bool OnlineUpdatesEnabled = false;
        public const string GitRemote = "";
        public const string GitFallbackMirror = "";
        public const string GitBranch = "main";
        public const string LauncherUpdateManifestGitHub =
            "";
        public const string LauncherUpdateManifestGitee =
            "";
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
        public static readonly string SettingsDirectory = System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "DFL-PT-WEBUI",
            "Launcher");
    }
}
