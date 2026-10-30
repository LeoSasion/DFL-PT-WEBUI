using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Threading;

namespace DflPtWebUi.Launcher {
    internal static class LauncherConstants {
        public const string ProductName = "DFL-PT-WEBUI Launcher";
        public static readonly bool LauncherSelfUpdatesEnabled = true;
        public static string SettingsDirectory { get { return Environment.GetEnvironmentVariable("DFL_UPDATE_TEST_SETTINGS"); } }
        public const string LauncherUpdateManifestGitHub = "https://raw.githubusercontent.com/LeoSasion/DFL-PT-WEBUI/launcher-updates/launcher-update.json";
        public const string LauncherUpdateManifestGitee = "https://gitee.com/LeoSasion/DFL-PT-WEBUI/raw/launcher-updates/launcher-update.json";
    }
    static class UpdateHarness {
        [STAThread]
        static int Main(string[] args) {
            if(args.Length>1 && args[0]=="--download") {
                try {
                    var manifest=LauncherUpdateManifest.Parse(File.ReadAllText(args[1]));
                    string first=LauncherSelfUpdater.DownloadUpdateAsync(manifest,null).GetAwaiter().GetResult();
                    string second=LauncherSelfUpdater.DownloadUpdateAsync(manifest,null).GetAwaiter().GetResult();
                    if(first!=second)return 10;
                    Console.WriteLine(first);return 0;
                }catch(Exception error){Console.Error.WriteLine(error);return 9;}
            }
            if(args.Length>0 && args[0]=="--old") { Thread.Sleep(100);return 0; }
            if(args.Length>1 && args[0]=="--apply-update") {
                try {
                    typeof(LauncherSelfUpdater).GetMethod("ApplyUpdate",BindingFlags.Static|BindingFlags.NonPublic).Invoke(null,new object[]{args[1]});
                    return 0;
                }catch(Exception error){Console.Error.WriteLine(error);return 7;}
            }
            // This private process represents the new window's readiness boundary.
            // Production emits the same receipt only after WebView navigation and native state delivery.
            if(Environment.GetEnvironmentVariable("DFL_UPDATE_TEST_FAIL_START")=="1")return 8;
            LauncherSelfUpdater.ScheduleCleanup(args);
            Thread.Sleep(2500);
            return 0;
        }
    }
}
