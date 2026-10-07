using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace DflPtWebUi.Launcher
{
    internal static class DflEnvironment
    {
        public static IDictionary<string, string> Load(string projectRoot, LogBuffer logs)
        {
            string setenvPath = Path.Combine(projectRoot, "_internal", "setenv.bat");
            if (!File.Exists(setenvPath))
            {
                throw new FileNotFoundException("缺少项目环境脚本。", setenvPath);
            }

            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = Environment.GetEnvironmentVariable("ComSpec") ?? "cmd.exe";
            startInfo.Arguments = "/d /s /u /c call " + ProcessRunner.Quote(setenvPath) + " >nul && set";
            startInfo.WorkingDirectory = projectRoot;
            startInfo.UseShellExecute = false;
            startInfo.CreateNoWindow = true;
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            startInfo.StandardOutputEncoding = Encoding.Unicode;
            startInfo.StandardErrorEncoding = Encoding.Unicode;

            using (Process process = Process.Start(startInfo))
            {
                string output = process.StandardOutput.ReadToEnd();
                string error = process.StandardError.ReadToEnd();
                process.WaitForExit();
                if (process.ExitCode != 0)
                {
                    if (logs != null && !String.IsNullOrWhiteSpace(error))
                    {
                        logs.Add("environment", error, "error");
                    }
                    throw new InvalidOperationException("无法加载 _internal\\setenv.bat。退出码：" + process.ExitCode);
                }

                IDictionary<string, string> values = ParseOutput(output, projectRoot);
                string workspace;
                string activeId;
                if (!values.TryGetValue("WORKSPACE", out workspace)
                    || !values.TryGetValue("DFL_ACTIVE_PROJECT_ID", out activeId)
                    || !System.Text.RegularExpressions.Regex.IsMatch(activeId, "^[a-z0-9][a-z0-9-]{0,47}$"))
                {
                    throw new InvalidOperationException("项目环境没有有效的当前项目与工作区。");
                }
                string expected = Path.GetFullPath(Path.Combine(projectRoot,
                    activeId == "default" ? "workspace" : Path.Combine("workspaces", activeId)));
                if (!String.Equals(Path.GetFullPath(workspace), expected, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException("当前项目环境工作区不一致。");
                if (logs != null) logs.Add("environment", "当前项目：" + activeId + " | 工作区：" + workspace, "info");
                return values;
            }
        }

        internal static IDictionary<string, string> ParseOutput(string output, string projectRoot)
        {
            Dictionary<string, string> values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            string internalRoot = Path.Combine(projectRoot, "_internal");
            string selectedPath = null;
            string[] lines = (output ?? String.Empty).Split(
                new[] { "\r\n", "\n" },
                StringSplitOptions.RemoveEmptyEntries);
            for (int index = 0; index < lines.Length; index++)
            {
                int separator = lines[index].IndexOf('=');
                if (separator <= 0)
                {
                    continue;
                }
                string key = lines[index].Substring(0, separator);
                string value = lines[index].Substring(separator + 1);
                if (String.Equals(key, "PATH", StringComparison.OrdinalIgnoreCase))
                {
                    if (selectedPath == null
                        || value.IndexOf(internalRoot, StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        selectedPath = value;
                        values["PATH"] = value;
                    }
                    continue;
                }
                values[key] = value;
            }
            return values;
        }
    }
}
