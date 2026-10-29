using System;
using System.IO;
using System.Threading;
using System.Reflection;
[assembly: AssemblyFileVersion("24.19.0.0")]
internal static class FakeWebUiNode
{
    private static int Main(string[] args)
    {
        string webui = Environment.CurrentDirectory;
        if (args.Length > 0 && args[0] == "-e")
        {
            string marker = Path.Combine(webui, "node_modules", "node-pty", "index.js");
            string value = File.Exists(marker) ? File.ReadAllText(marker) : "broken";
            if (value.Contains("hang")) Thread.Sleep(30000);
            if (value.Contains("broken")) { Console.Error.WriteLine("node-pty: synthetic ABI mismatch"); return 1; }
            Console.WriteLine("node-pty: OK\nvite/esbuild: OK"); return 0;
        }
        if (args.Length == 0) return 2;
        if (args[0].StartsWith("install", StringComparison.Ordinal))
        {
            string modules = Path.Combine(webui, "node_modules");
            Directory.CreateDirectory(Path.Combine(modules, "vite", "bin"));
            Directory.CreateDirectory(Path.Combine(modules, "node-pty"));
            File.WriteAllText(Path.Combine(modules, "vite", "bin", "vite.js"), "synthetic-vite");
            File.WriteAllText(Path.Combine(modules, "node-pty", "package.json"), "{}");
            File.WriteAllText(Path.Combine(modules, "node-pty", "index.js"), args[0] == "install-broken" ? "broken" : "ready-new");
            File.WriteAllText(Path.Combine(modules, "new-install.txt"), "synthetic only");
            return args[0] == "install-fail" ? 17 : 0;
        }
        if (args[0].StartsWith("build", StringComparison.Ordinal))
        {
            string dist = Path.Combine(webui, "dist", "client");
            Directory.CreateDirectory(dist);
            File.WriteAllText(Path.Combine(dist, "index.html"), "new synthetic build");
            return args[0] == "build-fail" ? 19 : 0;
        }
        return 3;
    }
}
