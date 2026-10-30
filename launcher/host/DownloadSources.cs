using System;
using System.Collections.Generic;

namespace DflPtWebUi.Launcher
{
    internal static class DownloadSources
    {
        public static string NormalizeMode(string value)
        {
            return value == "official" || value == "china" ? value : "auto";
        }

        public static IEnumerable<string> NpmRegistries(string mode)
        {
            if (mode == "china") yield return "https://registry.npmmirror.com";
            yield return "https://registry.npmjs.org";
            if (mode == "auto") yield return "https://registry.npmmirror.com";
        }

        public static void SetNodeRegistry(IDictionary<string, string> environment, string registry)
        {
            environment["NPM_CONFIG_REGISTRY"] = registry;
            environment["COREPACK_NPM_REGISTRY"] = registry;
            environment["npm_config_disturl"] = registry == "https://registry.npmmirror.com"
                ? "https://registry.npmmirror.com/-/binary/node" : "https://nodejs.org/download/release";
        }
    }
}
