using System;
using System.Collections.Generic;
using System.IO;
using System.Web.Script.Serialization;

namespace DflPtWebUi.Launcher
{
    internal static class RuntimeServiceIdentity
    {
        public const int MaxHealthBytes = 65536;

        public static bool SameProjectPath(string actual, string expected)
        {
            try
            {
                if (String.IsNullOrWhiteSpace(actual) || String.IsNullOrWhiteSpace(expected)
                    || !Path.IsPathRooted(actual) || !Path.IsPathRooted(expected)
                    || Path.GetPathRoot(actual).Length < 3 || Path.GetPathRoot(expected).Length < 3
                    || actual.Length < 3 || expected.Length < 3
                    || (actual[1] == ':' && actual[2] != '\\' && actual[2] != '/')
                    || (expected[1] == ':' && expected[2] != '\\' && expected[2] != '/')) return false;
                return String.Equals(Path.GetFullPath(actual).TrimEnd('\\', '/'),
                    Path.GetFullPath(expected).TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase);
            }
            catch { return false; }
        }

        public static bool Matches(string json, string projectRoot)
        {
            try
            {
                if (String.IsNullOrWhiteSpace(json) || json.Length > MaxHealthBytes) return false;
                JavaScriptSerializer serializer = new JavaScriptSerializer
                {
                    MaxJsonLength = MaxHealthBytes,
                    RecursionLimit = 16
                };
                IDictionary<string, object> health = serializer.DeserializeObject(json) as IDictionary<string, object>;
                if (health == null) return false;
                object value;
                if (health.TryGetValue("ok", out value) && !(value is bool && (bool)value)) return false;
                if (health.TryGetValue("data", out value)) health = value as IDictionary<string, object>;
                if (health == null || !health.TryGetValue("service", out value)
                    || !String.Equals(value as string, "DFL-PT-WEBUI Local Runtime", StringComparison.Ordinal)) return false;
                if (!health.TryGetValue("runtime", out value)) return false;
                IDictionary<string, object> runtime = value as IDictionary<string, object>;
                if (runtime == null || !runtime.TryGetValue("current", out value)) return false;
                IDictionary<string, object> current = value as IDictionary<string, object>;
                return current != null && current.TryGetValue("dflRoot", out value)
                    && SameProjectPath(value as string, Path.Combine(projectRoot, "_internal", "DeepFaceLab"));
            }
            catch { return false; }
        }
    }
}
