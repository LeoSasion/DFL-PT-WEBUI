import releaseVersion from "../../../release/version.json" with { type: "json" };

export const applicationVersion = releaseVersion.version;
export const launcherVersion = releaseVersion.launcherVersion;
