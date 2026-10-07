export function sameIdentityReferences(left = [], right = []) {
  const a = new Set(left), b = new Set(right);
  return a.size === b.size && [...a].every(member => b.has(member));
}

export function bestFacesetIdentityContract(plan, references, confirmed) {
  const expectedReferences = confirmed ? [...references] : [];
  const dirty = Boolean(plan) && !sameIdentityReferences(expectedReferences, plan.identityReferences ?? []);
  return { expectedReferences, dirty,
    canPublish: Boolean(plan?.selection) && plan?.state === "finalized" && plan?.qualityModel === "foreground_tenengrad"
      && !plan?.publication && !dirty };
}
