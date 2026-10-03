"""Preserved latent interpolation experiment; the old CLI was a placeholder."""


def main(model_class_name=None, saved_models_path=None, file_one=None, file_two=None):
    raise NotImplementedError("The original latent CLI did not implement an image workflow. ME latent editing is not available.")


def interpolate_from_a_to_b_for_c(model, X, labels, a=None, b=None, x_c=None, alpha=0):
    z_a = model.encode(X[labels == a])
    z_b = model.encode(X[labels == b])
    direction = z_a.mean(axis=0) - z_b.mean(axis=0)
    return model.decode(model.encode(x_c) + alpha * direction)
