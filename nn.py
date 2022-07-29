# This was taken from the OpenAI repository
# I plan to reimplement this later when I have more time


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
