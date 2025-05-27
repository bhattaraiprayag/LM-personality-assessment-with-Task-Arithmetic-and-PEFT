# src/utils/helper.py
"""
Utility module containing helper functions.
"""
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_output(*args, **kwargs) -> None:
    """
    Prints messages only from the process with rank zero to avoid
    duplicated logs while distributed training.

    Args:
        *args: Variable length argument list to pass to print.
        **kwargs: Arbitrary keyword arguments to pass to print.
    """
    print(*args, **kwargs)
