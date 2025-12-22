def rsi_from_avgs(ag: float, al: float) -> float:
    """Compute RSI from average gain and average loss.

    Parameters:
        ag (float): Average gain.
        al (float): Average loss.

    Returns:
        (float): Relative Strength Index value.
    """
    if al == 0.0:
        return 50.0 if ag == 0.0 else 100.0
    if ag == 0.0:
        return 0.0
    rs = ag / al
    return 100.0 - (100.0 / (1.0 + rs))
