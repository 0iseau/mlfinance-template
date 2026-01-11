"""Demonstration of Random Forest Regression methodology.

This script shows a minimal explanation of Random Forest Regression, to illustrate
how sklearn's RandomForestRegressor function can be used for regression tasks.

RF reggression is a robust method to forecast y(t+1) at yt using past data to "train".

Methdology :
    Let's say we have a dataset with prices, dates, and 2 indicators computed
    on this dataset : RSI and MACD. The dataset contains 30 rows (indicators ignore
    early data before computing).

    1) Data preparation :
        We want to explain with RSI and MACD how each row went from yk to y(k+1).
        To do this, we need to prepare our data by assigning at t=k : RSI_k,
        MACD_k and return at k+1. Basically, my data now tells : at t=k, with
        these values of indicators, the price varied by x% by tomorrow. If today is t=p,
        and
        we want to predict at p+1, we extract row p from dataset, as p+1 data is unknown.

    2) Bootstrap :
        Randomly create n subsets of 30 rows. Fully independent assignation :
        no time succession and subsets can contain several times the same rows,
        or some rows could be ignored. In our example, we create only 3 subsets.
        We then define a way to quantify the error of forecasted y for all subsets, for
        example using MSE.

    3) Tree training :
        We want to split each subset in two, using a specific rule that minimizes our error
        quantification (MSE for this example) in the new subsets. In practice, we sort the rows
        according to the RSI and MACD values and try all possible splits between each row, keeping
        the one with the highest error reduction. We then split our data and repeat the process with
        the new splitted subsets, until MSE is no more optimized or subsets are too small.

    4) Prediction :
        The final subsets of a tree are called leaves. A leaf predicts leaf_y(t+1) = mean(y), we
        call it leaf value. To predict y(p+1), we drop yp inside each tree, and extract the leaf
        value it falls into. This is a tree's prediction. Our final forest estimation is the mean of
        all tree predictions.

"""
