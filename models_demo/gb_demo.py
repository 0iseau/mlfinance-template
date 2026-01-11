"""Demonstration of Gradient Boosting methodology.

This script shows a minimal explanation of Gradient Boosting, to demonstrate how
XGBoost's gradient boosting can be used in tasks for prediction.

It is a robust method to regress y(t+1) at yt using past data to "train".

Methodology :
    1) Data preparation :
        We start with our dataset with prices indexed by time. Unlike Random Forest regression,
        we keep the dataset as it is, and use as first forecast value a simple mean. We quantify
        the error with a MSE of prediction - true value (here, mean - true value). Lets call it
        R.

    2) Tree training :
        Gradient boosting splits data trying to maximize the residuals similarities. This can be
        quantified by computing for every rule the SSEs of each subset. When a rule is found,
        we predict our data by letting it be filtered by the rule. Iterate until iteration limit
        reached, or residuals SSE doenst improve or too small subsets. At the end of this step,
        each row's prediction is filtered by the tree's rules and is attributed a correction :
        Current prediction - eta*mean(leaf value), with leaf value = mean(residuals inside leaf).
        We de not apply the full correction to avoid overfitting, basically we prefer little but
        safe corrections, multiplying by a learning rate. With these new prediction values,
        we compute our new residuals for the entire dataset and re-train a tree, until it
        no longer improves erros or maximum length reached.

    3) Prediction :
        At the end, every trained tree will form a large sequence of rules. To predict on new data,
        we let all the trees filter and apply their correction. We thus obtain our final t+1
        prediction by letting the dataset's mean be adjusted by filtering our row t on every
        rule and applying the correspondings trees corrections. So prediction(t+1) = dataset mean
        + sum(trees corrections). We can iterate with the new predicte t+1 row to predict t+2, etc.

Sources :
    https://www.datacamp.com/tutorial/guide-to-the-gradient-boosting-algorithm
    ?dc_referrer=https%3A%2F%2Fwww.google.com%2F
    https://www.datacamp.com/tutorial/xgboost-in-python
"""
