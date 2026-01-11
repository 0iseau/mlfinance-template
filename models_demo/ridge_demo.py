"""Ridge Regression.

Basic linear regression with OLS computes (X'X)^-1(X'Y). X'X contains the
variance and covariance between each estimator. In case of highly correlated
indicators, the variance and covariances are very colse, creating very similar
columns inside the matrix and may lead to linearity. When performing its inverse,
the determinant becomes very small, and thus our inverse matrix can contain huge numbers.
In the regression, they cancel themselves describing a acceptable impact, and graphically,
we still have a close to the best fitting line. But the equation will containt
these huge numbers, that would make rare big deviations between previously very close
data produce too high predictions.

Ridge regression fixes this by implementing a voluntary change on the X'X variances :
(X'X) + lambda * I, before computing the inverse.
Example :
X'X_1 = |30000 29999|
        |29999 30005|

X'X_2 = |30000 29999| + |10 0| (lambda=10 * I) = |30010 29999|
        |29999 30005|   |0 10|                   |29999 30015|

(R code) :

> m1 <- matrix(c(
+   3000000, 2999999,
+   2999999, 2999998
+ ), nrow = 2, ncol = 2)

> m2 <- matrix(c(
+   3000010, 2999999,
+   2999999, 3000008
+ ), nrow = 2, ncol = 2)

> solve(m1)
         [,1]     [,2]
[1,] -2999277  2999278
[2,]  2999278 -2999279

> solve(m2)
            [,1]        [,2]
[1,]  0.05000007 -0.04999992
[2,] -0.04999992  0.05000010

Because of possibility of having estimators of different units, we need to avoid
range disparities by normalizing the data.

Implementation source :
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
"""
