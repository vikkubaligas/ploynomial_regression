Polynomial Regression

Machine learning: Polynomial regression using numpy, training loop,
weight updating via gradient.

Using only numpy, this code shows how polynomial regression works. The
model takes the degree and learning rate as the inputs. Using weights W,
as the coefficients of the polynomial equation:

$$Y\  = \ {x^{0}w}_{0}\  + \ x^{1}w_{1}\  + x^{2}w_{2}\  + \ x^{3}w_{3}\ldots(1)$$

The model tries to arrive at the correct weights, which produce the same
result as the actual results. It must be noted here that $w_{0}\ $ is
the bias parameter in our typical polynomial equation. Instead of using
a bias parameter separately, we have used it as ${x^{0}w}_{0}$ , where
$x^{0}$ is 1. This is the same as all other notations.

At each iteration, the weights are updated using gradient descent
algorithm. The weight updating logic is as follows:

$$w_{i} = \ w_{i} - \left( l*\ \frac{\partial C}{\partial\ w_{i}}\  \right)\ldots(2)$$

$l$ = Learning rate, typically 0.0001, determines how much to update the
weight.

$C$ = Cost, is the amount of farness of the predicted output from the
actual output.

$\frac{\partial C}{\partial\ w_{i}}$ = Gradient of the cost function
with respect to that particular weight.

The cost function used here is the Mean square error, which can be
different based on user requirement. When a different cost function is
used, the entire code changes, as the gradient of cost function changes.
However, the weight updating logic, (2) remains the same.

The intuition of weight updating is simple. Gradient of a function is
nothing but the partial derivative of the function, treating all other
variables as constant, except the one we are interested in. Here, in our
case, gradient is taken with respect to the weight parameter. Keeping
that particular weight parameter as the variable and treating the rest
as (input x, as well as rest of the weights w) as constants, derivative
of the cost function gives the gradient.

This gradient indicates the direction towards which the cost function is
moving at the given combination of weights. So, intuitively, the
direction of the cost function must be towards zero, since 0 cost
indicates that the predicted output is the same as the required/actual
output.

Since we have calculated the direction towards which the cost function
is moving for the given combination of the weights, we need to modify
each of the weights such that the gradient moves towards zero, or, to be
clearer, modify weights in the negative direction of the gradient.

When weights are modified in the negative direction of the gradient, the
new cost which uses the updated weights, will be lesser comparatively.

It might naturally occur to us that we can get the weights in a single
step, by reducing/incrementing the weight as much as the gradient, so
that the next combination will bring the cost to zero.

The important practical aspect that comes to picture at this point of
time, is the learning rate. The gradient, even though it gives the
direction towards which the cost is moving, does not give the amount the
cost is away from zero. It must be noted that it gives only the
direction of movement of cost. Not the amount the cost has already
moved.

So, the weight must be updated by a small amount in the direction of the
gradient. If the learning rate is too small, the number of iterations to
arrive at the zero cost will be too large. But, at the same time, if the
learning rate is higher, the next combination will cross the optimal
combination. It will result in a gradient in the opposite direction and
the next update will again result in crossing the zero-cost point and
the weights will never converge. The cost will keep on oscillating from
one direction to another, without becoming zero.

The other practical aspect to note here is that cost might not become
perfectly zero. The ideal condition is for the cost to become zero, but
in practise the cost might arrive at a minimal non zero point. The
objective of the algorithm will be to find the point at which the cost
becomes minimum.

It must be noted here that the point at which the cost is minimum, is
the point at which the slope of the curve of the cost function becomes
zero. Since slope of a curve is nothing but the derivative of the curve,
it implies that the gradient is also zero. Our previous intuition of
updating the weight in the direction of gradient matches this statement.
When the slope is zero, the gradient is zero. When gradient is zero, the
weights are not updated in (2).

Back to mathematics, we define error as the simple difference between
the actual output and the predicted output.

$$E = Y - y\ldots(3)$$

The cost, MSE in our case, is defined as

$$C = \ \frac{\sum_{}^{}E^{2}}{N}\ldots(4)$$

Where N is the number of input samples. At the application level, after
we have attained the optimal weights, the number of samples we input to
our model to get the predictions is N.

The predicted output y is a scalar. A single floating-point number. The
required/actual output Y is also a single floating-point number. Thus,
if we provide N number of inputs to the model, N number of errors will
be produced. These are the errors whose mean we are calculating in (4)
as the cost function.

Now that the notations are clear, we will rewrite the equations, so as
to match the above situation where we will provide N inputs to the
model.

$$Y_{j}\  = \ {x_{j}^{0}w}_{0}\  + \ x_{j}^{1}w_{1}\  + x_{j}^{2}w_{2}\  + \ x_{j}^{3}w_{3}\ldots(5)$$

Where $j$ indicates the $j^{th}$ input among N inputs.

Now we can re write (3) and (4)

$$E_{j} = Y_{j} - y_{j}\ $$ … (6)

It must be noted that $i$ is exclusively used to subscript the weights.
For 3<sup>rd</sup> degree polynomial, $i\ $will range from 0 to 3 as
shown in (1) and (5). $i$ and $j$ must NOT be confused NOR be used
interchangeably.

$$C = \ \frac{1}{N}\sum_{j = 1}^{N}{E_{j}}^{2}\ldots(7)$$

So, cost C is calculated once for one set of inputs and weights are
updated once. When it comes to updating the weights, the gradient of
this cost is calculated with respect to each of the weights and each of
the weights is updated according to the calculated gradient. This
process is indicated in (2).

The process (2) is repeatedly calculated thousands of times for each of
the weights, so that the combination of the weights converges to the
optimal combination which produces minimum cost. This repeated
calculation and updating is called training the network.

Getting back to calculus. The gradient that we’re talking about from the
beginning of this post is not yet calculated. We’ll do it now. Expanding
(7)

$$C = \ \frac{1}{N}\ \sum_{j = 1}^{N}{(\ \ Y_{j} - \left( {x_{j}^{0}w}_{0}\  + \ x_{j}^{1}w_{1}\  + x_{j}^{2}w_{2}\  + \ x_{j}^{3}w_{3} \right)\ )}^{2}\ldots(8)$$

Here $x_{j}$ is the $j^{th}$ input among N inputs.

This cost function must be differentiated (degree+1) number of times to
get (degree+1) number of gradients that will be used to update
(degree+1) number weights. If the degree of the polynomial is 3, as
shown in (8), then 4 gradients will be calculated and 4 weights will be
updated for one set of N inputs. This process, of updating 4 weights,
will be repeated thousands of times for all 4 of them to arrive at the
optimal combination which produces minimum cost.

For $w_{0}$, $\ $all other variables except $w_{0}\ $are treated as
constants. So, derivative of (8) with respect to $w_{0}$ will simply be:

$$\frac{\partial C}{\partial\ w_{0}} = \ \frac{\partial\ }{\partial\ w_{0}}\ \left( \frac{1}{N}\ \sum_{j = 1}^{N}{(\ Y_{j} - ({x_{j}^{0}w}_{0}\  + \ K1\ )}^{2} \right)$$

Where K2 is the constant,
$x_{j}^{1}w_{1}\  + x_{j}^{2}w_{2}\  + \ x_{j}^{3}w_{3}$

The other weights are constant with respect to $w_{0}$. It means that
the change in one weight does not affect the other weights. Change in
any of the weights does not affect the other weights. A change in any of
the weights, affects only the parameters like error, cost and gradient.

$$\frac{\partial C}{\partial\ w_{0}} = \ \ \frac{1}{N}\ \sum_{j = 1}^{N}{- 2x_{j}^{0}\ (Y_{j}\  - \ ({x_{j}^{0}w}_{0}\  + \ K1\ ))}$$

Here we can notice that $x_{j}^{0}$ is 1, and the other part of the
summation is the error of the prediction. So, the above equation can be
simplified to

$$\frac{\partial C}{\partial\ w_{0}} = \ \ \frac{- 2\ }{N}\ \sum_{j = 1}^{N}{\left( E_{j} \right)\ }\ldots(9)$$

Similarly, gradient with respect to the next weight $w_{1}$ is shown
below.

$$\frac{\partial C}{\partial\ w_{1}} = \ \frac{\partial\ }{\partial\ w_{1}}\ \left( \frac{1}{N}\ \sum_{j = 1}^{N}{(\ Y_{j} - (K1\  + x_{j}^{1}w_{1} + \ K2)}^{2} \right)$$

$$\frac{\partial C}{\partial\ w_{1}} = \ \ \frac{1}{N}\ \sum_{j = 1}^{N}{- 2x_{j}^{1}\ (Y_{j}\  - \ (K1\  + x_{j}^{1}w_{1} + \ K2))}$$

This equation again simplifies when we notice that the second term is
nothing but the error of the prediction. The final gradient with respect
to weight $w_{1}$ is:

$$\frac{\partial C}{\partial\ w_{1}} = \ \ \frac{- 2\ }{N}\ \sum_{j = 1}^{N}{\left( {x_{j}E}_{j} \right)\ }\ldots(10)$$

Using the above method, we can calculate the gradient for all weights
and can write a generalised equation exclusively for polynomial
regressions that use MSE as the cost function:

$$\frac{\partial C}{\partial\ w_{i}} = \ \ \frac{- 2\ }{N}\ \sum_{j = 1}^{N}{\left( {x_{j}^{i}E}_{j} \right)\ }\ldots(11)$$

Here $x_{j}^{i}$ is the $i^{th}$ power of $j^{th}$ input. Where $i$
ranges from 0 to degree of polynomial and j ranges from 1 to number of
inputs/outputs/predictions of the regression model. $i$ and $j$ must NOT
be confused NOR be used interchangeably.

Finally using equation (11) in our main weight updating equation (2) we
arrive at

$$w_{i} = \ w_{i} - \left( \ \frac{- 2l\ }{N}\ \sum_{j = 1}^{N}{\left( {x_{j}^{i}E}_{j} \right)\ }\  \right)\ldots(12)$$

This becomes our main equation does all the magic inside out regression
model, which is used in the code.
