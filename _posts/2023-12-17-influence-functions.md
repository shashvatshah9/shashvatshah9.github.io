# Influence Functions
## Introduction
It's a classic technique from robust statistics (1), to understand and improve machine learning models. By tracing a model's prediction back to its training data, influence functions determine the training points most responsible for a given prediction. This information can be used to improve the model by selecting and removing noisy or irrelevant data and to debug models by identifying errors in the training data or the model's assumptions. Overall, influence functions are a powerful tool for understanding and improving machine learning models (2). They are relatively easy to compute and can be used with linear and non-linear models, making them increasingly popular in machine learning research and practice. 

Hampel first introduced influence functions in (3). In his work, some useful robustness measures were derived like gross error sensitivity. So we shall inspect the derivations of Influence Functions (IF) first, derive some useful properties they exhibit, and then lay out some of the related work that has been done concerning statistical modeling, and how it is being extended to the realm of machine learning and a specific topic within machine learning which is called Adversarial training, and how Influence Functions can be leveraged to make machine learning models robust from noise or perturbations in data.

It can be used to trace a model’s prediction through the learning algorithm and back to its training data, thereby identifying training points most responsible for a given prediction. Influence functions are used to understand how a statistical model or a machine learning model makes predictions. By tracing the predictions back to the training data, influence functions can identify which training points are most important for a given prediction, and how the model would change if those points were different. This information can be used to improve the model by identifying and removing noisy or irrelevant data and to debug models by identifying errors in the training data or the 
model's assumptions. Influence functions are a powerful tool for understanding and improving machine learning models. They are relatively easy to compute and can be used with linear and non-linear models. As a result, influence functions are becoming increasingly popular in machine learning research and practice.

The influence function, by definition, exhibits a close mathematical relationship with leave-one-out criteria in statistical learning. This criterion is frequently employed to evaluate the generalizability of a method. In statistical analysis, the influence function serves a similar purpose, being utilized to examine the statistical efficiency of a method. Connections between these realms are investigated, where the influence function is mathematically linked to the initial term of a Taylor expansion. Traditionally they were not used very much in machine learning and were only utilized in methods of model selection for example in Linear Regression or Kernel based models (4).

A training-set attack is an attempt to fool a machine-learning model by adding adversarial examples to the training set. Adversarial examples are carefully crafted inputs that are designed to cause the model to make an incorrect prediction. Influence functions can be used to defend against training-set attacks by identifying the training points that are most vulnerable to attack. Removing these points from the training set makes the model more robust to adversarial examples.

## Derivation
Let T be a domain, and T is Gateaux differentiable at the distribution F in the domain (T) if there exists a real function $a_1$ such that for all G in the domain (T) it holds that
\begin{equation}
    \underset{t \rightarrow 0}{lim} \dfrac{T((1-t)F + tG) - T(F)}{t} = \int a_1(x)dG(x)
\end{equation}
which can also be written as 

\begin{equation}
    \frac{\partial}{\partial t} [T((1-t)F + tG)]_{t=0} = \int a_1(x)dG(x)
\end{equation}

Next  Hampel pointed out that the importance of the influence function lies in its heuristic interpretation, so we look at the effect of infinitesimal contamination at a point standardized by the mass of the contamination. In literature, influence functions are sometimes referred to by the notation $\Omega (x; T, F)$. If some distribution G is near F, then the first-order von Mises expansion of T at F can be evaluated using Taylor series expansion as 

\begin{equation}
    T(G) = T(F) + \int IF(x; T, F)d(G-F)(x) + remainder
\end{equation}

If we have n i.i.d observations according to F, then the empirical distribution $F_N$ will tend to F Glivenko-Cantelli theorem. By substituting G in the above equation with $F_n$ for a sufficiently large n, the integral evaluates to 

\begin{equation}
    \sqrt{n}(T_n - T(F)) \simeq \frac{1}{\sqrt{n}} \sum_{i=1}^n IF(X_i; T, F) + remainder
\end{equation}

The first term on the right-hand side becomes asymptotically normal and the remainder converges to 0 for $n \rightarrow 0$. 

Having defined the IF, we can look at the sensitivity measures that help one evaluate a statistical model from the robustness point of view. The first one of this measure is absolute of the supremum value. One can define gross error sensitivity as 

\begin{equation}
    \gamma^{*} = \underset{x}{sup}|IF(x; T, F)|
\end{equation}

This gross sensitivity measures the maximum influence a small amount of contamination of fixed size can have on the estimator. Another important measure is called local shift sensitivity, which signifies the change in estimation caused by moving slightly around the input point x. Hence it signifies the approximate and standardized value of moving locally around a point x, which is nothing by the slope of IF in that point.

\begin{equation}
    \lambda^{*} = \underset{x \neq y}{sup}\frac{|IF(y; T, F) - IF(x; T, F)|}{|y-x|}
\end{equation}

## References
1] Ronchetti E. M. Rousseeuw P. J. Hampel F. R. and Stahel. Robust Statistics: The Approach Based on Influence
Functions. Wiley, 1986. \
[2] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions, 2020.\
[3] F.R. Hampel ̇The Influence Curve and Its Role in Robust Estimation. Journal of the American Statistical Associ-
ation, 1974\
[4] Jia Li and Andrew W. Moore. Model selection in kernel based regression using the influence function(special topic
on model selection). Journal of Machine Learning Research, 9
