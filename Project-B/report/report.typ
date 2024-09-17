#import "tablex.typ": *

// document setting
#set document(title: "B1 Project: Optimisation", author: "Chih Jung Lee")
// Require 20mm margins
#set page(paper: "a4", numbering: "1", margin: (x: 20mm, y: 20mm))
// require Arial, 11pt font
#set text(lang: "en", region: "UK", 
  size: 11pt, 
  hyphenate: false, 
  font: (
  "Arial",
  "Twitter Color Emoji"),
)
#set par(leading: 1em, justify: true)
#set enum(indent: 1em)
#set table(inset: 8pt)
#set heading(numbering: "1.1.")
#set math.mat(delim: "[", column-gap: 0.5em)

#show heading: it => {
  if it.numbering == none {
    block(inset: (y: 1em))[
      #text(font: ("Arial"), weight: "bold")[#it.body]
    ]
  } else {
    block(inset: (y: 0.4em))[
      #text(font: ("Arial"), weight: "bold")[#counter(heading).display() #it.body]
    ]
  }
}

#show math.equation: set block(spacing: 1em)
#show par: set block(spacing: 1.5em)
#show raw.where(block: false): box.with(
  fill: luma(240),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 2pt,
)
#show raw.where(block: true): block.with(
  fill: luma(240),
  inset: 10pt,
  radius: 6pt,
)
#show ref: it => {
  let el = it.element
  if el != none and el.func() == heading {
    numbering(
      el.numbering,
      ..counter(heading).at(el.location())
    )
  } else {
    it
  }
}
#show link: it => {
  underline(offset: 3pt, text(blue, it.body))
}
#show emph: it => {
  text(font: "Arial", style: "italic", it.body)
}
#show figure.where(kind : table): set figure.caption(position: top)

#let derive(body) = {
  block(fill:rgb(250, 250, 250),
   width: 100%,
   inset: 14pt,
   radius: 4pt,
   stroke: rgb(50, 50, 50),
   body)
  }


// Cover Page (Not counted)
#set page(numbering: none)
// title
#align(center, text(size: 24pt, font: ("Arial"))[
  *B1 Project: Optimisation*
])
#align(center, [John Lee])

// content
#align(left, text(17pt)[*Introduction*])
This is the report for B1 Engineering Computation - Project B: Optimisation for regression and classification models. The project investigates how to apply different optimisation methods for learning optimal parameters of a model that can predict a value of interest for a given input data point. A total of 6 tasks were given and completed using the MATLAB programming language with the「Statistics and Machine Learning Toolbox version 23.2」 and the 「Optimization Toolbox version 23.2」. Results shown in this report are generated using ```matlab rng(12345)``` unless otherwise stated.
#outline(title: "Content", depth: 1, indent: auto)
#pagebreak()

// Start counting page numbers
#set page(numbering: "1")
#counter(page).update(1)


// Task 1
= Task 1: Linear Regression via analytical solution to MSE <Analytical_Regression>
== Optimal Parameters <task1-optimal_param>
With 1000 training samples, the resulting $bold(w)$ and $b$ are sensible given that they roughly equal to the coefficients of the equation within the data generating function: 
$upright(y) = 1.5 + 0.6x#super[(1)] + 0.35x#super[(2)]$
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  align(horizon)[
    #figure(
      tablex(
      columns: 4,
      inset: 4pt,
      align: center,
      auto-lines: false,
      [], [$b$], [$w^((1))$], [$w^((2))$],
      hlinex(),
      [Learnt], [1.4862], [0.6083], [0.3435],
      [Actual], [1.5], [0.6], [0.35],
      ),
      kind: table,
      caption: [Training with 1000 samples],
    )
  ],
  align(horizon)[
    #figure(
      tablex(
      columns: 3,
      inset: 4pt,
      align: center,
      auto-lines: false,
      [], [Training], [Test],
      hlinex(),
      [MSE], [0.0478], [0.0495],
      ),
      kind: table,
      caption: [MSE for 1000 training samples],
    )
  ]
)

== Changing to Training Sample Size <task1-thousand_to_ten>
This set of parameters differs from the one obtained in @task1-optimal_param With too small a training sample size, there is insufficient data to accurately capture the coefficients for the linear regression.
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  align(horizon)[
    #figure(
      tablex(
      columns: 4,
      inset: 4pt,
      align: center,
      auto-lines: false,
      [], [$b$], [$w^((1))$], [$w^((2))$],
      hlinex(),
      [Learnt], [1.1395], [0.6049], [0.5678],
      [Actual], [1.5], [0.6], [0.35],
      ),
      kind: table,
      caption: [Training with 10 samples],
    )
  ],
  align(horizon)[
    #figure(
      tablex(
      columns: 3,
      inset: 4pt,
      align: center,
      auto-lines: false,
      [], [Training], [Test],
      hlinex(),
      [MSE], [0.0548], [0.0927],
      ),
      kind: table,
      caption: [MSE for 10 training samples],
    )
  ]
)

== Experimentation with Training Sample Sizes and RNG Seeds <task1-experiment>
Prior to this, everything was done with ```matlab rng(12345)```. Now,  we will vary seed values and training sample sizes to observe how the number of training samples affects `Training MSE` and `Test MSE`.\
#figure(
  tablex(
    columns: (auto, 17em, auto),
    inset: 5pt,
    align: center,
    auto-lines: false,
    hlinex(),
    [Training Samples], [Training MSE], [Test MSE],
    hlinex(stroke: 0.3pt),
    [4], [0.0083 $plus.minus$ 0.0093], [0.2205 $plus.minus$ 0.2989],
    [10], [0.0355 $plus.minus$ 0.0188], [0.0808 $plus.minus$ 0.0571],
    [20], [0.0414 $plus.minus$ 0.0147], [0.0586 $plus.minus$ 0.0068],
    [100], [0.0493 $plus.minus$ 0.0078], [0.0518 $plus.minus$ 0.0012],
    [1000], [0.0504 $plus.minus$ 0.0028], [0.0503 $plus.minus$ 0.0004],
    [10000], [0.0499 $plus.minus$ 0.0007], [0.0501 $plus.minus$ 0.0003],
    hlinex(),
  ),
  kind: table,
  caption: [Experimentation MSE Means and Standard Deviations],
)
As the training sample size increases, both `Training MSE` and `Test MSE` converge to the variance for regression target as mentioned in the data generating function ```matlab r_noise_var = 0.05```. To understand this, we firstly introduce the $epsilon$ term as random error independent of $bold(upright(X))$ with mean of zero $upright(E) [epsilon] = 0$ such that $bold(upright(y))$ can be expressed as
#align(center)[
  $bold(upright(y)) = f(bold(upright(X))) + epsilon$
]
Now we can re-express the MSE in terms of an expectation and a variance
#derive[
  $upright(M S E) = upright(E)[bold((upright(y - accent(y,hat))))^2]
  &= upright(E)[(f(bold(upright(X))) + epsilon - accent(f,hat)(bold(upright(X))))^2] \
  &= upright(E) [f(bold(upright(X)))^2 + 2epsilon f(bold(upright(X))) - 2 f(bold(upright(X))) accent(f,hat)(bold(upright(X))) + epsilon^2 - 2accent(f,hat)(bold(upright(X))) epsilon + accent(f,hat)(bold(upright(X)))^2] \
  &= upright(E) [(f(bold(upright(X))) - accent(f,hat)(bold(upright(X))))^2] + cancel(2upright(E)[epsilon]upright(E)[f(bold(upright(X)))], angle: #80deg) + upright(E)[epsilon^2] - cancel(2upright(E)[epsilon]upright(E)[accent(f,hat)(bold(upright(X)))], angle: #80deg) \
  &= upright(E) [(f(bold(upright(X))) - accent(f,hat)(bold(upright(X))))^2] + upright(E)[epsilon^2] = upright(E) [(f(bold(upright(X))) - accent(f,hat)(bold(upright(X))))^2] + (upright(E)[epsilon^2] - upright(E)[epsilon]^2) \
  &= upright(E) [(f(bold(upright(X))) - accent(f,hat)(bold(upright(X))))^2] + upright(V a r)(epsilon)$
]
/*
#derive[
  $upright(M S E) = upright(E)[bold((upright(y - accent(y,hat))))^2]
  &= upright(E)[(f(bold(upright(X))) + epsilon - accent(f,hat)(bold(upright(X))))^2] = upright(E) [(f(bold(upright(X))) - accent(f,hat)(bold(upright(X))))^2] + upright(E)[epsilon^2]\
  &= upright(E) [(f(bold(upright(X))) - accent(f,hat)(bold(upright(X))))^2] + (upright(E)[epsilon^2] - upright(E)[epsilon]^2) = upright(E) [(f(bold(upright(X))) - accent(f,hat)(bold(upright(X))))^2] + upright(V a r)(epsilon)$
]*/
Given this re-expression, we observe that as the error between prediction and target goes to zero, there remains an irreducible error source in the form of the regression target's variance, explaining the convergence to 0.05 in agreement with ```matlab r_noise_var = 0.05```.


// Task 2
= Task 2: Linear Regression via Gradient Descent <Gradient_Descent_Linear_Regression>
== Optimal Learning Rate 1 <task2-optimal_lr_1>
To find the optimal learning rate $lambda$, we keep ```matlab n_iters = 1000``` and perform the experiment for ```matlab lambdas = [0.00001 0.0001 0.001 0.01 0.1 1]```, picking the one with lowest corresponding `Training MSE` and `Validation MSE` $==>$ ```matlab lambda = 0.1```.
#figure(
  tablex(
    columns: (auto, 17em, auto),
    inset: 4pt,
    align: center,
    auto-lines: false,
    hlinex(),
    [$lambda$], [Training MSE], [Validation MSE],
    hlinex(stroke: 0.3pt),
    [0.00001], [6.5561], [6.8993],
    [0.0001],[1.2430],[1.2485], 
    [0.001], [0.1428], [0.1279],
    [0.01], [0.0472], [0.0532],
    vlinex(start: 5, end: 6, stroke: red),
    hlinex(stroke: red),
    vlinex(start: 5, end: 6, stroke: red),
    [0.1], [0.0465], [0.0528],
    hlinex(stroke: red),
    [1], [NaN], [NaN],
    hlinex(),
  ),
  kind: table,
  caption: [Training and Validation MSEs for Different Learning Rates at 1000 iterations],
)

== Test Set <task2-testing>
With the optimal parameters derived via ```matlab lambda = 0.1```, we calculate the ``` Test MSE```. There is a slight difference between the two MSE values, and this is simply a result of ```matlab rng(12345)``` "shuffling" the numbers. The MSE values obtained so far have been done with test set being created after the train-val set. The results will be different if the ordering was swapped.
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  align(horizon)[
    #figure(
      tablex(
        columns: 3,
        inset: 4pt,
        align: center,
        auto-lines: false,
        [], [Validation], [Test],
        hlinex(),
        [MSE], [0.0528], [0.0496],
      ),
      kind: table,
      caption: [200 validation and 20000 test samples],
    )
  ],
  align(horizon)[
    #figure(
      tablex(
        columns: 3,
        inset: 4pt,
        align: center,
        auto-lines: false,
        [], [Validation], [Test],
        hlinex(),
        [MSE], [0.0486], [0.0497],
      ),
      kind: table,
      caption: [Reversed order of data generation],
    )
  ]
)

== Optimal Learning Rate 2 <task2-optimal_lr_2>
Changing from ```matlab iters_total = 1000``` $arrow.r$ ```matlab iters_total = 10000```, we get ```matlab lambda = 0.01```.
#figure(
  tablex(
    columns: (auto, 17em, auto),
    inset: 4pt,
    align: center,
    auto-lines: false,
    hlinex(),
    [$lambda$], [Training MSE], [Validation MSE],
    hlinex(stroke: 0.3pt),
    [0.00001], [1.2439], [1.2495],
    [0.0001],[0.1428],[0.1279], 
    [0.001], [0.0473], [0.0532],
    vlinex(start: 4, end: 5, stroke: red),
    hlinex(stroke: red),
    vlinex(start: 4, end: 5, stroke: red),
    [0.01], [0.0465], [0.0528],
    hlinex(stroke: red),
    [0.1], [0.0465], [0.0528],
    [1], [NaN], [NaN],
    hlinex(),
  ),
  kind: table,
  caption: [Training and Validation MSEs for Different Learning Rates at 10000 iterations],
)

== Relationship between $bold(n#sub[iters])$ and $bold(lambda)$ <task2-n_and_lambda>
Based on the experiments in @task2-optimal_lr_1 and @task2-optimal_lr_2, we can see that as $n#sub[iters]$ decreases, the minimum optimal learning rate, $lambda$, increases. This is reasonable since a suitable larger $lambda$ would lead to faster convergence, thereby reducing the $n#sub[iters]$ necessary. Between the two, it is $n#sub[iters]$ that affects runtime, because it dictates the number of loops for the learning function while $lambda$ is only involved in one multiplication step within each loop. Therefore, in practice, it is preferable to use a short $n#sub[iters]$ coupled with the optimal $lambda$ so that runtime is shortened.

== Comparison with Task 1 <task2-comparison>
Running the analytical solution from Task 1 on the Task 2 training data produces the same optimal parameters, which is reasonable since Gradient Descent should converge to the exact solution.
#figure(
  tablex(
    columns: 4,
    inset: 4pt,
    align: center,
    auto-lines: false,
      [], [$b$], [$w^((1))$], [$w^((2))$],
    hlinex(),
    [Task 1], [1.4899], [0.6089], [0.3382],
    [Task 2], [1.4899], [0.6089], [0.3382],
  ),
  kind: table,
  caption: [Comparison of optimal parameters between Task 1 and Task 2],
)


// Task 3
= Task 3: Logistic Regression using Gradient Descent <Gradient_Descent_Logistic_Regression>
== Derivation of Log-Loss Gradient <task3-derivation>
#derive[
  $
  nabla#sub[$bold(theta)$] cal(L)#sub[C] &= nabla#sub[$bold(theta)$] {-y log(sigma (bold(upright(accent(x, hat)))^T bold(theta))) - (1 - y) log(1 - sigma (bold(upright(accent(x, hat)))^T bold(theta)))} \
  &= nabla#sub[$bold(theta)$] {-y log(1/(1 + exp(-bold(upright(accent(x, hat))^T bold(theta))))) - (1 - y) log(1 - 1/(1 + exp(-bold(upright(accent(x, hat))^T bold(theta)))))}\
  &= nabla#sub[$bold(theta)$] {y log(1 + exp(-bold(upright(accent(x, hat))^T bold(theta)))) - (1 - y) log(exp(-bold(upright(accent(x, hat))^T bold(theta)))/(1 + exp(-bold(upright(accent(x, hat))^T bold(theta)))))}\
  &= nabla#sub[$bold(theta)$] {y log(1 + exp(-bold(upright(accent(x, hat))^T bold(theta)))) + (1 - y) log(1 + exp(-bold(upright(accent(x, hat))^T bold(theta)))) - (1- y) log(exp(-bold(upright(accent(x, hat))^T bold(theta))))}\
  $
  $#h(34.5pt) &= nabla#sub[$bold(theta)$] {log(1 + exp(-bold(upright(accent(x, hat))^T bold(theta)))) - (1 - y) log(exp(-bold(upright(accent(x, hat))^T bold(theta))))}\
  &= nabla#sub[$bold(theta)$] {log(1 + exp(-bold(upright(accent(x, hat))^T bold(theta)))) + (1 - y) bold(upright(accent(x, hat))^T bold(theta))}\
  &= -exp(-bold(upright(accent(x, hat))^T bold(theta)))/(1 + exp(-bold(upright(accent(x, hat))^T bold(theta)))) bold(upright(accent(x, hat))) + (1 - y) bold(upright(accent(x, hat)))\
  &= ((-1 - exp(-bold(upright(accent(x, hat))^T bold(theta))))/(1 + exp(-bold(upright(accent(x, hat))^T bold(theta)))) + 1/(1 + exp(-bold(upright(accent(x, hat))^T bold(theta))))) bold(upright(accent(x, hat))) + (1 - y) bold(upright(accent(x, hat)))\
  &= (-1 + accent(y, macron)) bold(upright(accent(x, hat))) + (1 - y) bold(upright(accent(x, hat))) = (accent(y, macron) - y) bold(upright(accent(x, hat))) #h(200pt) qed
  $
]

== Optimal Learning Rate 1 <task3-optimal_lr_1>
To find the optimal learning rate $lambda$, we keep ```matlab n_iters = 1000``` and perform the experiment for ```matlab lambdas = [0.00001 0.0001 0.001 0.01 0.1 1]```, picking the one with lowest corresponding `Training Error` and `Validation Error` $==>$ ```matlab lambda = 1```.
#figure(
  tablex(
    columns: (auto, 17em, auto),
    inset: 4pt,
    align: center,
    auto-lines: false,
    hlinex(),
    [$lambda$], [Training $e$], [Validation $e$],
    hlinex(stroke: 0.3pt),
    [0.00001], [0.5000], [0.4450],
    [0.0001],[0.4988],[0.4450], 
    [0.001], [0.4537], [0.4400],
    [0.01], [0.0963], [0.1050],
    [0.1], [0.0275], [0.0350],
    vlinex(start: 6, end: 7, stroke: red),
    hlinex(stroke: red),
    vlinex(start: 6, end: 7, stroke: red),
    [1], [0.0200], [0.0300],
    hlinex(stroke: red),
  ),
  kind: table,
  caption: [Training and Validation $e$ for Different Learning Rates at 1000 iterations],
)

== Optimal Learning Rate 2 <task3-optimal_lr_2>
With ```matlab n_iters = 1000``` and ```matlab lambda = 1```, the loss plateaus at the end of training, suggesting convergence of mean log-loss. 
#figure(
  image("task3-mean_loglosses.png",
  width: 60%,
  ),
  caption: [Mean Log-loss for ```matlab lambda = 1``` against number of iterations],
)

== Test Set <task3-test>
#figure(
  tablex(
    columns: 3,
    inset: 4pt,
    align: center,
    auto-lines: false,
    [], [Validation], [Test],
    hlinex(),
    [$e$], [0.0300], [0.0242],
  ),
  kind: table,
  caption: [Validation and Test $e$ with Optimal Parameters],
)

== Performance <task3-performance>
Using seeds from ```matlab rng(1)``` to ```matlab rng(20)```, we collect 20 classification error ratios for each dataset, over which we obtain their means and standard deviations.
#figure(
  tablex(
    columns: 4,
    inset: 5pt,
    align: center,
    auto-lines: false,
    hlinex(),
    [Training Samples], [Training Error], [Validation Error], [Test Error],
    hlinex(stroke: 0.3pt),
    vlinex(start: 1, end: 3, stroke: red),
    hlinex(stroke: red),
    vlinex(start: 1, end: 3, stroke: red),
    [10], [0.0000 $plus.minus$ 0.0000], [0.0500 $plus.minus$ 0.1500], [0.0600 $plus.minus$ 0.0248],
    [20], [0.0000 $plus.minus$ 0.0000], [0.0250 $plus.minus$ 0.0750], [0.0399 $plus.minus$ 0.0119],
    hlinex(stroke: red),
    [100], [0.0150 $plus.minus$ 0.0135], [0.0250 $plus.minus$ 0.0296], [0.0285 $plus.minus$ 0.0052],
    [1000], [0.0247 $plus.minus$ 0.0039], [0.0255 $plus.minus$ 0.0086], [0.0253 $plus.minus$ 0.0013],
    [10000], [0.0249 $plus.minus$ 0.0016], [0.0254 $plus.minus$ 0.0032], [0.0251 $plus.minus$ 0.0014],
    hlinex(),
  ),
  kind: table,
  caption: [Experimentation $e$ Means and Standard Deviations],
)
Over-fitting is when the error ratio for one dataset is much lower than that for the other 2. This is observed in the first 2 rows, where `Training Error` deviates greatly from both `Validation Error` and `Test Error`. In fact, the `Training Error` reaches 0, implying perfect inference by the trained model. This over-fitting may be due to the training sample size being too small (10 and 20 samples). Another observation from the table is that `Validation Error` and `Test Error` are very similar. This suggests that the model's performance on the validation samples gives us a good indication of that for the test samples. 


= Task 4: Logistic Regression with Stochastic Gradient Descent <Stochastic_Gradient_Descent_Logistic_Regression>
== Configuring Hyperparameters <task4-hyperparameters>
Using seeds from ```matlab rng(1)``` to ```matlab rng(20)```, we collect 20 classification error ratios for each dataset, over which we obtain their means and standard deviations. From there, we pick the batch size that yields the lowest corresponding `Training Error` and `Validation Error` $==>$ ```matlab batch_size = 100```
#figure(
  tablex(
    columns: (auto, 17em, auto),
    inset: 5pt,
    align: center,
    auto-lines: false,
    hlinex(),
    [Batch Size], [Training Error], [Validation Error],
    hlinex(stroke: 0.3pt),
    [1], [0.0276 $plus.minus$ 0.0058], [0.0410 $plus.minus$ 0.0041],
    [10], [0.0218 $plus.minus$ 0.0025], [0.0370 $plus.minus$ 0.0029],
    [20], [0.0210 $plus.minus$ 0.0017], [0.0355 $plus.minus$ 0.0035],
    [50], [0.0209 $plus.minus$ 0.0015], [0.0345 $plus.minus$ 0.0035],
    vlinex(start: 5, end: 6, stroke: red),
    hlinex(stroke: red),
    vlinex(start: 5, end: 6, stroke: red),
    [100], [0.0207 $plus.minus$ 0.0009], [0.0328 $plus.minus$ 0.0029], 
    hlinex(stroke: red),
  ),
  kind: table,
  caption: [Training and Validation $e$ Means and Standard Deviations],
)


== Convergence <task4-convergence>
With ```matlab n_iters = 1000```, ```matlab lambda = 1```, and ```matlab batch_size = 100```, the loss plateaus at the end of training, suggesting convergence of mean log-loss. 
#figure(
  image("task4-mean_loglosses.png",
  width: 60%,
  ),
  caption: [Mean Log-loss for ```matlab batch_size = 100``` and ```matlab lambda = 1``` against number of iterations],
)

== Comparison with Task @Gradient_Descent_Logistic_Regression <task4-comparison>
By zooming into  Figure 2, we can see that the plot is not smooth. This "noise" exists because the actual training sample size is smaller than the full training dataset, which leads to greater variance in mean log-loss when checked against the full training dataset.

== Test Set <task4-test>
The resulting classification error ratios are very similar to those in @task3-test, with test error ratio outperforming validation error ratio by a similar margin given the same order of data generation.
#figure(
  tablex(
    columns: (auto, 10em, auto),
    inset: 4pt,
    align: center,
    auto-lines: false,
    [], [Validation], [Test],
    hlinex(),
    [$e$], [0.0328 $plus.minus$ 0.0029], [0.0251 $plus.minus$ 0.0011],
  ),
  kind: table,
  caption: [Validation and Test Accuracy with Optimal Learning Rate and Batch Size],
)

== Runtime and Memory Usage Differences <task4-performance_diff>
The number of steps within each of the $n#sub[iters]$ loops directly affect model training runtime. Unlike in GD, SGD only considers a subset of the full training dataset. As such, the number of calculations in each loop is smaller, resulting in shorter runtime. However, to continually get batches of random elements within the full training dataset, the algorithm requires extra memory as large as the `batch_size`. The larger the `batch_size`, the slower the training and the larger the extra memory needed. For large databases in real-world applications, applying an SGD makes sense because it has the potential of greatly shortening model training runtime. To realise this potential, however, there must be sufficient RAM to store the batches.


= Task 5: Optimizing SVM via Linear Programming <SVM_Linear_Programming>
== Linear Programming implementation <task5-linear_prog_imp>
To be compatible with MATLAB's ```matlab linprog``` solver, $theta$ and $xi$ can be combined into $psi$ where
#align(center, text(size: 15pt)[
  $bold(psi) = mat(xi_1, xi_2, xi_3, ... , xi_(n-1), xi_n, b, w^((1)), w^((2)))^T$
])
and since $bold(upright(f)^T psi) = sum_(i=1)^n xi_i$
#align(center, text(size: 15pt)[
  $bold(upright(f)) = mat(1, 1, 1, ... , 1, 1, 0, 0, 0)^T$
])
Another necessary part is to form $bold(upright(A)), bold(beta), bold(upright(A)#sub[eq]) bold(beta#sub[eq]), bold(upright(l b)), bold(upright(u b))$ such that they map to the original conditions of the optimisation equation. These then can be used in ```matlab linprog(f, A, b, Aeq, beq, lb, ub)```. Four of these are trivial to setup due to the lack of an equality condition as well as there only being a lower bound for $bold(xi)$. Parameters $bold(upright(A))$ and $bold(beta)$ can be derived by first manipulating the original condition into 
#align(center, [
  $-（bold(upright(accent(x, hat))_i^T)theta) y_i - xi_i <= -1 $
])
#align(center, [
```matlab
function theta_opt = train_SVM_linear_progr(X, y)
    f = [ones(length(y), 1); 0; 0; 0];    % [1 1 1 ... 1 1 0 0 ]^ T with n + 3 elements
    b = -ones(length(y), 1);              % [-1 -1 -1 ... -1 -1] ^ T with n elements
    A = -eye(length(y));                  % Only ith slack param for the ith inequality
    % Based on an observation from the altered original condition
    for i = 1:length(y)
        if y(i, 1) == 1
            X(i, :) = -X(i, :);
        end
    end
    A = [A X];
    Aeq = [];                             % No equality condition
    beq = [];                             % No equality condition
    lb = [zeros(length(y), 1); -inf; -inf; -inf]; % Lower bound applies to slack params
    ub = [inf(length(lb), 1)];            % No upper bound
    theta_opt = linprog(f, A, b, Aeq, beq, lb, ub);
    theta_opt = theta_opt(end-2:end, 1);  % The last 3 elements are the optimal params
end```
])

== Performance <task5-performance>
Compared to the `Test Error` in both the GD (0.0242) and SGD (0.0251 $plus.minus$ 0.0011) cases for logistic regression, the SVM method yields very similar results.
#figure(
  tablex(
    columns: (auto, 5em, auto),
    inset: 4pt,
    align: center,
    auto-lines: false,
    [], [Train], [Test],
    hlinex(),
    [$e$], [0.0200], [0.0241],
  ),
  kind: table,
  caption: [Training and Test Accuracy with SVM Linear Programming],
)

== Decision Boundary Derivation <task5-decision_boundary>
Given the form $accent(y, macron) = b + w^((1))x^((1)) + w^((2))x^((2))$, making $x^((2))$ the subject of formula generates the following form: $x^((2)) = alpha x^((1)) + beta$ where $alpha = (-w^((1)) / w^((2)))$ and $beta = (accent(y, macron) - b) / w^((2))$.
#figure(
  tablex(
    columns: (5em, auto, 10em, auto),
    inset: 4pt,
    align: center,
    auto-lines: false,
    hlinex(),
    [], [SVM ($accent(y, macron) = 0$)], [SGD LR ($accent(y, macron) = 0.5$)], [GD LR ($accent(y, macron) = 0.5$)],
    hlinex(stroke: 0.3pt),
    [$b$], [-9.6567], [-10.9964], [-10.5063],
    [$w^((1))$], [1.9256], [1.9981], [1.9365],
    [$w^((2))$], [5.4760], [6.5872], [6.2485],
    hlinex(stroke: 0.5pt),
    [$alpha$], [-0.3516], [-0.3033], [-0.3099],
    [$beta$], [1.7635], [1.7453], [1.7614],
    hlinex(),
  ),
  kind: table,
  caption: [Optimal Parameters and Resultant $alpha$ and $beta$ for SVM, SGD and GD Logistic Regression],
)
The $alpha$ and $beta$ values are very similar in all 3 cases, and their graphs reflect the similarity too.
#figure(
  image("task5-decision_boundaries.png",
  width: 60%,
  ),
  caption: [2D Plots of Decision Boundaries for SVM, SGD, and GD Logistic Regression],
)


= Task 6: Optimizing SVM via Gradient Descent and Hinge Loss <SVM_GD_Hinge_Loss>
== Hinge Loss Gradient Computation <task6-hinge_loss_grad>
#align(center, [```matlab
for i = 1:iters_total
    grad_loss = zeros(n_features, 1);                 % Initialise Hinge Loss Gradients
    hinge_losses = hinge_loss_per_sample(X_train, y_train, theta_curr);
    for j = 1:length(hinge_losses)
        if hinge_losses(j) > 0                        % Satisfy the Gradient condition
            grad_loss = grad_loss - y_train(j) * X_train(j, :)';
        end
    end
    theta_curr = theta_curr - learning_rate / length(y_train) * grad_loss;
end
```
])

== Configuring Hyperparameters <task6-hyperparameters>
To find the optimal learning rate $lambda$, we keep ```matlab n_iters = 10000``` and perform the experiment for ```matlab lambdas = [0.00001 0.0001 0.001 0.01 0.1 1]```, picking the one with lowest corresponding `Training Error` and `Validation Error` $==>$ ```matlab lambda = 1```.
#figure(
  tablex(
    columns: (auto, 17em, auto),
    inset: 4pt,
    align: center,
    auto-lines: false,
    hlinex(),
    [$lambda$], [Training $e$], [Validation $e$],
    hlinex(stroke: 0.3pt),
    [0.00001], [0.5000], [0.4450],
    [0.0001],[0.4250],[0.4050], 
    [0.001], [0.0438], [0.0400],
    [0.01], [0.0200], [0.0350],
    [0.1], [0.0175], [0.0300],
    vlinex(start: 6, end: 7, stroke: red),
    hlinex(stroke: red),
    vlinex(start: 6, end: 7, stroke: red),
    [1], [0.0163], [0.0300],
    hlinex(stroke: red),
    [10], [0.0163], [0.0300],
    hlinex(),
  ),
  kind: table,
  caption: [Training and Validation $e$ for Different Learning Rates at 10000 iterations],
)

== Average Loss <task6-ave_loss>
The optimal combination of ```matlab n_iters = 10000``` and ```matlab lambda = 1``` generates the following average loss behaviour against number of iterations. We can see that the plot plateaus, suggesting convergence of the average loss.
#figure(
  image("task6-average_loss.png",
  width: 60%,
  ),
  caption: [Average Loss for ```matlab n_iters = 10000``` and ```matlab lambda = 1``` against number of iterations],
)

== Performance <task6-performance>
Compared with @task5-performance, the `Test Error` are very similar, implying similar performance of the learnt optimal parameters in both techniques.
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  align(horizon)[
    #figure(
      tablex(
        columns: (auto, 5em, auto, auto),
        inset: 4pt,
        align: center,
        auto-lines: false,
        [], [Train], [Validation], [Test],
        hlinex(),
        [$e$], [0.0163], [0.0300], [0.0238],
      ),
      kind: table,
      caption: [Errors with SVM Hinge Loss GD],
    )
  ],
  align(horizon)[
    #figure(
      tablex(
      columns: 3,
      inset: 4pt,
      align: center,
      auto-lines: false,
      [], [Hinge Loss GD], [Linear Programming],
      hlinex(),
      [$e$], [0.0238], [0.0241],
      ),
      kind: table,
      caption: [Errors for Different SVM methods],
    )
  ]
)

== Comparison of Optimal Parameters <task6-compare_svm_params>
When compared with the Linear Programming method, we do not get the same parameters, explaining the slight real-world performance difference of the 2 sets of optimal parameters.
#figure(
  tablex(
    columns: (auto, 8em, auto),
    inset: 4pt,
    align: center,
    auto-lines: false,
    hlinex(),
    [], [Hinge Loss GD], [Linear Programming],
    hlinex(stroke: 0.3pt),
    [$b$], [-10.0675], [-9.6567], 
    [$w^((1))$], [2.0265], [1.9256],
    [$w^((2))$], [5.7141], [5.4760],
    hlinex(),
  ),
  kind: table,
  caption: [Optimal Parameters learnt from Different methods],
)

== Decision Boundary Derivation <task6-decision_boundary>
Referring to the altered form: $x^((2)) = alpha x^((1)) + beta$ where $alpha = (-w^((1)) / w^((2)))$ and $beta = (accent(y, macron) - b) / w^((2))$, since comparison is only made between different SVM methods, $accent(y, macron) = 0$, allowing for $beta = (- b) / w^((2))$ and yielding the following $alpha$ and $beta$ values.
#figure(
  tablex(
    columns: (5em, auto, 10em),
    inset: 4pt,
    align: center,
    auto-lines: false,
    hlinex(),
    [], [Hinge Loss GD], [Linear Programming],
    hlinex(stroke: 0.3pt),
    [$alpha$], [-0.3546], [-0.3516], 
    [$beta$], [1.7619], [1.7635],
    hlinex(),
  ),
  kind: table,
  caption: [Resultant $alpha$ and $beta$ for SVM, SGD and GD Logistic Regression],
)
The $alpha$ and $beta$ values are very similar for both cases, and their graphs reflect this too.
#figure(
  image("task6-decision_boundaries.png",
  width: 60%,
  ),
  caption: [2D Plots of Decision Boundaries for Hinge Loss GD and Linear Programming]
)
