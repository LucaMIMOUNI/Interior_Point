# Interior Point Algorithm

## Claim
This project was developed in collaboration with Baptiste LARDINOIT, within the framework of Nils FOIX-COLONIER's Ph.D. research on UNMIXING problems, supervised by Sébastien BOURGUIGNON, Research Professor at the LS2N Laboratory of Centrale Nantes.

## Testing the algorithm
In the `code_ex_UNMIX.py`, you can specify respectively the *number of spectra* to search in the dictionnary, the *number of atomes* (end-members) and the *noise std*, for exemple `exemple1(P=90, K=3, sigma=0.0164)`; then just run the code to plot the results.

## Context
The exploration of Mars has yielded a vast dataset, particularly spectrographic data of Martian soil. Analyzing this data can help identify the mineral composition of Mars, thereby enhancing our understanding of the planet's past.

<img src="https://github.com/user-attachments/assets/23910554-6601-4588-8826-a4e8eec51f05" alt="Mars_soil" width="500"/>

*Picture of Martian soil*

To model this problem, consider a dictionary $D$ (nxp) with $p$ spectra, each consisting of $n$ wavelength data points. Let $y$ be the observation vector (n), which may be noisy and is assumed to be a linear combination of $k$ columns of $D$, where $k \ll p$. The goal of UNMIXING problems is to find the optimal vector $x$ (p) that contains the correct proportion of each mineral in the observation while satisfying physical constraints.

Various algorithms can be employed to solve Spectral UNMIXING problems. In this project, we have chosen to develop an **Interior-Point** method to minimize the Least-Squared criterion $||y - Dx||$ under the constraints $\sum x_i = 1$ and $x_i \geq 0$.

## Quadratic Program

The Least-Squared criterion can be reformulated as a quadratic program with linear constraints, making it suitable for solution using an Interior-Point algorithm. Thus, our problem can be formulated as $\min_x 0.5 x^T G x + d^T x$ subject to $A_{ineq} x \geq b_{ineq}$ and $A_{eq} x = b_{eq}$, where $G = D^T D$ and $d = -D^T y$.

## Perturbed KKT Conditions

To solve such a constrained problem, we can utilize the **perturbed KKT** conditions to reformulate the constrained problem into a system of equations:

<img src="https://github.com/user-attachments/assets/fb9e3cb6-7234-4062-94d8-194e54dbc634" alt="KKT" width="600"/>


## Results and Discussion

Consider the following example problem: *Given a hyperspectral cube of Martian soil (observation), we aim to identify $k$=3 end-members from a dictionary of $p$=90 spectra that minimize the Least-Squares criterion and determine their proportions, while adhering to the constraints of positivity and summing to 1.*

We then compare two methods: the Least-Squares solution $x_{star} = (D^T D)^{-1} D^T y$ and our *Interior-Point algorithm* for this specific problem.

<img src="https://github.com/user-attachments/assets/533a2ad1-6614-48ef-b289-d98d0f36a326" alt="LSvsIP" width="900"/>


The results are promising, as the solution obtained by the Interior-Point method yields an error of approximately $10^{-2}$ with a Signal-to-Noise Ratio (SNR) of $28$ dB, while maintaining the sum of the components equal to 1. However, further improvements are necessary, particularly in fully respecting the positivity constraint and ensuring more components are zero.

It is important to note that this algorithm is employed within the context of a *Branch & Bound algorithm*. The primary objective here is to quickly evaluate whether a specific combination of minerals could be a viable candidate for the Martian soil mineral composition. The goal was to obtain the value of the objective function within a few iterations, enabling smarter and faster exploration of the solution tree, which we successfully achieved.

<img src="https://github.com/user-attachments/assets/970d601c-da10-4d8f-aef0-c18ced9eb23f" alt="error" width="600"/>

