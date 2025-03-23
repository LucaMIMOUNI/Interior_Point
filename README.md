# Interior Point Algorithm

## Claim
*This project was made with Baptiste LARDINOIT, in the context of Nils FOIX-COLONIER Ph.D on UNMIXING problems supervised by Sebastien BOURGIGNON, research Professor of Centrale Nantes Laboratory LS2N.*

## Context
Space exploration of Planet Mars has led to a wide set of data, especially spectrography of the Mars soil. Analysis of these data could help us identify the Mars mineral composition and thus understand better the past of this planet.

![Mars_soil](https://github.com/user-attachments/assets/23910554-6601-4588-8826-a4e8eec51f05)

*Picture of the Mars soil*

To model this problem, lets consider a dictionnary $D$ (nxp) with p spectras, each composed by n wavelength data; then lets denote $y$ the observation (n), possibly noised and supposed to be a linear combination of $k$ columns of $D$, where $k << p$. The goal of UNMIXING problems is to find the best $x$ vector (p) that contain the right proportion of each mineral in the observation while satisfying physics constraints.

Different algorithm can be used to solve Spectral UNMIXING problems. Here we choose to develop an **Interior-Point** methode to minimise the Least-Squared criteria $||y-Dx||$ under constraints $\sum x_i = 1$ and $x_i \geq 0$  

## Quadratic Program

The Least-Squared criteria can be reformulated as a quadratic program with linear constraint; which is why we can use an Interior-Point algorithm to solve this problem. Thus, our problem can be formulated as $min_x 0.5 x^T G x + d^T x$ s.c $A_{ineq} x \geq b_{ineq}$ and $A_eq x = b_eq$ with $G = D^T D$ and $d=-D^T y$.

## KKT pertubated

To solve such constrained problem, we can used the **pertubated KKT** theorem to reformulate the constrained problem to a system of equation:

![KKT](https://github.com/user-attachments/assets/fb9e3cb6-7234-4062-94d8-194e54dbc634)

## Results and discussion
