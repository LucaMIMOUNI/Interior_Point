# Interior Point Algorithm

## Claim
*This project was made with Baptiste LARDINOIT, in the context of Nils FOIX-COLONIER Ph.D on UNMIXING problems supervised by Sebastien BOURGIGNON, research Professor of Centrale Nantes Laboratory LS2N.*

## Context
Space exploration of Planet Mars has led to a wide set of data, especially spectrography of the Mars soil. Analysis of these data could help us identify the Mars mineral composition and thus understand better the past of this planet.

![Mars_soil](https://github.com/user-attachments/assets/23910554-6601-4588-8826-a4e8eec51f05)

*Picture of the Mars soil*

To model this problem, lets consider a dictionnary $D$ (nxp) with p spectras, each composed by n wavelength data; then lets denote $y$ the observation (n), possibly noised and supposed to be a linear combination of $k$ columns of $D$, where $k << p$ 
