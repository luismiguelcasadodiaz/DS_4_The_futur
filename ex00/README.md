#F1score 

F1score is hte harmonic mean of re Precision and the reacl.


The harmonic mean is a type of average. You take the reciprocals (1/x) of the values, average those, and then take the reciprocal of the result.

```math
Harmonic_mean = \frac{1}{\frac{\frac{1}{p} + \frac{1}{r}}{2}} = \frac{2}{\frac{1}{p} + \frac{1}{r}} =\frac{2}{\frac{r +p}{p * r} } = \frac{2 * p * r}{r +p}
```