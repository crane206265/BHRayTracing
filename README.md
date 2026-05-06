# BHRayTracing
**: Black hole randering code by ray tracing**

## Basic
### Geodesic
Geodesic Equation:
$$\frac{d^2x^\mu}{d\lambda^2}+{\Gamma^{\mu}}_{\alpha\beta}\frac{dx^\alpha}{d\lambda}\frac{dx^\beta}{d\lambda}=0$$
or
$$U^\alpha\nabla_\alpha U^\mu=0$$

-----
### Schwarzchild Black Hole
Metric:
$$ds^2=-\left(1-\frac{r_s}{r}\right)dt^2+\left(1-\frac{r_s}{r}\right)^{-1}dr^2+r^2d\theta^2+r^2\sin^2\theta d\phi^2$$
where $r_s=\frac{2GM}{c^2}$ is Schwarzchild radius.

-----
--------------------
## Examples
#### [Example 1]
**Settings**
- Schwarzchild Black Hole
- Mass of BH : $M=1.00$
- inclination : $i=80\degree$
<br><br>
![BH_Example1](/saved_data/BH_img.png)