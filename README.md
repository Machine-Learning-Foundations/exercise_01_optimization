# Optimization Exercise
During this exercise, you will implement gradient descent to solve optimization problems.
For all activities, the task is to find the lowest point on a function surface or in 
mathematical language

$$ \min_{x}f(x) .$$

So we search for the point x on the x-axis, where the function f takes on its minimal value on the y-axis.

### ⊙  Task 1: 1d
To get started, take a look at `src/optimize_1d.py`.
Using gradient descent find the minimum of the parabola starting from point 5, or in other words

$$ \min_{x} x^2,  \text{   with   } x_0 = 5 $$

The problem is illustrated below, where the red dot indicates the position on the parabola at our starting position:
![parabola_task](./figures/parabola_task.png)

Your task is to solve this problem using gradient descent.
>> **Recap**:
Starting from position $x_0 = 5 $, we use the gradient of the parabola to find the next position on the x-axis that should be closer to the optimal position, where f takes on its minimum value. 
Specifically:
**a.** Take the last position $x_{n-1}$ in **pos_list**.
**b.** Calculate the derivative of $f$ at $x_{n-1}$.
    - This gives us the direction of the steepest ascent at the last position $x_{n-1}$.
    - Since our goal is to reach the minimum, we want to go into the opposite direction.
**c.** The step-size-parameter ($\epsilon$) tells us, how big of a step we take. This value is multiplied with the derivative calculated in **b.**
**d.** Substract the value you get in **c.** from the last position $x_{n-1}$ to get the new position $x_n$
All these steps together define one iteration of the gradient descent algorithm:
$$x_n = x_{n-1} - \epsilon\frac{df(x)}{dx}$$
This step is repeated, until a stopping criterion is met. In this task, we decide to set a fixed number of iterations we do until we stop.

1. Compute the derivative of the parabola in the method `derivative_parabola`.

Go to the `main`-function.

2. Choose suitable values for the **step_size**- and **step_total**-parameters.
    >> There is no receipt to find the best parameters! You have to try some values and see what works best for this specific task.

3. Implement the gradient-descent algorithm as described in the lecture.
    - Don't forget to append the new position calculated in each iteration to the **pos_list**.   

### ⊙  Task 2: 2d
Next we consider a paraboloid, i.e. a parobola with 2-dimensional input. That means, that instead of plugging in only one value $x$ into the function, we plug in a vector of two values: $\mathbf{x} = (x_0, x_1)$. The function value $f(x_0, x_1)$ is still a singular value like before. 
We mark vectors by writing them in bold font ($\mathbf{x}$) and $\cdot$ denotes the scalar product:

$$\min_{\mathbf{x}} \mathbf{x} \cdot \mathbf{x} = (x_0, x_1) \cdot (x_0, x_1) = x_0x_0 + x_1x_1,  \text{   with   } \mathbf{x_0} = (2.9, -2.9) .$$

The paraboloid is already implemented in `src/optimize_2d.py`. 
Once more the problem is illustrated below:
![paraboloid_task](./figures/paraboloid_task.png)
The horizontal axis represents the first entry $x_0$ of $\mathbf{x}$ and the vertical axis the second entry $x_1$. The red dot indicates the starting position. The function values are now represented as color values: The darker the color, the lower the corresponding function value. You can think of the illustration as looking at the 2-dimensional paraoboloid from above. 

Your task is to solve this problem using two-dimensional gradient descent.
You can apply the very same logic as in Task 1:
1. Compute the gradient of the paraboloid in the method `grad_paraboloid`.
   - The return value should now be an array containing two values, since we are in the 2d-case!   
  
Go to the `main`-function.

2. Choose suitable values for the **step_size**- and **step_total** parameters.

3. Implement the gradient-descent algorithm as described in the lecture.


### ⊙  Task 3: Bumpy Paraboloid
Additionally we consider a bumpy paraboloid:

$$\min_{\mathbf{x}} \mathbf{x} \cdot \mathbf{x} + \cos(2  \pi x_0) + \sin(2 \pi x_1 ), \text{   with   } \mathbf{x_0} = (2.9, -2.9) .$$

The addtional sine and cosine terms will require momentum for convergence in order to overcome the bumps and not get stuck in a sub-optimal, local minimum.
The bumpy paraboloid is already implemented in `src/optimize_2d_momentum_bumpy.py`. 
Once more the problem is illustrated below:
![bumpy_paraboloid_task](./figures/bumpy_paraboloid_task.png)

Your task is to solve this problem using two-dimensional gradient descent with momentum.
Here, you will make a slight modification to the gradient descent step you used before by adding a velocity term. Instead of just moving in the direction of the current gradient, it also takes into account the past gradients by maintaining a velocity term. This velocity is a weighted average of past gradients, giving more weight to recent ones. It contains a share of the previous velocity term and the current gradient:
$$\mathbf{v_n} = \alpha \mathbf{v_{n-1}} - \epsilon \nabla f(\mathbf{x_{n-1}})$$
**$\alpha$** is the momentum parameter and **$\epsilon$** the **step-size**-parameter from before. 
Thus, the update step is now given by:
$$\mathbf{x_n} = \mathbf{x_{n-1}} + \mathbf{v_n}$$
The workflow is the same as before:
1. Compute the gradient of the bumpy paraboloid in the method `bumpy_grad`.

Go to the `main`-function.

2. Choose suitable values for the **step_size**-, **alpha-** and **step_total**-parameters.

3. Implement the gradient-descent algorithm **with momentum** as described in the lecture.

### ⊙  Task 4: Automatic Differentiation
Finally, to explore the automatic differentiation functionality we consider the problem:

$$\min_{\mathbf{x}} \mathbf{x} \cdot \mathbf{x} + \cos(2 \pi x_0 ) + \sin(2 \pi x_1)  + 0.5 \cdot \text{relu}(x_0) + 10 \cdot \tanh( \|\mathbf{x} \| )  \text{   with   } \mathbf{x_0} = (2.9, -2.9) .$$

The function is already defined in  `src/optimize_2d_momentum_bumpy_torch.py`.
As you can probably already tell, this function looks very complicated. 
Fortunately, you don't have to compute the gradient by yourself!
Use `torch.func.grad` to compute the gradient automatically. It takes the already defined function as a parameter and outputs its derivative - which is also a method! So you can use the return-value of `torch.func.grad` exactly the same as the methods for the derivatives you implemented by yourself in the previous tasks. Use the result to find the minimum using gradient descent with momentum just like before.  
1. **Automatically** compute the gradient of the bumpy paraboloid using `torch.func.grad`.
2. Choose suitable values for the **step_size**-, **alpha-** and **step_total**-parameters.
3. Implement the gradient-descent algorithm with momentum as described in the lecture.
---
While coding use `nox -s test`, `nox -s lint`, and `nox -s typing` to check your code.
Autoformatting help is available via `nox -s format`.
Feel free to read mode about nox at https://nox.thea.codes/en/stable/ .


