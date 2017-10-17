# Notes about project in COMPPY17
In this project we will implement a boundary integral solver for Laplace's equation on parametrised domain. Furthermore, we will implement a special quadrature to deal with nearly-singular integrals. This is a costly proceedure and we will implement it both completely in Python, as well as interfaced with C.

### Todo

- [ ] tests
- [ ] Python impl. 
- [ ] interface  
- [ ] paralellisation 

## Implementation in Python
There are several things we need to implement:

- [ ] parametrising domain
- [ ] creating GL-quadrature nodes and weights     
- [ ] reading in interpolation weights from precomp.
- [ ] computing complex density omega
- [ ] computing solution to laplaces equation with GL-quadrature
- [ ] using special quadrature    
- [ ] plot solution and errors

## TDD
We should use test driven development in this project. So, we should come up with tests to use, write the tests and *then* write the function.

### Tests to use:

- [ ] test the solution of a linear system
- [ ] sum weights of GL quadrature  
- [ ] test the computation of an integral 
- [ ] computed solution with unit density is known inside, outside and on boundary.

## Interface
We should implement an interface to C for the special quadrature, as it is the most expensive component.

## Parallelisation
Does Python have built-in parallelisation capabilities? We should check, and try them!  