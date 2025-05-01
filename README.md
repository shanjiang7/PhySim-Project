# Physical Simulation Final Project
This is the final project for Shan Jiang and Nina De La Torre

In this project we will implement two things: First, we will extend the cloth simulation assignment to handling body collisions with a sphere object as well as handle self-collisions, so that when the cloth is twisted or pulled over itself it will not intersect with itself. Second, we are implementing is a mass-spring simulator using python on GPUs. Using the NVIDIA Warp developer framework, we will move the energy and gradient computations to the GPU with the intent of being able to run large simulations efficiently and quickly. We chose these two things to work on because Nina's beloved 2018 Macbook Pro has a particularly difficult time simulating cloth, and we wanted to take the opportunity to learn more about GPU-based programming for mesh simulations to take the burden off of her computer's hard-working CPU.  

## Cloth

## Mass-Spring
See the mass_spring.py file and run it on python to see the animation. The intent of this simulator was to use Warp to solve the energy value and gradient
functions on GPUs, and although we successfully did that, we were unable to debug completely so it does not run
how it’s supposed to. There is a bug in our step forward function in which we get stuck inside of a while loop when
trying to converge to the correct direction to move the particles. We temporarily solved this by adding a maximum
amount of iterations, but this means that we never converge to the correct direction to move our particles, so the
animation looks really funky. Nonetheless, the show must go on!
