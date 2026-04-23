# EASYGNC

EasyGNC is a project dedicated to simplifying the general process of testing control models for a variety of project craft. We seek to emulate 6 Degrees of Freedom (DoF), within an active simulation environment such that we are able to change conditions within the environment easily. The end game goal is to be able to perform Monte Carlo analysis. 

## Project Components

This project is composed of various components detailed below:

- Simulation Environment (either using a web front end or Godot)
    - weather module (includes varying wind conditions and atmospheric conditions)
    - urban module (includes skyscraper adjacent models for evaluating urban safety of model)
    - Gravity (9.81 m/s)
- Mathematical Solver (for applying environmental conditions to the active model)
- control module - (adding model control into the simulation environment such that an operator can easily route the model)
    - Inputing control theory for model
    - custom model support (in .STEP)

- Interposer module (suppose we include the ability to run the project 500 times, we want to be able to visualize the results of a given control module)

### Simulation Environment

### Mathematical Solver

### control module

### Interposer module
