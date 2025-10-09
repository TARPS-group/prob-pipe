# Probabilistic Pipeline Dev


The core files of this project live under probpipe/core directory. 

## distributions.py
consist of some distribution classes. The Distribution class is the abstract base class. You can see other distribution classes inside this file like EmpiricalDistribution and BootstrapDistribution. 

## multivariate.py
consist of other abstract class called Multivariate. Note that Multivariate doesn't imply the high-dimensionality. Instead it is more like a type. You can see the distributions with density inside this file. 
Please take a look at docs/notebooks/Distributions.ipynb to understand how the distribution classes and their methods work. 

## module.py
is an example of a basic standard module class. The core of this file is the run_func function: You can see what our run decorator deals with. In a high-level overview, it does the following:
1) Ensures that required dependencies are registered.
2) Ensures that required inputs are satisfied.
3) Checks the if run parameters have the correct type. If not, it handles the automatic type conversion.

Important things to know: 
1) The required inputs are inferred from the signature of the run function. In the background each input is stored as an instance of InputSpec class which has three parameters: type, required, and default. If the user specifies the type of the parameter then the type of InputSpec is set accordingly. If the user sets a default value, then we treat that input as not required (required= False). But, if user doesn't set a default value for that parameter we treat it as a required input.

2) Almost every distribution class has the from_distribution method. We are using this method to deal with the distribution conversion. Briefly, from_distribution samples from "convert_from" distribution and calculates an estimate of parameters from the sample. At the end it returns the distribution class that the from_distribution part of. There is also another way of doing the type conversion: GaussianKDE deduced from "convert_from" distribution (this not implemented yet). 

We have created two examples inside probpipe directory to show how one can create basic modules: example_mcmc.py (core/mcmc.py) and example_module.py

Let's take a look at one example:

## example_module.py
In this example, we implemented a single Gaussian Posterior update block. 
Let's look at the constructor:

The user doesn't have to set inputs manually. But, if they still want to, they can do it by calling the set_inputs method.

self._conv_num_samples is the attribute to set the number of samples you want to take from "convert_from" distributio in from_distribution method to do the type conversion. 

self._conv_by_kde is for selecting the method the user wants to choose to do the type conversion. If self._conv_by_kde=True, then type conversion is made via GaussianKDE.

self.run_func() is for registering the run function. It optionally takes the as_task parameter which users can use to decide if they want to treat the run function as a task or flow (to understand the difference please look at Prefect task and flow definitions). 

## Notes for Prefect 
To start the prefect engine, open up a fresh terminal and type:
prefect server start

Then, go back to the terminal where you run the code, and type:
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api 

You can paste "http://127.0.0.1:4200" in your browser to see the Prefect interface in the server side. 

Now, when you run a task or flow you can see the details in the Prefect interface. 

But, you may run into problems if you set as_task as True, meaning you are treating your run function as flow. You may run into a problem if type conversion will be happening. Prefect flows automatically validate their inputs before it runs the function. So, you would get an error before our code even attempts for conversion. So, if that's the problem you are facing just set validate_parameters as False. validate_parameters is the parameter of the flow function (see flow(validate_parameters=False)(wrapper) in module.py)

## How to set the environment to run the code. 
You can either use a virtual or conda environment. We have imported many libraries for this code. So, you will also need to satisfy the proper environment to be able to run the code. So, I am sharing my conda environment details in environment.yml (you can use this file to create a copy of my environment). 

Store the environment.yml in your local computer and then type the following in your terminal:
conda env create -f environment.yml 













