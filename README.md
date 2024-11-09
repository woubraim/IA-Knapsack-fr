# Genetic Algorithm for Knapsack Problem and Portfolio Management

This repository implements a Genetic Algorithm (GA) to solve optimization problems like the Knapsack Problem (KP) and adapts it for portfolio management. The GA simulates an evolutionary approach to generate and select optimal solutions, balancing constraints such as weight in the KP or budget and risk limits in portfolio management.

#Project Objectives

    Knapsack Problem: Develop and apply a GA to maximize item value within a weight limit.
    Portfolio Management: Adapt the GA for investment portfolios, maximizing returns while adhering to budget and risk constraints.

Problem Description

    Knapsack Problem: Given a set of items, each with a weight and a value, the goal is to select a subset of items that maximizes total value without exceeding a weight limit.
    Portfolio Management: Select financial assets to maximize expected returns, with limits on the maximum investment in any one asset to manage risk.

Repository Structure

    question2A.py: Implements GA for KP using a fitness penalty proportional to weight overage.
    question2B.py: Ensures feasibility by checking constraints at each stage.
    question3.py and question4.py: Adapt the GA for portfolio management, considering risk and budget constraints.
    instancesQ2/ and instancesQ4/: CSV files providing test cases for both KP and portfolio management problems.

Key Components and Approaches
1. Knapsack Problem (KP)

The GA in question2A.py and question2B.py demonstrates two methods to address constraint handling:

    Method 1 (Penalty): In question2A.py, the GA imposes a fitness penalty if the weight exceeds capacity, reducing the fitness score proportionally.
    Method 2 (Constraint Check): In question2B.py, the GA removes items until the total weight is under the limit, ensuring all solutions are feasible.

2. Portfolio Management

In question3.py and question4.py, the GA is adapted for financial portfolio management. It considers budget limits and a maximum allocation per asset, ensuring diversity and managing risk:

    Risk Management: Limits investment to a maximum percentage of the budget for each asset.
    Fitness Calculation: Fitness reflects expected returns while penalizing excessive allocation to any one asset.

Genetic Algorithm Workflow

    Initialization: Generates an initial population of random solutions (chromosomes).
    Fitness Calculation: Computes fitness for each individual based on value (KP) or return (portfolio management).
    Selection: Chooses top-performing individuals to act as parents.
    Crossover: Combines genes from two parents to form offspring.
    Mutation: Randomly alters genes to maintain diversity.
    Optimization: Iteratively applies selection, crossover, and mutation to evolve toward an optimal solution.

Algorithms and Techniques

    Selection: Uses a tournament selection method to retain top solutions.
    Crossover: Single and two-point crossover strategies are implemented to mix parent solutions.
    Mutation: Randomly adjusts selected bits or asset quantities to explore the solution space.

Requirements

    Python with numpy, pandas, and matplotlib for data handling and visualization.

Usage

Run Knapsack Problem:
    
    python question2A.py  # For penalty-based approach
    python question2B.py  # For feasibility-based approach

Run Portfolio Management:

    python question3.py  # Random data generation
    python question4.py  # Uses predefined instances

Example Output

    Knapsack: Displays optimal item selection, maximizing value within weight constraints.
    Portfolio: Shows the best asset allocation, with fitness representing expected returns.

Results and Observations

    KP Approaches: The constraint-checking approach generally yields better solutions by ensuring feasibility, although it may require more computational steps.
    Portfolio Optimization: The model balances return and risk constraints effectively, allowing only feasible portfolios that adhere to budget and risk limits.

References

This project follows academic guidelines for AI courses focusing on evolutionary algorithms for optimization problems.
