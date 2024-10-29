"""
This code contains the ioh problem
"""

import numpy as np
import metaheuristic as mh
import benchmark_func as bf
import ioh
import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

class P1(bf.BasicProblem):  # not needed
    def __init__(self, variable_num, problem):
        super().__init__(variable_num)
        self.max_search_range = problem.bounds.ub
        self.min_search_range = problem.bounds.lb
        self.func_name = 'P1'
        self.problem = problem
    
    def create_ioh_problem(problem_id, instance, dimension):
        problem = ioh.get_problem(problem_id, instance=instance, dimension=dimension)
        return problem

    def get_func_val(self, variables, *args):
        fcost = self.problem(variables)
        return fcost