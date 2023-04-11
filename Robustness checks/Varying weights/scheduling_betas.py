
# Load packages

import os
import warnings
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from pulp import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")
    
rw_data = np.load('data/rw_data.npz')
due_dates = rw_data['due_dates']

#%% Load predictions

run = 1
random.seed(run)
np.random.seed(run)

for theta in [1,2,3,4]:
    nonlin = 'rf'
    error = 'gaussian'
    
    for gradient_penalty_weight in [0.8,0.9,1,1.1,1.2]:
    
        pred_path = "theta_" + str(theta) + "_run_" + str(run) + "_" + nonlin + "_" + error + "_beta_" + str(gradient_penalty_weight) + "_pred_results.npz"
        pred_res = np.load('results_pred/' + pred_path)
        y_test = pred_res['y_true']
        y_pred_wdgrl = pred_res['adversarial']
        y_pred_models = np.concatenate([y_pred_wdgrl], axis=1)

        y_pred_models[y_pred_models < 1] = 1
        y_test[y_test < 1] = 1
        predictions = ["Adversarial"]
        
        # Add due dates
        backlog = pd.DataFrame(np.zeros(shape=(y_pred_models.shape[0], 2)), columns=["due", "duration"])
        backlog["duration"] = y_test.copy()
        backlog["due"] = due_dates
        backlog = backlog.sort_values(by="due")
        
        # Take 100 earliest orders
        test_size = 100
        backlog = backlog.iloc[:test_size]
        selection = backlog.index.tolist()
        y_pred_models = y_pred_models[selection,:]
        y_test = y_test[selection,:]

        # Predefine capacity and cost ratio
        capacity = 70
        ratio = 1
        
        # Dataframe for final results
        results = pd.DataFrame(np.zeros(shape=(2,1)), 
                               columns = predictions, index = ["MAE", "Scheduling cost"])
        results.loc["MAE", "Adversarial"] = np.round(mean_absolute_error(y_test, y_pred_models[:,0:1]),1)
        
        for k in range(1):
        
            # Copy predictions of model
            y_pred = y_pred_models[:,k].copy()
        
            # Define the maximum number of available time slots in Gantt Chart
            max_slots = int(max(test_size, backlog["due"].max()))
        
            # Copy predictions to backlog and round to nearest integer
            backlog["pred"] = np.round(y_pred, 0)
            backlog["duration"] = np.round(backlog["duration"], 0)
            backlog = backlog.astype(int)
            backlog = backlog.reset_index(drop=True)
        
            tasks = list(range(test_size))
            slots = list(range(max_slots))
        
            # Setup cost matrix for every possible starting date (columns) and task (rows)
            costs = np.zeros(shape=(test_size, max_slots))
            costs[:,0] = backlog["due"] - backlog["pred"]
            for i in range(1, costs.shape[1]):
                costs[:,i] = costs[:, (i-1)] - 1
            costs = np.where(costs < 0, costs*ratio, costs)
            costs = abs(costs.astype(int))
        
            # Setup task-specific Gantt Chart for every possible starting date
            chart = np.zeros(shape=(max_slots, max_slots, test_size))
            for task in tasks:
                pred = backlog.loc[task,"pred"] # This is the predicted length of the bar in the Gantt Chart (throughput time)
                for slot in slots:
                    for i in range(pred):
                        if slot+i < max_slots:
                            chart[slot+i, slot, task] = 1
        
            # Define binary decision variables
            z = pulp.LpVariable.dicts("var", ((task, slot) for task in tasks for slot in slots), cat="Binary")
        
            # Objective function
            prob = LpProblem("The Miracle Worker", LpMinimize)
            prob += pulp.lpSum(z[task, slot]*costs[task, slot] for task in tasks for slot in slots)
        
            # Order fulfillment constraints (i.e., all tasks must be produced)
            for task in tasks:
                prob += pulp.lpSum(z[task, slot] for slot in slots) == 1
        
            # Capacity constraints
            possibilities = list(range(max_slots)) 
            # Define the possible starting dates for each task (i.e., same number as max_slots)
            mask = np.array(np.zeros(shape=(0, max_slots))) # Empty mask
            for slot in slots: 
                # This loop creates a mask that multiplies the decision variables with all possible task-specific Gantt Charts
                # The mask is reshaped to (tasks*possibilities, max_slots)
                # For each slot there are n = tasks*possibilities starting possibilities
                # For example slot = 0 generates a test_size x max_slots matrix where all tasks start at time = 0 and end at time = pred
                # For example slot = 1 generates a test_size x max_slots matrix where all tasks start at time = 1 and end at time = pred + 1
                mask = np.concatenate((mask,np.array(list(z[task, slot]*chart[possibility, slot, task] for task in tasks for possibility in possibilities)).reshape(test_size, max_slots)), axis=0)
        
            for slot in slots: 
                # This loop generates n = max_slots capacity constraints
                prob += pulp.lpSum(mask[i, slot] for i in range(mask.shape[0])) <= capacity, "Constraint_" + str(slot)
        
            # Solve optimization problem
            prob.solve(apis.GLPK_CMD(msg=0)) 
            starts = np.array(np.zeros(shape=(test_size,)))
            for v in prob.variables():
                if v.varValue == 1:
                    starts[int(''.join(c for c in v.name.split("_")[1] if c.isdigit()))] = int(''.join(c for c in v.name.split("_")[2] if c.isdigit()))

            # Create the production plan as solved by the optimization problem
            plan = pd.DataFrame(starts).reset_index()
            plan.columns = ["task", "start"]
            plan["duration"] = backlog.loc[:,"duration"]
            plan["due"] = backlog.loc[:,"due"]
            plan = plan.sort_values(by="start")
        
            # Create the real schedule with the real throughput times
            schedule = np.zeros(shape=(test_size, max_slots*5))
            for slot in slots: 
                # This loop delays production tasks if the capacity constraint is violated
                if schedule[:,slot].sum() < capacity:
                    for task in plan["task"][plan["start"]==slot]:
                        start = int(plan.loc[task, "start"])
                        duration = int(plan.loc[task, "duration"])
                        for i in range(start, start+duration):
                            schedule[task,i] = 1
                else:
                    for task in plan["task"][plan["start"]==slot]:
                        plan.loc[task,"start"] += 1
        
            # Compute the total cost of the schedule
            total_cost = 0
            for task in plan["task"]:
                c = plan.loc[task,"start"] + plan.loc[task,"duration"] - plan.loc[task,"due"]
                if c > 0:
                    c = c*ratio
                total_cost += abs(c)
            results.loc["Scheduling cost", predictions[k]] = total_cost
        
        file_name = "theta_" + str(theta) + "_run_" + str(run) + "_" + nonlin + "_" + error + "_beta_" + str(gradient_penalty_weight) + "_capacity_" + str(capacity) + "_cost-ratio_" + str(ratio) +  "_final_results.csv"  
        results.to_csv(file_name, index=True)
