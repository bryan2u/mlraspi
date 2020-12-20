# Import required libraries
import numpy as np
import pylab as plt
import networkx as nx
# Map cell to cell, add circular cell to goal point
points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]
# Set target node
goal = 7
# Create and display graph
G=nx.Graph()
G.add_edges_from(points_list)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.show()
# Define points in graph
MATRIX_SIZE = 8
# Create matrix (MATRIX_SIZE * MATRIX_SIZE)
R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
R *= -1
# Assign zeros to paths and 100 to goal-reaching point
for point in points_list:
    print(point)
    if point[1] == goal:
        R[point] = 100
    else:
        R[point] = 0
    if point[0] == goal:
        R[point[::-1]] = 100
    else:
        # Reverse of point
        R[point[::-1]]= 0
# Add goal point round trip
R[goal,goal]= 100
# Create Q matrix
Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))
# Set learning parameter gamma
gamma = 0.8
# Set initial start point
initial_state = 1
# Define available_actions method
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act
# Create variable to hold possible actions
available_act = available_actions(initial_state)
# Define method to randomly select next action
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action
# Create variable to hold the randomly selected action
action = sample_next_action(available_act)
# Define method to update state if needed
def update(current_state, action, gamma):
  max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
  if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
  else:
      max_index = int(max_index)
  max_value = Q[action, max_index]
  # Bellman's equation
  Q[current_state, action] = R[current_state, action] + gamma * max_value
  print('max_value', R[current_state, action] + gamma * max_value)
  if (np.max(Q) > 0):
    return(np.sum(Q/np.max(Q)*100))
  else:
    return (0)
# Update the state based on selected action
update(initial_state, action, gamma)
# Training starts now
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    score = update(current_state,action,gamma)
    scores.append(score)
    print ('Score:', str(score))
# Display the rewards matrix
print('\nRewards matrix R\n')
print(R)
# Display the enhanced Q matrix
print('\nEnhanced Q matrix\n')
print(Q/np.max(Q)*100)
# Testing starts now
current_state = 0
steps = [current_state]
# Loop to determine optimal path
while current_state != 7:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index
print("Most efficient path:")
print(steps)
plt.plot(scores)
plt.show()
