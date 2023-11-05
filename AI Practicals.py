#!/usr/bin/env python
# coding: utf-8

# In[5]:


def selection_sort(arr):
    # Traverse through all array elements
    for i in range(len(arr)):
        # Find the minimum element in the unsorted part
        min_index = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j

        # Swap the found minimum element with the first unsorted element
        arr[i], arr[min_index] = arr[min_index], arr[i]

# User input for the array
array = list(map(int, input("Enter the array elements separated by spaces: ").split()))

# Display the original array
print("Original array:", array)

# Perform selection sort
selection_sort(array)

# Display the sorted array
print("Sorted array:", array)


# In[6]:


def dfs(graph, start, visited):
    if start not in visited:
        print(start)
        visited.add(start)
        for neighbor in graph[start]:
            dfs(graph, neighbor, visited)

# Initialize an empty graph
graph = {}

# User input to create the graph
num_nodes = int(input("Enter the number of nodes: "))
for _ in range(num_nodes):
    node = input("Enter node: ")
    connections = input(f"Enter connections for node {node} (separated by space): ").split()
    graph[node] = connections

# User input for the starting node
start_node = input("Enter the starting node for DFS: ")

# DFS traversal
visited_nodes = set()
dfs(graph, start_node, visited_nodes)


# In[ ]:


from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node)
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

# Input for the graph
graph = {}
nodes = int(input("Enter the number of nodes: "))

for _ in range(nodes):
    node = input("Enter node: ")
    connections = input(f"Enter connections for node {node} (separated by space): ").split()
    graph[node] = connections

# Input for the starting node
start_node = input("Enter the starting node for BFS: ")

# BFS traversal
bfs(graph, start_node)


# In[ ]:


# Branch and Bound for Graph Coloring

def is_safe(graph, colors, v, c):
    for i in range(len(graph)):
        if graph[v][i] == 1 and colors[i] == c:
            return False
    return True

def promising(graph, colors, v, m):
    for i in range(v):
        if graph[v][i] == 1 and colors[i] == 0:
            return False
    return True

def graph_coloring_bb(graph, m, colors, v):
    if v == len(graph):
        return True

    for c in range(1, m + 1):
        if is_safe(graph, colors, v, c):
            colors[v] = c
            if promising(graph, colors, v, m):
                if graph_coloring_bb(graph, m, colors, v + 1):
                    return True
            colors[v] = 0

    return False

# Example usage
graph = [
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
]

num_of_colors = 3
color_assignment = [0] * len(graph)

if graph_coloring_bb(graph, num_of_colors, color_assignment, 0):
    print("Graph can be colored with", num_of_colors, "colors:", color_assignment)
else:
    print("No possible coloring with", num_of_colors, "colors.")


# In[10]:


from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node)
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

# Input for the graph
graph = {}
nodes = int(input("Enter the number of nodes: "))

for _ in range(nodes):
    node = input("Enter node: ")
    connections = input(f"Enter connections for node {node} (separated by space): ").split()
    graph[node] = connections

# Input for the starting node
start_node = input("Enter the starting node for BFS: ")

# BFS traversal
bfs(graph, start_node)


# In[7]:


import random

responses = {
    "hello": ["Hello!", "Hi there!", "Hey!"],
    "how are you": ["I'm just a bot, but I'm doing fine. How can I help you?", "I'm here to assist you!"],
    "what's your name": ["I'm just a simple chatbot.", "I don't have a name. You can call me ChatBot."],
    "goodbye": ["Goodbye!", "See you later!"],
    "what are todays top products": ["Lenovo yoga and Asus TUF gaming are todays top products."],
    "what is the lowest pricing laptop today": ["I'ts Acer Nitro."],
    "what are its specifications": ["It comes with a 16GB RAM / 512GB SSD, intel core i5 12th generation, 4GB Graphics card RTX 3050."],
    "is there any discount on it": ["Not today but it'll have a deal discount of 29% Tommorow."],
                                       
}
                               


def get_response(user_input):
    user_input = user_input.lower()
    for key in responses:
        if key in user_input:
            return random.choice(responses[key])
    return "I don't understand that. Please ask something else."

print("ChatBot: Hi! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("ChatBot: Goodbye!")
        break
    response = get_response(user_input)
    print(response)


# In[1]:


def is_safe(graph, colors, v, c):
    for i in range(len(graph)):
        if graph[v][i] == 1 and colors[i] == c:
            return False
    return True

def graph_coloring_bt(graph, m, colors, v):
    if v == len(graph):
        return True

    for c in range(1, m + 1):
        if is_safe(graph, colors, v, c):
            colors[v] = c
            if graph_coloring_bt(graph, m, colors, v + 1):
                return True
            colors[v] = 0

    return False

def promising(graph, colors, v, m):
    for i in range(v):
        if graph[v][i] == 1 and colors[i] == 0:
            return False
    return True

def graph_coloring_bb(graph, m, colors, v):
    if v == len(graph):
        return True

    for c in range(1, m + 1):
        if is_safe(graph, colors, v, c):
            colors[v] = c
            if promising(graph, colors, v, m):
                if graph_coloring_bb(graph, m, colors, v + 1):
                    return True
            colors[v] = 0

    return False

# User input for the graph
def create_graph():
    num_nodes = int(input("Enter the number of nodes: "))
    graph = []
    for i in range(num_nodes):
        row = list(map(int, input(f"Enter connections for node {i} (0 or 1, separated by space): ").split()))
        graph.append(row)
    return graph

# User input for the number of colors
num_of_colors = int(input("Enter the number of colors: "))

# Example usage
graph = create_graph()
color_assignment = [0] * len(graph)

# Backtracking solution
if graph_coloring_bt(graph, num_of_colors, color_assignment, 0):
    print("Backtracking - Graph can be colored with", num_of_colors, "colors:", color_assignment)
else:
    print("Backtracking - No possible coloring with", num_of_colors, "colors.")

color_assignment = [0] * len(graph)

# Branch and Bound solution
if graph_coloring_bb(graph, num_of_colors, color_assignment, 0):
    print("Branch and Bound - Graph can be colored with", num_of_colors, "colors:", color_assignment)
else:
    print("Branch and Bound - No possible coloring with", num_of_colors, "colors.")


# In[2]:


def selection_sort(arr):
    for i in range(len(arr)):
        min_index = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j
        
        arr[i], arr[min_index] = arr[min_index], arr[i]
        
array = list(map(int, input("Enter array elements seperated by spaces").split()))
print(array)

selection_sort(array)

print(array)


# In[3]:


def dfs(graph, start, visited):
    if start not in visited:
        print(start)
        visited.add(start)
        for neighbour in graph[start]:
            dfs(graph, start, visited)
            
graph = {}

num_nodes = int(input("Enter the number of nodes"))
for _ in range(num_nodes):
    node = input("Enter the node")
    connections = input(f"Enter the connection for {node} seperated by space").split()
    graph[node] = connections
    
start_node = input("Enter the starting node")
visited_nodes = set()
dfs(graph, start_node, visited_nodes)


# In[6]:


import random

responses = {
    "hello": ["Hello!", "Hi there!", "Hey!"],
    "how are you": ["I'm just a bot, but I'm doing fine. How can I help you?", "I'm here to assist you!"],
    "what's your name": ["I'm just a simple chatbot.", "I don't have a name. You can call me ChatBot."],
    "goodbye": ["Goodbye!", "See you later!"],
    "what are todays top products": ["Lenovo yoga and Asus TUF gaming are todays top products."],
    "what is the lowest pricing laptop today": ["I'ts Acer Nitro."],
    "what are its specifications": ["It comes with a 16GB RAM / 512GB SSD, intel core i5 12th generation, 4GB Graphics card RTX 3050."],
    "is there any discount on it": ["Not today but it'll have a deal discount of 29% Tommorow."],
                                       
}

def get_response(user_input):
    user_input = user_input.lower()
    for key in responses:
        if key in user_input:
            return random.choice(responses[key])
        return "I dont understand that. Please ask something else"

print("Chatbot: How can I assist you")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    response = get_response(user_input)
    print(response)


# In[4]:


def is_safe(graph, colors, v, c):
    return all(graph[v][i] == 0 or colors[i] != c for i in range(len(graph)))

def graph_coloring_bt(graph, m, colors, v):
    if v == len(graph):
        return True
    for c in range(1, m+1):
        if is_safe(graph, colors, v, c):
            colors[v] = c
            if graph_coloring_bt(graph, m, colors, v+1):
                return True
            colors[v] = 0
    return False

def promising(graph, colors, v, c):
    return all(graph[v][i] == 0 or colors[i] !=c for i in range(v))

def graph_coloring_bb(graph, m, colors, v):
    if v == len(graph):
        return True
    for c in range(1, m+1):
        if is_safe(graph, colors, v, c):
            colors[v] = c
            if promising(graph, colors, v, c):
                if graph_coloring_bb(graph, m, colors, v):
                    return True
            colors[v] = 0
    return False


# In[ ]:


import heapq

class Node:
    def __init__(self, state, parent=None, action=None, g_cost=0, h_cost=0):
        self.state = state  # Current state of the node
        self.parent = parent  # Parent node
        self.action = action  # Action to reach this node
        self.g_cost = g_cost  # Cost from start node to current node
        self.h_cost = h_cost  # Heuristic cost (estimated cost from current node to goal)

    def f_cost(self):
        return self.g_cost + self.h_cost  # Total estimated cost

def astar_search(start_state, goal_state, actions, heuristic):
    start_node = Node(start_state)
    start_node.h_cost = heuristic(start_state, goal_state)  # Calculate heuristic cost for start node
    priority_queue = [(start_node.f_cost(), start_node)]

    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)

        if current_node.state == goal_state:
            path = []
            while current_node:
                path.append((current_node.state, current_node.action))
                current_node = current_node.parent
            return list(reversed(path))

        for action in actions(current_node.state):
            next_state = action  # assuming the action returns the next state
            g_cost = current_node.g_cost + 1  # In this example, each step cost is considered 1

            # Create the next node with updated costs
            next_node = Node(next_state, current_node, action, g_cost, heuristic(next_state, goal_state))

            # Check if this state is not already present with lower cost
            if not any(node[1].state == next_state and node[1].f_cost() <= next_node.f_cost() for node in priority_queue):
                heapq.heappush(priority_queue, (next_node.f_cost(), next_node))

    return None  # No path found

# Example usage:
# Define your actions function, heuristic function, start_state, and goal_state
def actions(state):
    # Define actions to generate neighboring states
    pass

def heuristic(state, goal_state):
    # Define heuristic function to estimate the cost from the current state to the goal state
    pass

start_state = ...  # Define your start state
goal_state = ...   # Define your goal state

path = astar_search(start_state, goal_state, actions, heuristic)
if path:
    print("Path found:", path)
else:
    print("No path found.")

