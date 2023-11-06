#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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


# In[ ]:


def is_safe(graph, colors, v, c):
    return all(graph[v][i] == 0 or colors[i] != c for i in range(len(graph)))

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
    return all(graph[v][i] == 0 or colors[i] != 0 for i in range(v))

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

def create_graph():
    num_nodes = int(input("Enter the number of nodes: "))
    graph = []
    for i in range(num_nodes):
        row = list(map(int, input(f"Enter connections for node {i} (0 or 1, separated by space): ").split()))
        graph.append(row)
    return graph

num_of_colors = int(input("Enter the number of colors: "))

graph = create_graph()
color_assignment = [0] * len(graph)

if graph_coloring_bt(graph, num_of_colors, color_assignment, 0):
    print("Backtracking - Graph can be colored with", num_of_colors, "colors:", color_assignment)
else:
    print("Backtracking - No possible coloring with", num_of_colors, "colors.")

color_assignment = [0] * len(graph)

if graph_coloring_bb(graph, num_of_colors, color_assignment, 0):
    print("Branch and Bound - Graph can be colored with", num_of_colors, "colors:", color_assignment)
else:
    print("Branch and Bound - No possible coloring with", num_of_colors, "colors.")


# In[1]:


# Prism


# In[ ]:


from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v, w):
        self.graph[u].append((v, w))
        self.graph[v].append((u, w))  # For an undirected graph

    def prim_mst(self):
        visited = {list(self.graph.keys())[0]}  # Start from the first node
        mst = []
        edges = self.graph[list(self.graph.keys())[0]]

        while len(visited) < len(self.graph):
            edges.sort(key=lambda x: x[1])  # Sort edges by weight
            for edge in edges:
                if edge[0] not in visited:
                    visited.add(edge[0])
                    mst.append((edge[0], edge[1]))
                    edges.extend(self.graph[edge[0]])
                    break

        return mst

# User input for the graph
g = Graph()
num_edges = int(input("Enter the number of edges: "))

for _ in range(num_edges):
    u, v, w = input("Enter edge and weight (format: node1 node2 weight): ").split()
    g.add_edge(u, v, int(w))

mst = g.prim_mst()
print("Minimum Spanning Tree (MST):", mst)


# In[2]:


# Kruskals


# In[ ]:


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def find_parent(self, parent, i):
        if parent[i] == i:
            return i
        return self.find_parent(parent, parent[i])

    def union(self, parent, rank, x, y):
        x_root = self.find_parent(parent, x)
        y_root = self.find_parent(parent, y)

        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
        elif rank[x_root] > rank[y_root]:
            parent[y_root] = x_root
        else:
            parent[y_root] = x_root
            rank[x_root] += 1

    def kruskal_mst(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:
            u, v, w = self.graph[i]
            i += 1
            x = self.find_parent(parent, u)
            y = self.find_parent(parent, v)

            if x != y:
                e += 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)

        return result

# User input for the graph
num_vertices = int(input("Enter the number of vertices: "))
g = Graph(num_vertices)

num_edges = int(input("Enter the number of edges: "))
for _ in range(num_edges):
    u, v, w = map(int, input("Enter edge and weight (format: node1 node2 weight): ").split())
    g.add_edge(u, v, w)

mst = g.kruskal_mst()
print("Minimum Spanning Tree (MST):", mst)


# In[3]:


# Digikstras


# In[ ]:


from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(dict)

    def add_edge(self, u, v, w):
        self.graph[u][v] = w
        self.graph[v][u] = w  # For an undirected graph

    def dijkstra(self, start):
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        visited = set()

        while len(visited) < len(self.graph):
            min_node = None
            min_distance = float('inf')

            for node in self.graph:
                if distances[node] < min_distance and node not in visited:
                    min_node = node
                    min_distance = distances[node]

            visited.add(min_node)

            for neighbor, weight in self.graph[min_node].items():
                if distances[neighbor] > distances[min_node] + weight:
                    distances[neighbor] = distances[min_node] + weight

        return distances

# Example usage:
g = Graph()
g.add_edge('A', 'B', 4)
g.add_edge('A', 'C', 2)
g.add_edge('B', 'C', 5)
g.add_edge('B', 'D', 10)
g.add_edge('C', 'D', 3)

start_node = 'A'
shortest_paths = g.dijkstra(start_node)
print("Shortest paths from node", start_node, "to all nodes:", shortest_paths)


# In[ ]:


import nltk
from nltk.chat.util import Chat, reflections

# Define patterns for the chatbot
patterns = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!']),
    (r'how are you', ['I am good, thank you. How are you?']),
    (r'what is your name', ['I am a chatbot.']),
    (r'bye|goodbye', ['Goodbye!', 'See you later.']),
    (r'name', ['Jarvis', 'Hi Im Jarvis'])
]

# Create a chatbot with defined patterns
chatbot = Chat(patterns, reflections)

# Start the chat
print("Hello! I'm a simple chatbot. You can type 'bye' to exit.")
while True:
    user_input = input("You: ")
    response = chatbot.respond(user_input)
    print("Chatbot:", response)
    if user_input.lower() == 'bye':
        break

