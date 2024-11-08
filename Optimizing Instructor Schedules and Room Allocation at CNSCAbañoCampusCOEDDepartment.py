import pulp
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
time_periods = 20  # Number of simulated scheduling periods
max_workload = 2   # Maximum classes per instructor

# Project-specific data: Title and context
title = "Optimizing Instructor Schedules and Room Allocation at CNSC Abaño Campus COED Department"

# Define instructors, rooms, and time slots
instructors = ["Instructor1", "Instructor2", "Instructor3"]
rooms = ["Room1", "Room2"]
time_slots = ["9-10AM", "10-11AM", "11-12PM"]

# Base room capacity and instructor availability
base_room_capacity = {"Room1": 30, "Room2": 25}
base_availability = {
    ("Instructor1", "9-10AM"): 1,
    ("Instructor1", "10-11AM"): 1,
    ("Instructor1", "11-12PM"): 0,
    ("Instructor2", "9-10AM"): 1,
    ("Instructor2", "10-11AM"): 0,
    ("Instructor2", "11-12PM"): 1,
    ("Instructor3", "9-10AM"): 1,
    ("Instructor3", "10-11AM"): 1,
    ("Instructor3", "11-12PM"): 1,
}

# Storage for results
total_scheduled_classes = []
room_utilizations = {room: [] for room in rooms}
room_allocation_data = []

# Initialize figure and axes for live plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(title)
line1, = ax1.plot([], [], 'b-', label="Total Scheduled Classes", marker="o")
ax1.set_xlim(0, time_periods)
ax1.set_ylim(0, len(instructors) * len(time_slots))
ax1.set_xlabel("Time Period")
ax1.set_ylabel("Total Scheduled Classes")
ax1.legend()

room_lines = {}
colors = ['r', 'g']
for idx, room in enumerate(rooms):
    room_lines[room], = ax2.plot([], [], color=colors[idx], label=f"{room} Utilization", marker="o")
ax2.set_xlim(0, time_periods)
ax2.set_ylim(0, len(time_slots))
ax2.set_xlabel("Time Period")
ax2.set_ylabel("Room Utilization")
ax2.legend()

# Text annotations for room allocation (updated for each frame)
annotations = []

# Function to randomly adjust availability and room capacity for each time period
def simulate_availability_and_capacity():
    availability = {}
    for instr in instructors:
        for slot in time_slots:
            availability[(instr, slot)] = base_availability[(instr, slot)] if random.random() > 0.2 else 0
    room_capacity = {room: max(20, cap + random.randint(-5, 5)) for room, cap in base_room_capacity.items()}
    return availability, room_capacity

# Function to perform the optimization for a single period
def optimize_schedule(period):
    availability, room_capacity = simulate_availability_and_capacity()
    
    # Initialize the MILP model
    model = pulp.LpProblem(f"Scheduling_Optimization_Period_{period}", pulp.LpMaximize)
    
    # Decision variables
    x = pulp.LpVariable.dicts("Assignment", 
                              ((instr, room, slot) for instr in instructors for room in rooms for slot in time_slots),
                              cat="Binary")
    
    # Objective function: maximize the total number of scheduled classes
    model += pulp.lpSum(x[(instr, room, slot)] for instr in instructors for room in rooms for slot in time_slots)
    
    # MILP Constraints
    # 1. Instructor availability constraint
    for instr in instructors:
        for slot in time_slots:
            for room in rooms:
                if availability[(instr, slot)] == 0:
                    model += x[(instr, room, slot)] == 0  # Instructor not available
    
    # 2. Room capacity constraint (one class per room per time slot)
    for room in rooms:
        for slot in time_slots:
            model += pulp.lpSum(x[(instr, room, slot)] for instr in instructors) <= 1
    
    # 3. Instructor workload constraint (limit max workload)
    for instr in instructors:
        model += pulp.lpSum(x[(instr, room, slot)] for room in rooms for slot in time_slots) <= max_workload
    
    # 4. No double-booking constraint (instructor can be in only one room per time slot)
    for instr in instructors:
        for slot in time_slots:
            model += pulp.lpSum(x[(instr, room, slot)] for room in rooms) <= 1

    # Solve the model for the current period
    model.solve()
    
    # Track the total scheduled classes and room utilization
    scheduled_classes = pulp.value(model.objective)
    total_scheduled_classes.append(scheduled_classes)
    
    # Track room utilization and room allocation details
    period_utilization = {room: 0 for room in rooms}
    period_allocation = []
    
    for instr in instructors:
        for room in rooms:
            for slot in time_slots:
                if x[(instr, room, slot)].value() == 1:
                    period_utilization[room] += 1
                    period_allocation.append(f"{instr} in {room} at {slot}")
    
    room_allocation_data.append(period_allocation)
    
    for room in rooms:
        room_utilizations[room].append(period_utilization[room])

# Update function for FuncAnimation
def update(frame):
    # Restart data for continuous simulation
    if frame == 0:
        total_scheduled_classes.clear()
        for room in rooms:
            room_utilizations[room].clear()
        room_allocation_data.clear()

    # Clear previous annotations
    global annotations
    for annotation in annotations:
        annotation.remove()
    annotations = []
    
    # Run optimization for the current frame
    optimize_schedule(frame)
    
    # Update the plot data
    line1.set_data(range(len(total_scheduled_classes)), total_scheduled_classes)
    for room in rooms:
        room_lines[room].set_data(range(len(room_utilizations[room])), room_utilizations[room])
    
    # Update room allocation table with annotations
    period_allocation = room_allocation_data[frame]
    y_position = 0.9  # Starting y-position for text annotation
    
    for allocation in period_allocation:
        annotation = ax2.annotate(allocation, (time_periods - 1, y_position), xycoords='axes fraction', 
                                  ha='left', va='top', fontsize=10, color='blue')
        annotations.append(annotation)
        y_position -= 0.05  # Shift text downward for each entry
    
    # Adjust the axes if needed
    ax1.set_xlim(0, max(1, len(total_scheduled_classes)))
    ax2.set_xlim(0, max(1, len(total_scheduled_classes)))
    return line1, *room_lines.values(), *annotations

# Run the animation with a shorter interval for smoother transition, with repeat=True for infinite loop
ani = FuncAnimation(fig, update, frames=range(time_periods), blit=True, repeat=True, interval=500)
plt.show()
