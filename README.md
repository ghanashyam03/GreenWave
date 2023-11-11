
# Green Wave Project

## Overview

The Green Wave project aims to revolutionize urban traffic management by introducing an intelligent, eco-friendly traffic control system. This open-source initiative incorporates machine learning, V2V communication, and swarm intelligence to optimize traffic flow, reduce congestion, and minimize environmental impact.

## Features

- **TrafficControl:** Open-source traffic light control system dynamically adjusting signal timings using machine learning algorithms.
- **V2V Communication:** Enables vehicles to communicate traffic events and optimize traffic flow.
- **Swarm Intelligence:** Facilitates dynamic adjustments to vehicle speeds based on real-time data.
- **Dynamic Route Planner:** Provides vehicles with optimal routes, considering both traffic conditions and environmental impact.
- **Eco-Routing:** Promotes fuel efficiency by guiding vehicles through environmentally friendly routes.

## Getting Started

### Prerequisites

- Python (>=3.6)
- NetworkX library: Install using `pip install networkx`

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/green-wave.git
cd green-wave
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. Run the main simulation:

```bash
python main.py
```

2. Enter the number of vehicles, initial speeds, and simulation time steps when prompted.

## Project Structure

- **communication:** Contains modules for V2V communication and traffic event management.
- **intelligence:** Houses swarm intelligence and dynamic route planning modules.
- **infrastructure:** Includes the main traffic simulator and road network infrastructure.
- **routing:** Consists of eco-routing and route calculation logic.
- **vehicle.py:** Defines the Vehicle class.

## Contributing

We welcome contributions! 
