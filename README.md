# ESGDT-server
A middleman server for the ESGDT Project that helps with Input and Video Feed Transmission between Simulation and Control software.

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone the Project Repository](#step-1-clone-the-project-repository)
  - [Step 2: Install Dependencies](#step-2-install-dependencies)
  - [Step 3: Run the Project](#step-4-run-the-project)
- [Usage](#usage)

## Installation
Follow the steps below to set up and run this project.

### Prerequisites

Before you begin, ensure you have the following installed on your machine:

- **Python**: The programming language used for this project.

### Step 1: Clone the Project Repository

Clone the repository containing the project:

```bash
git clone https://github.com/sahilg2000/ESGDT-server.git
```

### Step 2: Install Dependencies

Install the required dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Project

1. **Start the Flask App**:

    Run the Flask application using:

    ```bash
    python stream_server.py
    ```

2. **Open the Application in a Browser**:

    Navigate to `http://localhost:8080` in your browser to view the video stream.

## Usage

To use this application with the ESGDT Car simulation, please navigate to:
[ESGDT](https://github.com/sahilg2000/ESGDT/)

Follow the installation steps on that application and run an OBS Virtual Camera on the live simulation.

