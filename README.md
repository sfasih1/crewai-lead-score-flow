# Lead Score Flow

Welcome to the Lead Score Flow project, powered by [crewAI](https://crewai.com). This example demonstrates how you can leverage Flows from crewAI to automate the process of scoring leads, including data collection, analysis, and scoring. By utilizing Flows, the process becomes much simpler and more efficient.

## Overview

This flow will guide you through the process of setting up an automated lead scoring system. Here's a brief overview of what will happen in this flow:

1. **Load Leads**: The flow starts by loading lead data from a CSV file named `leads.csv`.

2. **Score Leads**: The `LeadScoreCrew` is kicked off to score the loaded leads based on predefined criteria.

3. **Human in the Loop**: The top 3 candidates are presented for human review, allowing for additional feedback or proceeding with writing emails.

4. **Write and Save Emails**: Emails are generated and saved for all leads.

By following this flow, you can efficiently automate the process of scoring leads, leveraging the power of multiple AI agents to handle different aspects of the lead scoring workflow.

## Installation

Ensure you have Python >=3.10 <=3.13 installed on your system. First, if you haven't already, install CrewAI:

```bash
pip install crewai
```

Next, navigate to your project directory and install the dependencies:

1. First lock the dependencies and then install them:

```bash
crewai install
```

### Customizing & Dependencies

**Add your `OPENAI_API_KEY` into the `.env` file**  
**Add your `SERPER_API_KEY` into the `.env` file**


## Running the Project

### Run the Flow

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
crewai run
```
```bash
uv run kickoff
```
### Plot the Flow

```bash
uv run plot
```

This command initializes the lead_score_flow, assembling the agents and assigning them tasks as defined in your configuration.

When you kickstart the flow, it will orchestrate multiple crews to perform the tasks. The flow will first collect lead data, then analyze the data, score the leads and generate email drafts.


