from __future__ import annotations
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from lead_score_flow.lead_types import CandidatePersonaDeepDive

@CrewBase
class PersonaDeepDiveCrew:
    """LLM Deep Dive Persona Evaluation Crew.

    Takes a candidate and their top 3 lightweight persona matches and performs a
    comparative analysis to select the single best persona for strategic outreach.
    """
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def deepdive_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["deepdive_analyst"],
            verbose=True,
        )

    @task
    def evaluate_personas(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_personas"],
            output_pydantic=CandidatePersonaDeepDive,
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
