import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from lead_score_flow.lead_types import CandidateScore


@CrewBase
class LeadScoreCrew:
    """Lead Score Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def hr_evaluation_agent(self) -> Agent:
        llm_model = os.getenv("LLM_MODEL")
        return Agent(
            config=self.agents_config["hr_evaluation_agent"],
            verbose=True,
            llm=llm_model if llm_model else None,
        )

    @task
    def evaluate_candidate_task(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_candidate"],
            output_pydantic=CandidateScore,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Lead Score Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
