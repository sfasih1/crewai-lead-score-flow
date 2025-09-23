from __future__ import annotations
from typing import List, Dict, Any
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class PersonaMatchCrew:
    """Lightweight Persona Match Crew (lexical, low-cost)."""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def persona_match_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["persona_match_agent"],
            verbose=True,
        )

    @task
    def match_personas(self) -> Task:
        return Task(
            config=self.tasks_config["match_personas"],
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
