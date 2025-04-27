#!/usr/bin/env python3
from anthropic import Anthropic
import json
from datetime import datetime
import os
import numpy as np
from typing import Dict, Any, List

class LLMInterpreter:
    def __init__(self, api_key: str):
        """Initialize the LLM interpreter with Anthropic API key."""
        self.claude = Anthropic(api_key=api_key)
        self.context_window = []
        self.max_context_steps = 5
        self.replay_data = []
        
    def _format_entity_state(self, entity_data: Dict[str, Any]) -> str:
        """Format entity state information into readable text."""
        entity_types = {0: "B1", 1: "AWACS", 2: "MADDOG"}
        formatted = []
        
        for entity_id, data in entity_data.items():
            if isinstance(data, dict):  # Skip if not a valid entity dict
                entity_type = entity_types.get(int(data.get("type", -1)), "Unknown")
                position = data.get("position", [0, 0])
                health = data.get("health", 0)
                status = "Active" if data.get("active", False) else "Inactive"
                
                formatted.append(
                    f"{entity_type} {entity_id}:\n"
                    f"  Position: ({position[0]:.1f}, {position[1]:.1f})\n"
                    f"  Health: {health:.1f}\n"
                    f"  Status: {status}"
                )
        
        return "\n".join(formatted)

    def _format_mission_status(self, mission_data: Dict[str, Any]) -> str:
        """Format mission status information."""
        if not isinstance(mission_data, dict):
            return "No mission status available"
            
        status = []
        if "objective_status" in mission_data:
            status.append(f"Objective Status: {mission_data['objective_status']}")
        if "time_remaining" in mission_data:
            status.append(f"Time Remaining: {mission_data['time_remaining']:.1f}s")
        if "score" in mission_data:
            status.append(f"Current Score: {mission_data['score']:.1f}")
            
        return "\n".join(status)

    def _create_state_description(self, obs: Dict[str, Any], info: Dict[str, Any]) -> str:
        """Convert environment state to human-readable format."""
        sections = []
        
        # Format entities
        if "entities" in obs:
            sections.append("=== Entities ===")
            sections.append(self._format_entity_state(obs["entities"]))
        
        # Format mission status
        if "mission_status" in obs:
            sections.append("\n=== Mission Status ===")
            sections.append(self._format_mission_status(obs["mission_status"]))
        
        # Format visibility information
        if "visibility" in obs:
            sections.append("\n=== Visibility Status ===")
            visible_entities = sum(1 for v in obs["visibility"].values() if v)
            sections.append(f"Visible Entities: {visible_entities}")
        
        # Add any additional info
        if info:
            sections.append("\n=== Additional Information ===")
            sections.append(json.dumps(info, indent=2))
            
        return "\n".join(sections)

    def _format_action_analysis(self, action_type: int, params: List[float]) -> str:
        """Create detailed analysis prompt based on action type."""
        action_prompts = {
            0: "Analyze why the agent chose to take no action. Consider:\n"
               "- Current tactical situation\n"
               "- Potential threats or opportunities being evaluated\n"
               "- Whether this demonstrates strategic patience",
               
            1: "Analyze the movement decision. Consider:\n"
               "- Strategic value of the chosen position\n"
               "- Risk exposure during movement\n"
               "- Coordination with other entities",
               
            2: "Analyze the Return to Base decision. Consider:\n"
               "- Current entity status and mission impact\n"
               "- Timing of the decision\n"
               "- Alternative options available",
               
            3: "Analyze the engagement decision. Consider:\n"
               "- Target selection rationale\n"
               "- Engagement parameters chosen\n"
               "- Risk/reward balance of the engagement"
        }
        
        return action_prompts.get(action_type, "Analyze the tactical and strategic implications of this action.")

    def interpret_action(self, obs: Dict[str, Any], action: Dict[str, Any], 
                        info: Dict[str, Any], reward: float) -> str:
        """Interpret an action in the context of the current state."""
        # Format current state and action
        state_desc = self._create_state_description(obs, info)
        action_type = action["action_type"]
        action_params = action["params"]
        
        # Update context window
        current_step = {
            "state": state_desc,
            "action_type": action_type,
            "action_params": action_params,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        }
        
        self.context_window.append(current_step)
        self.context_window = self.context_window[-self.max_context_steps:]
        self.replay_data.append(current_step)
        
        # Create specialized analysis prompt
        analysis_prompt = self._format_action_analysis(action_type, action_params)
        
        # Create complete prompt for Claude
        prompt = f"""As an expert military strategist, analyze this combat simulation step:

Current Battlefield State:
{state_desc}

Action Details:
Type: {['NO-OP', 'MOVE', 'RTB', 'ENGAGE'][action_type]}
Parameters: {action_params}
Immediate Reward: {reward}

{analysis_prompt}

Previous Context:
{self._format_context()}

Please provide:
1. Tactical Analysis: Immediate implications of this action
2. Strategic Assessment: Long-term mission impact
3. Decision Quality: Rate 1-10 with justification
4. Recommendations: Potential improvements or alternatives"""

        # Get Claude's interpretation
        response = self.claude.messages.create(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content

    def _format_context(self) -> str:
        """Format previous steps for context."""
        if len(self.context_window) <= 1:
            return "No previous context available"
            
        context_entries = []
        for step in self.context_window[:-1]:  # Exclude current step
            context_entries.append(
                f"Time: {step['timestamp']}\n"
                f"State:\n{step['state']}\n"
                f"Action: {['NO-OP', 'MOVE', 'RTB', 'ENGAGE'][step['action_type']]}\n"
                f"Reward: {step['reward']:.3f}\n"
            )
        
        return "\n---\n".join(context_entries)

    def save_replay(self, filepath: str) -> None:
        """Save the complete replay with interpretations to a file."""
        replay_path = os.path.join(
            os.path.dirname(filepath),
            f"llm_interpreted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(replay_path, 'w') as f:
            json.dump({
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_steps": len(self.replay_data)
                },
                "steps": self.replay_data
            }, f, indent=2)
        
        print(f"Saved interpreted replay to: {replay_path}")
