#!/usr/bin/env python3
from anthropic import Anthropic
import json
from datetime import datetime
import os
import numpy as np
from typing import Dict, Any, List
import time
from collections import deque
from datetime import datetime, timedelta

class LLMInterpreter:
    def __init__(self, api_key: str):
        """Initialize the LLM interpreter with Anthropic API key."""
        self.claude = Anthropic(api_key=api_key)
        self.context_window = []
        self.max_context_steps = 5
        self.replay_data = []
        self.interpretations = []
        self.current_episode = 0
        # Rate limiting
        self.request_timestamps = deque(maxlen=5)  # Keep track of last 5 request times
        self.requests_per_minute = 5
        
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limit by waiting if necessary."""
        now = datetime.now()
        
        # Remove timestamps older than 1 minute
        while self.request_timestamps and (now - self.request_timestamps[0]) > timedelta(minutes=1):
            self.request_timestamps.popleft()
        
        # If we've made 5 requests in the last minute, wait until we can make another
        if len(self.request_timestamps) >= self.requests_per_minute:
            # Calculate how long to wait
            oldest_request = self.request_timestamps[0]
            wait_time = (oldest_request + timedelta(minutes=1) - now).total_seconds()
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # Record this request
        self.request_timestamps.append(now)

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
        self._wait_for_rate_limit()
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

    def _format_entity_info(self, entity) -> str:
        """Format entity information into a readable string."""
        try:
            entity_type = entity.type if hasattr(entity, 'type') else 'Unknown'
            entity_id = entity.id if hasattr(entity, 'id') else 'Unknown'
            position = entity.position if hasattr(entity, 'position') else [0, 0, 0]
            health = entity.health if hasattr(entity, 'health') else 0
            
            return (f"Entity {entity_id} ({entity_type}):\n"
                   f"  Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})\n"
                   f"  Health: {health:.1f}")
        except (AttributeError, IndexError):
            return f"Entity (incomplete information)"

    def _format_state(self, env, obs: Dict[str, Any]) -> str:
        """Format the current state into a readable description."""
        state_desc = []
        
        if "entities" in obs and "entity_id_list" in obs:
            state_desc.append("=== Entities ===")
            entity_id_list = obs["entity_id_list"]
            
            # Make sure we only process valid entities
            num_entities = min(len(obs["entities"]), len(entity_id_list))
            
            for entity_idx in range(num_entities):
                try:
                    entity_id = int(entity_id_list[entity_idx])
                    if entity_id in env.entities:
                        entity = env.entities[entity_id]
                        state_desc.append(self._format_entity_info(entity))
                except (ValueError, IndexError, KeyError, AttributeError):
                    continue
        
        if not state_desc:
            state_desc = ["No entity information available"]
            
        return "\n".join(state_desc)

    def _format_attribution(self, attribution_dict, env):
        """Format attribution information into readable text."""
        if not attribution_dict or not isinstance(attribution_dict, dict):
            return "No attribution information available."

        formatted_text = []
        for action_type, attributions in attribution_dict.items():
            if not attributions or not isinstance(attributions, (list, tuple)) or len(attributions) == 0:
                continue
                
            # Get entity attributions for this action
            try:
                entity_attributions = np.array(attributions[0], dtype=np.float32)
            except (ValueError, TypeError, IndexError):
                continue

            # Sort entities by absolute attribution value
            entity_indices = np.argsort(np.abs(entity_attributions))[::-1][:3]  # Top 3 entities
            
            # Format top influential entities
            top_entities = []
            for entity_idx in entity_indices:
                attr_value = entity_attributions[entity_idx]
                if abs(attr_value) > 0.01:  # Only include significant influences
                    try:
                        entity_id = int(env.entity_id_list[entity_idx])
                        entity = env.entities[entity_id]
                        entity_type = entity.type if hasattr(entity, 'type') else 'Unknown'
                        influence = "positive" if attr_value > 0 else "negative"
                        top_entities.append(f"{entity_type} (influence: {influence}, strength: {abs(attr_value):.3f})")
                    except (IndexError, KeyError, AttributeError):
                        continue

            if top_entities:
                action_name = self._get_action_type_string(action_type)
                formatted_text.append(f"Action {action_name} was most influenced by: {', '.join(top_entities)}")

        return "\n".join(formatted_text) if formatted_text else "No significant attributions found."

    def interpret_flag_frenzy_action(self, env, action: Dict[str, Any], attribution_dict: Dict[str, List[float]], obs: Dict[str, Any]) -> str:
        """Interpret an action taken by the agent with its attribution."""
        try:
            # Format the current state and action information
            state_desc = self._format_state(env, obs)
            action_desc = self._format_action(action)
            attribution_desc = self._format_attribution(attribution_dict, env)
            
            # Create the prompt for Claude
            prompt = f"""Current State:
{state_desc}

Action Taken:
{action_desc}

Attribution Analysis:
{attribution_desc}

Based on the above information, provide a brief tactical analysis of the agent's decision. Focus on:
1. Why this action was chosen
2. Which entities influenced the decision most
3. Whether this appears to be a good tactical choice"""

            # Make the API call with required parameters
            response = self.claude.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract and return the response content
            return response.content[0].text if response and response.content else "Unable to generate interpretation."
            
        except Exception as e:
            return f"Error interpreting action: {str(e)}"

    def start_new_episode(self):
        """Start tracking a new episode by clearing previous interpretations."""
        self.interpretations = []
        self.current_episode += 1
        
    def save_episode_interpretations(self, output_dir):
        """Save the current episode's interpretations to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"episode_{self.current_episode}_interpretations.json")
        with open(output_file, "w") as f:
            json.dump({
                "episode": self.current_episode,
                "timestamp": datetime.now().isoformat(),
                "num_steps": len(self.interpretations),
                "interpretations": self.interpretations
            }, f, indent=2)
        
        return output_file

    def save_flag_frenzy_replay(self, replay_path: str):
        """Save the current context window to a replay file."""
        if not self.context_window:
            return
            
        with open(replay_path, 'w') as f:
            for step in self.context_window:
                f.write(f"Step {step['step']}\n")
                f.write(f"Action: {step['action']}\n")
                f.write(f"Interpretation: {step['interpretation']}\n\n")

    def _get_action_type_string(self, action_type: int) -> str:
        action_types = {0: "NO-OP", 1: "MOVE", 2: "RTB", 3: "ENGAGE"}
        return action_types.get(action_type, "UNKNOWN")

    def _format_action_params(self, action_type: int, params: List[float]) -> str:
        if action_type == 1:  # MOVE
            return f"({params[0]:.1f}, {params[1]:.1f})"
        elif action_type == 3:  # ENGAGE
            return f"Target ID: {int(params[0])}, Parameters: {params[1:]}"
        else:
            return ""

    def _format_action(self, action: Dict[str, Any]) -> str:
        action_type = action["action_type"]
        action_name = self._get_action_type_string(action_type)
        param_desc = self._format_action_params(action_type, action["params"])
        return f"{action_name} with parameters {param_desc}"
