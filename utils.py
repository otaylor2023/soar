from typing import Optional, Dict

def format_explanation(attribution: Optional[Dict] = None, env: Optional[Dict] = None) -> str:
        """Format attribution into human-readable explanation."""

        if attribution is None:
            return "No attribution available"

        action_name = attribution["action_type"]["name"]
        confidence = attribution["action_type"]["confidence"]
        
        explanation = [
            f"Decision: {action_name} (confidence: {confidence:.2f})\n",
            "Most influential entities:"
        ]
        
        # Add top 3 most important entities
        for entity_id, entity_attr in list(attribution["entities"].items())[:3]:
            explanation.append(f"\nEntity {entity_id}:")

            explanation.append(format_entity_info(env.entities[entity_id]))
            for feature, importance in entity_attr["top_features"].items():
                explanation.append(f"  - {feature}: {importance:.3f}")
        
        # Add target analysis for engage actions
        if "target_analysis" in attribution:
            target = attribution["target_analysis"]
            explanation.extend([
                f"\nTarget Analysis (Entity {target['target_id']}):",
                f"Engagement confidence: {target['engagement_confidence']:.2f}"
            ])
            
            for feature, importance in target["target_features"]["top_features"].items():
                explanation.append(f"  - {feature}: {importance:.3f}")
        
        return "\n".join(explanation)

def format_entity_info(entity):
    string = "" 
    # Basic entity properties
    string += f"Basic Information:\n"
    string += f"  Entity ID: {entity.EntityId}\n"
    string += f"  Name: {entity.Identifier}\n"
    string += f"  Faction: {entity.Faction}\n"
    # Format Vector3 position
    pos = entity.Pos
    string += f"  Position: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})\n"
    # Format Quat4 rotation
    rot = entity.Rot
    string += f"  Rotation: ({rot.x:.2f}, {rot.y:.2f}, {rot.z:.2f}, {rot.w:.2f})\n"
    # Format Vector3 velocity
    vel = entity.Vel
    string += f"  Velocity: ({vel.x:.1f}, {vel.y:.1f}, {vel.z:.1f})\n"
    string += f"  Domain: {entity.GetDomain()}\n"
    
    # If it's a controllable entity
    if hasattr(entity, "Controllable"):
        string += f"\nControllable Entity Properties:\n"
        string += f"  Is Controllable: {entity.Controllable}\n"
        string += f"  Current Maneuver: {entity.CurrentManouver}\n"
        string += f"  Ammo: {entity.GetAmmo()}\n"
        string += f"  Engagement Level: {entity.GetEngagementLevel()}\n"
        string += f"  Weapons Usage Mode: {entity.GetWeaponsUsageMode()}\n"
        string += f"  Has Reached Base: {entity.HasReachedBase}\n"
        string += f"  Is Alive: {entity.IsAlive()}\n"
        
        # Parent relationship
        if entity.HasParent():
            string += f"  Parent: {entity.Parent.Identifier}\n"
            
        # Target information
        target = entity.GetTargetEntity()
        if target:
            string += f"  Current Target: {target.Identifier}\n"
            
    # If it's a unit (individual entity)
    if hasattr(entity, "Health"):
        string += f"\nUnit Specific Properties:\n"
        string += f"  Health: {entity.Health}\n"
        
    # If it's a group entity (like Squadron)
    if hasattr(entity, "Entities"):
        string += f"\nGroup Entity Properties:\n"
        string += f"  Member Entities:\n"
        for member in entity.Entities:
            string += f"    - {member.Identifier}\n"
    return string