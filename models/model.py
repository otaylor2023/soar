# Model for FlagFrenzy Env
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
import json

torch, nn = try_import_torch()

def print_tensor_dict(d, indent=0):
    """Helper function to print a dict containing tensors."""
    for key, value in d.items():
        prefix = ' ' * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_tensor_dict(value, indent + 2)
        elif isinstance(value, torch.Tensor):
            if value.dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
                min_val = value.min().item()
                max_val = value.max().item()
                print(f"{prefix}{key}: Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, min={min_val:.4f}, max={max_val:.4f})")
            else:
                print(f"{prefix}{key}: Tensor(shape={tuple(value.shape)}, dtype={value.dtype})")
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in ['i', 'u', 'f']:
                min_val = value.min()
                max_val = value.max()
                print(f"{prefix}{key}: ndarray(shape={value.shape}, dtype={value.dtype}, min={min_val:.4f}, max={max_val:.4f})")
            else:
                print(f"{prefix}{key}: ndarray(shape={value.shape}, dtype={value.dtype})")
        else:
            print(f"{prefix}{key}: {type(value)}")

class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.last_obs = None
        self.last_attribution = None
        self.enable_attribution = model_config.get("enable_attribution", False)
        

        # Load dimensions from model_config if available
        self.max_entities = model_config.get("max_entities", 100)
        self.entity_feat_dim = model_config.get("entity_feat_dim", 26)
        self.mission_dim = model_config.get("mission_dim", 7)
        self.action_dim_param = model_config.get("action_dim_param", 10)

        # Entity encoder
        self.entity_encoder = nn.Sequential(
            nn.Flatten(),  # (max_entities * entity_feat_dim,)
            nn.Linear(self.max_entities * self.entity_feat_dim, 256),
            nn.ReLU()
        )

        # Radar visibility encoder
        self.visibility_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.max_entities * 2, 64),
            nn.ReLU()
        )

        # Mission status encoder
        self.mission_encoder = nn.Sequential(
            nn.Linear(self.mission_dim, 64),
            nn.ReLU()
        )

        # Controllable entities encoder
        self.controllable_encoder = nn.Sequential(
            nn.Linear(self.max_entities, 64),
            nn.ReLU()
        )

        # Combine and output
        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 64 + 64 + 64, 256),
            nn.ReLU()
        )

        # Discrete action type head
        self.action_type_head = nn.Linear(256, 4)

        # Continuous param vector head
        self.param_head_mu = nn.Linear(256, self.action_dim_param)
        self.param_head_log_std = nn.Linear(256, self.action_dim_param)

        self._value_branch = nn.Linear(256, 1)
        self._last_value = None

        # Feature names for attribution
        self.entity_feature_names = [
            "pos_x", "pos_y", "vel_x", "vel_y",
            "team_legacy", "team_dynasty", "entity_index", "priority",
            "domain_land", "domain_air", "domain_sea",
            "target_domain_land", "target_domain_air", "target_domain_sea",
            "ammo_remaining", "is_alive", "reached_base", "is_committing",
            "engagement_defensive", "engagement_cautious", "engagement_assertive", "engagement_offensive",
            "weapons_tight", "weapons_selective", "weapons_free",
            "target_index"
        ]

    def _compute_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        target: torch.Tensor,
        steps: int = 50,
        batch_num: int = 0
    ) -> torch.Tensor:
        """
        Compute integrated gradients for a given input and target.
        """
        # print("\nStarting integrated gradients computation:")
        # print(f"Input tensor range: min={input_tensor.min().item():.6f}, max={input_tensor.max().item():.6f}")
        
        # Make sure we're working with fresh tensors that we can modify
        input_tensor = input_tensor.detach().clone()
        baseline = baseline.detach().clone()
        
        # Generate alphas for path integral
        alphas = torch.linspace(0, 1, steps).to(input_tensor.device)
        
        # Accumulate gradients
        integrated_gradients = torch.zeros_like(input_tensor)
        
        for i, alpha in enumerate(alphas):
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass with interpolated input
            output = self.forward_for_attribution(interpolated, batch_num)
            
            if not output.requires_grad:
                output.requires_grad_(True)
            
            # Compute gradients
            if target.requires_grad:
                target = target.detach()
            
            try:
                gradients = torch.autograd.grad(
                    outputs=output[0, target.argmax()],
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(output[0, target.argmax()]),
                    create_graph=False,
                    retain_graph=False
                )[0]
                
                integrated_gradients += gradients
                
            except Exception as e:
                print(f"Error computing gradients: {str(e)}")
                import traceback
                print(traceback.format_exc())
                raise e
            
            # Clean up
            interpolated.requires_grad_(False)
            del interpolated
            del gradients
        
        # Average gradients and multiply by input difference
        attribution = ((input_tensor - baseline) * integrated_gradients) / steps
        # print(f"\nAttribution stats: mean={attribution.mean().item():.6f}, std={attribution.std().item():.6f}")
        
        return attribution.detach()

    def forward_for_attribution(self, interpolated_input: torch.Tensor, batch_num: int) -> torch.Tensor:
        """
        Forward pass specifically for attribution computation.
        Mirrors the regular forward pass but handles interpolated inputs.
        """
        batch_size = interpolated_input.shape[0]
        interpolated_input = interpolated_input.unsqueeze(0)
        
        # Run through entity encoder
        entity_encoded = self.entity_encoder(interpolated_input.unsqueeze(0))
        
        # Get the other encodings from last_obs (they stay constant during attribution)
        with torch.no_grad():
            # Get visibility encoding
            legacy = self.last_obs["visibility"]["legacy"][batch_num].float()
            dynasty = self.last_obs["visibility"]["dynasty"][batch_num].float()
            legacy = legacy.unsqueeze(0)
            dynasty = dynasty.unsqueeze(0)
            vis_concat = torch.cat([legacy, dynasty], dim=-1)
            vis_encoded = self.visibility_encoder(vis_concat)

            # Get mission encoding
            mission = self.last_obs["mission"][batch_num].unsqueeze(0)
            mission_encoded = self.mission_encoder(mission)

            # Get controllable encoding
            controllable = self.last_obs["controllable_entities"][batch_num].unsqueeze(0)
            controllable_encoded = self.controllable_encoder(controllable)

        # Combine all encodings
        x = torch.cat([controllable_encoded, entity_encoded, vis_encoded, mission_encoded], dim=-1)
        features = self.combined_fc(x)

        # Get action logits
        logits = self.action_type_head(features)
        return logits

    def _compute_batch_attribution(self, logits: torch.Tensor, mu: torch.Tensor, entities: torch.Tensor, chosen_action: int) -> List[Dict]:
        """
        Compute attributions for a batch of inputs.
        """
        # print(f"\nStarting batch attribution computation for {logits.shape[0]} samples")
        batch_attributions = []
        
        for i in range(logits.shape[0]):
            try:
                # Get the predicted action type for this sample
                sample_logits = logits[i]
                sample_mu = mu[i]
                action_type = chosen_action
                # print(f"action logits: {sample_logits}")
                # Compute attribution
                entities_tensor = entities[i]
                baseline = torch.zeros_like(entities_tensor)
                
                # Get attribution scores for entities
                entity_attribution = self._compute_integrated_gradients(
                    entities_tensor,
                    baseline,
                    sample_logits,
                    steps=20,
                    batch_num=i
                )
                # print(f"sample_logits: {sample_logits} for action type: {action_type}")
                # Create attribution dictionary
                attribution = {
                    "action_type": {
                        "name": ["NoOp", "Move", "RTB", "Engage"][action_type],
                        "confidence": F.softmax(sample_logits, dim=0)[action_type].item(),
                        "logits": sample_logits.detach().cpu().numpy()
                    },
                    "entities": {},
                }
                
                # Process each entity
                # print(f"Processing batch {i} with entity_id_list: {self.entity_id_list[i]}")
                for j in range(min(self.max_entities, len(self.entity_id_list[i]))):
                    try:
                            
                        entity_id = self.entity_id_list[i][j].item()
                        if entity_id == 0:  # Skip empty entities
                            continue
                        
                        feature_importance = entity_attribution[j].abs().tolist()
                        entity_features = {
                            name: importance
                            for name, importance in zip(self.entity_feature_names, feature_importance)
                        }
                        
                        sorted_features = sorted(
                            entity_features.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )
                        
                        attribution["entities"][int(entity_id)] = {
                            "overall_importance": sum(abs(x) for x in feature_importance),
                            "top_features": dict(sorted_features[:5])
                        }
                    except Exception as e:
                        import traceback
                        print(f"Error processing entity {j} in batch {i}: {str(e)}")
                        print(traceback.format_exc())
                        continue
                
                batch_attributions.append(attribution)
                
            except Exception as e:
                import traceback
                print(f"Error processing batch item {i}: {str(e)}")
                print(traceback.format_exc())
                batch_attributions.append(None)
                
        return batch_attributions

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        self.last_obs = obs
        self.entity_id_list = input_dict["obs"]["entity_id_list"]

        entities = obs["entities"]
        mission = obs["mission"]
        visibility = obs["visibility"]
        controllable_entities = obs.get("controllable_entities", None)
        engage_mask = obs.get("valid_engage_mask", None)

        # print(f"entities shape in forward: {entities.shape}")
        entity_encoded = self.entity_encoder(entities)
        # print(f"entity_encoded shape in forward: {entity_encoded.shape}")

        legacy = visibility["legacy"].float()   # (batch, 100)
        dynasty = visibility["dynasty"].float() # (batch, 100)
        vis_concat = torch.cat([legacy, dynasty], dim=-1)   # (batch, 200)
        vis_encoded = self.visibility_encoder(vis_concat)   # (batch, 64)

        mission_encoded = self.mission_encoder(mission)

        controllable_encoded = self.controllable_encoder(controllable_entities)  # (B, 64)

        # Combine encoders
        x = torch.cat([controllable_encoded, entity_encoded, vis_encoded, mission_encoded], dim=-1)
        features = self.combined_fc(x)  # (batch, 256)
        self._features = features

        # Compute logits + params
        logits = self.action_type_head(features)
        mu = self.param_head_mu(features)
        log_std = self.param_head_log_std(features).clamp(-5, 2)
        std = torch.exp(log_std)
        params = torch.cat([mu, std], dim=1)

        # print(f"logits before mask: {logits}")
        # Mask out "engage target" if no valid targets
        if engage_mask is not None:
            if engage_mask.dim() == 1:
                engage_mask = engage_mask.unsqueeze(0)
            # print(f"engage_mask: {engage_mask}")
            engage_mask = engage_mask.view(engage_mask.shape[0], -1)
            # print(f"engage_mask again: {engage_mask}")
            has_valid_targets = (engage_mask > 0.5).any(dim=1)
            mask_value = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            logits[:, 3] = torch.where(has_valid_targets, logits[:, 3], mask_value)
            # print(f"has_valid_targets: {has_valid_targets}")
            # print(f"logits masked to: {logits}")
        
        # print(f"logits: {logits.shape}")
        # Compute attribution for this forward pass
        # We'll do this during both training and inference
        if self.enable_attribution:
            try:
                # Temporarily enable gradients for attribution computation
                with torch.set_grad_enabled(True):
                    attribute_dict = {}
                    for i in range(logits.shape[1]):
                        attribute_dict[i] = self._compute_batch_attribution(logits, mu, entities, chosen_action=i)                
                    self.last_attribution = attribute_dict
            except Exception as e:
                print(f"Warning: Attribution computation failed in forward pass: {e}")
                self.last_attribution = None
        print(f"logits: {logits}")
        output = torch.cat([logits, params], dim=-1)
        # print(f"output: {output}")
        return output, state

    def get_last_attribution(self) -> Optional[Dict]:
        """Get the attribution from the last forward pass."""
        return self.last_attribution

    def format_explanation(self, attribution: Optional[Dict] = None) -> str:
        """Format attribution into human-readable explanation."""
        if attribution is None:
            attribution = self.last_attribution
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

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        value = self._value_branch(self._features)
        self._last_value = value.squeeze(1)
        return self._last_value

    def set_attribution_enabled(self, enabled: bool):
        """Enable or disable attribution computation."""
        self.enable_attribution = enabled
        return self
