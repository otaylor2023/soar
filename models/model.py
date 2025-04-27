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


torch, nn = try_import_torch()

class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.last_obs = None
        self.last_attribution = None

        # TODO: Don't hardcode!
        self.max_entities = 100
        self.entity_feat_dim = 26
        self.mission_dim = 7

        self.action_dim_param = 10

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
        print("\nStarting integrated gradients computation:")
        print(f"Input tensor range: min={input_tensor.min().item():.6f}, max={input_tensor.max().item():.6f}")
        print(f"Baseline tensor range: min={baseline.min().item():.6f}, max={baseline.max().item():.6f}")
        
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
                print(f"\nStep {i} gradients:")
                print(f"Target logits: {target}")
                print(f"Output logits: {output}")
                print(f"Selected action: {target.argmax().item()}")
                print(f"Action logit value: {output[0, target.argmax().item()].item():.6f}")
                
                gradients = torch.autograd.grad(
                    outputs=output[0, target.argmax()],
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(output[0, target.argmax()]),
                    create_graph=False,
                    retain_graph=False
                )[0]
                
                print(f"Gradient range: min={gradients.min().item():.6f}, max={gradients.max().item():.6f}")
                integrated_gradients += gradients
                print(f"Integrated gradients sum: {integrated_gradients.abs().sum().item():.6f}")
                
            except Exception as e:
                print(f"Error computing gradients: {str(e)}")
                raise e
            
            # Clean up
            interpolated.requires_grad_(False)
            del interpolated
            del gradients
        
        # Average gradients and multiply by input difference
        input_diff = (input_tensor - baseline)
        print(f"\nInput difference range: min={input_diff.min().item():.6f}, max={input_diff.max().item():.6f}")
        
        attribution = ((input_tensor - baseline) * integrated_gradients) / steps
        print(f"\nAttribution components:")
        print(f"- Input difference stats: mean={input_diff.mean().item():.6f}, std={input_diff.std().item():.6f}")
        print(f"- Integrated gradients stats: mean={integrated_gradients.mean().item():.6f}, std={integrated_gradients.std().item():.6f}")
        print(f"- Final attribution stats: mean={attribution.mean().item():.6f}, std={attribution.std().item():.6f}")
        
        return attribution.detach()

    def forward_for_attribution(self, interpolated_input: torch.Tensor, batch_num: int) -> torch.Tensor:
        """
        Forward pass specifically for attribution computation.
        Mirrors the regular forward pass but handles interpolated inputs.
        """
        batch_size = interpolated_input.shape[0]
        print(f"\nForward for attribution:")
        print(f"Input shape: {interpolated_input.shape}, batch_size: {batch_size}")
        interpolated_input = interpolated_input.unsqueeze(0)
        # Run through entity encoder
        entity_encoded = self.entity_encoder(interpolated_input.unsqueeze(0))
        print(f"Entity encoded shape: {entity_encoded.shape}")
        
        # Get the other encodings from last_obs (they stay constant during attribution)
        with torch.no_grad():
            # Get visibility encoding
            legacy = self.last_obs["visibility"]["legacy"][batch_num].float()
            dynasty = self.last_obs["visibility"]["dynasty"][batch_num].float()
            print(f"Initial visibility shapes - legacy: {legacy.shape}, dynasty: {dynasty.shape}")
            
            # Match batch size while preserving the original dimensions
            legacy = legacy.unsqueeze(0)
            dynasty = dynasty.unsqueeze(0)
            # if legacy.shape[0] != batch_size:
            #     if legacy.dim() == 1:
            #         legacy = legacy.unsqueeze(0)
            #         dynasty = dynasty.unsqueeze(0)
            #         print("Unsqueezed visibility tensors")
            #     # Repeat to match batch size
            #     legacy = legacy.repeat(batch_size, 1)
            #     dynasty = dynasty.repeat(batch_size, 1)
            #     print(f"Repeated visibility to match batch - new shapes - legacy: {legacy.shape}, dynasty: {dynasty.shape}")
            
            vis_concat = torch.cat([legacy, dynasty], dim=-1)
            vis_encoded = self.visibility_encoder(vis_concat)
            print(f"Visibility encoded shape: {vis_encoded.shape}")

            # Get mission encoding
            mission = self.last_obs["mission"][batch_num].unsqueeze(0)
            print(f"Initial mission shape: {mission.shape}")
            # if mission.shape[0] != batch_size:
            #     if mission.dim() == 1:
            #     mission = mission.unsqueeze(0)
            #     print("Unsqueezed mission tensor")
            #     mission = mission.repeat(batch_size, 1)
            #     print(f"Repeated mission to match batch - new shape: {mission.shape}")
            mission_encoded = self.mission_encoder(mission)
            print(f"Mission encoded shape: {mission_encoded.shape}")

            # Get controllable encoding
            controllable = self.last_obs["controllable_entities"][batch_num].unsqueeze(0)
            print(f"Initial controllable shape: {controllable.shape}")
            # if controllable.shape[0] != batch_size:
            #     if controllable.dim() == 1:
            #     controllable = controllable.unsqueeze(0)
            #     print("Unsqueezed controllable tensor")
            #     controllable = controllable.repeat(batch_size, 1)
            #     print(f"Repeated controllable to match batch - new shape: {controllable.shape}")
            controllable_encoded = self.controllable_encoder(controllable)
            print(f"Controllable encoded shape: {controllable_encoded.shape}")

        # Combine all encodings
        x = torch.cat([controllable_encoded, entity_encoded, vis_encoded, mission_encoded], dim=-1)
        print(f"Combined encodings shape: {x.shape}")
        features = self.combined_fc(x)
        print(f"Features shape after FC: {features.shape}")

        # Get action logits
        logits = self.action_type_head(features)
        print(f"Final logits shape: {logits.shape}")
        print(f"Logits values: {logits}")
        
        return logits

    def compute_attribution(self, input_dict: Dict, logits: torch.Tensor, params: torch.Tensor) -> Dict:
        """
        Compute attribution for the current forward pass.
        """
        # Get the predicted action type
        action_type = torch.argmax(logits).item()
        
        # Initialize attribution dictionary
        attribution = {
            "action_type": {
                "name": ["NoOp", "Move", "RTB", "Engage"][action_type],
                "confidence": F.softmax(logits, dim=0)[action_type].item(),
                "logits": logits.detach().cpu().numpy()
            },
            "entities": {},
        }
        
        # Compute attribution for chosen action type
        entities_tensor = input_dict["obs"]["entities"]
        batch_size = entities_tensor.shape[0]
        baseline = torch.zeros_like(entities_tensor)
        
        # Get attribution scores for entities
        with torch.set_grad_enabled(True):
            entity_attribution = self._compute_integrated_gradients(
                entities_tensor,
                baseline,
                logits[action_type],
                steps=20  # Reduced steps for efficiency during training
            )
        
        # For each entity, compute its importance
        for i in range(self.max_entities):
            entity_id = input_dict["obs"]["entity_id_list"][0][i].item()  # Use first batch item for IDs
            if entity_id == 0:  # Skip empty entities
                continue
                
            # Get feature importance for this entity
            # Average across batch dimension if needed
            if entity_attribution.dim() == 3:  # If we have batch dimension
                feature_importance = entity_attribution[:, i].mean(0).abs().tolist()
            else:
                feature_importance = entity_attribution[i].abs().tolist()
            
            # Create feature importance dictionary
            entity_features = {
                name: importance
                for name, importance in zip(self.entity_feature_names, feature_importance)
            }
            
            # Sort features by importance
            sorted_features = sorted(
                entity_features.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            attribution["entities"][int(entity_id)] = {
                "overall_importance": sum(abs(x) for x in feature_importance),
                "top_features": dict(sorted_features[:5])  # Top 5 most important features
            }
        
        # For engage actions, try to identify target
        if action_type == 3:
            # Assuming param[1] contains target information
            target_idx = int(params[1].item() * self.max_entities)  # Denormalize
            if 0 <= target_idx < self.max_entities:
                target_id = input_dict["obs"]["entity_id_list"][0][target_idx].item()  # Use first batch item
                if target_id != 0:
                    attribution["target_analysis"] = {
                        "target_id": int(target_id),
                        "target_features": attribution["entities"].get(int(target_id), {}),
                        "engagement_confidence": F.sigmoid(params[0]).item()  # Using first param as confidence
                    }
        
        # Sort entities by overall importance
        attribution["entities"] = dict(
            sorted(
                attribution["entities"].items(),
                key=lambda x: x[1]["overall_importance"],
                reverse=True
            )
        )
        
        return attribution

    def _compute_batch_attribution(self, logits: torch.Tensor, mu: torch.Tensor, entities: torch.Tensor) -> List[Dict]:
        """
        Compute attributions for a batch of inputs.
        """
        print("\nStarting batch attribution computation:")
        print(f"Batch shapes - logits: {logits.shape}, mu: {mu.shape}, entities: {entities.shape}")
        print(f"Last obs shapes:")
        print(f"- visibility legacy: {self.last_obs['visibility']['legacy'].shape}")
        print(f"- visibility dynasty: {self.last_obs['visibility']['dynasty'].shape}")
        print(f"- mission: {self.last_obs['mission'].shape}")
        print(f"- controllable: {self.last_obs['controllable_entities'].shape}")
        print(f"- entity_id_list: {len(self.entity_id_list)}")
        
        batch_attributions = []
        for i in range(logits.shape[0]):
            try:
                print(f"\nProcessing batch item {i}:")
                print(f"Accessing tensors at index {i}:")
                print(f"- logits[{i}].shape: {logits[i].shape}")
                print(f"- mu[{i}].shape: {mu[i].shape}")
                print(f"- entities[{i}].shape: {entities[i].shape}")
                print(f"- last_obs visibility legacy size: {self.last_obs['visibility']['legacy'].size()}")
                print(f"- last_obs visibility dynasty size: {self.last_obs['visibility']['dynasty'].size()}")
                print(f"- last_obs mission size: {self.last_obs['mission'].size()}")
                print(f"- last_obs controllable size: {self.last_obs['controllable_entities'].size()}")
                
                # Get the predicted action type for this sample
                sample_logits = logits[i]
                sample_mu = mu[i]
                action_type = torch.argmax(sample_logits).item()
                
                print(f"Predicted action type: {action_type}")
                print(f"entities shape: {entities.shape}")
                
                # Compute attribution
                entities_tensor = entities[i]
                print(f"entities tensor shape: {entities_tensor.shape}")
                baseline = torch.zeros_like(entities_tensor)
                
                print(f"About to compute integrated gradients for batch {i}")
                print(f"- entity_id_list length: {len(self.entity_id_list)}")
                print(f"- entity_id_list[{i}] exists: {i < len(self.entity_id_list)}")
                
                # Get attribution scores for entities
                entity_attribution = self._compute_integrated_gradients(
                    entities_tensor,
                    baseline,
                    sample_logits,
                    steps=20,
                    batch_num=i
                )
                
                print(f"Attribution shape: {entity_attribution.shape}")
                
                # Create attribution dictionary
                attribution = {
                    "action_type": {
                        "name": ["NoOp", "Move", "RTB", "Engage"][action_type],
                        "confidence": F.softmax(sample_logits, dim=0)[action_type].item(),
                        "logits": sample_logits.detach().cpu().numpy()
                    },
                    "entities": {},
                }
                
                # For each entity, compute its importance
                for j in range(min(self.max_entities, len(self.entity_id_list[i]))):
                    try:
                        print(f"Processing entity {j} for batch {i}")
                        print(f"entity_id_list shape: {len(self.entity_id_list)}")
                        print(f"entity_id_list[{i}] length: {len(self.entity_id_list[i]) if i < len(self.entity_id_list) else 'index out of range'}")
                        print(f"max_entities: {self.max_entities}")
                        print(f"Current j: {j}")
                        
                        if j >= len(self.entity_id_list[i]):
                            print(f"Skipping j={j} as it's beyond entity list length {len(self.entity_id_list[i])}")
                            continue
                            
                        entity_id = self.entity_id_list[i][j].item()
                        if entity_id == 0:  # Skip empty entities
                            continue
                        
                        # Get feature importance for this entity
                        feature_importance = entity_attribution[j].abs().tolist()
                        
                        # Create feature importance dictionary
                        entity_features = {
                            name: importance
                            for name, importance in zip(self.entity_feature_names, feature_importance)
                        }
                        
                        # Sort features by importance
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
                        print("Stack trace:")
                        print(traceback.format_exc())
                        raise e
                
                batch_attributions.append(attribution)
                
            except Exception as e:
                import traceback
                print(f"Error processing batch item {i}: {str(e)}")
                print("Stack trace:")
                print(traceback.format_exc())
                print(f"Tensor states - entities: {entities_tensor.requires_grad}, logits: {sample_logits.requires_grad}")
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

        print(f"entities shape in forward: {entities.shape}")
        entity_encoded = self.entity_encoder(entities)
        print(f"entity_encoded shape in forward: {entity_encoded.shape}")

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

        # Mask out "engage target" if no valid targets
        if engage_mask is not None:
            if engage_mask.dim() == 1:
                engage_mask = engage_mask.unsqueeze(0)
            engage_mask = engage_mask.view(engage_mask.shape[0], -1)
            has_valid_targets = (engage_mask > 0.5).any(dim=1)
            mask_value = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            logits[:, 3] = torch.where(has_valid_targets, logits[:, 3], mask_value)

        # Compute attribution for this forward pass
        # We'll do this during both training and inference, but only if we have gradients enabled
        if torch.is_grad_enabled():
            try:
                with torch.set_grad_enabled(True):
                    self.last_attribution = self._compute_batch_attribution(logits, mu, entities)
            except Exception as e:
                print(f"Warning: Attribution computation failed in forward pass: {e}")
                self.last_attribution = None

        output = torch.cat([logits, params], dim=-1)
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
