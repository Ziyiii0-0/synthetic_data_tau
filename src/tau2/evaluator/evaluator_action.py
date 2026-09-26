import logging

from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.data_model.simulation import ActionCheck, RewardInfo
from tau2.data_model.tasks import Action, RewardType, Task
from tau2.evaluator.evaluator_base import EvaluatorBase

logger = logging.getLogger(__name__)


class ActionEvaluator(EvaluatorBase):
    """
    Evaluates whether or not the agent communicated the required information.
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> RewardInfo:
        """
        Calculate the reward based on whether the agent communicated the required information.
        """

        
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                action_checks=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )
        golden_actions = task.evaluation_criteria.actions
        if not golden_actions:
            return RewardInfo(
                reward=1.0,
                info={"note": "No actions to evaluate"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )

        action_checks = cls.evaluate_actions(full_trajectory, golden_actions)

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.action_match for result in action_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            action_checks=action_checks,
            reward_breakdown={RewardType.ACTION: reward},
        )

    @classmethod
    def evaluate_actions(
        cls,
        full_trajectory: list[Message],
        golden_actions: list[Action],
    ) -> list[ActionCheck]:
        """
        Evaluate whether the agent communicates the information correctly.
        """
        if len(golden_actions) == 0:
            return []

        predicted_tool_calls: list[ToolCall] = []
        for message in full_trajectory:
            if (
                isinstance(message, AssistantMessage)
                or isinstance(message, UserMessage)
            ) and message.is_tool_call():
                predicted_tool_calls.extend(message.tool_calls)

        # Check if all the gold actions are in the predicted actions
        action_checks = []
        for gold_action in golden_actions:
            found = False
            for pred_tool_call in predicted_tool_calls:
                if gold_action.compare_with_tool_call(pred_tool_call):
                    found = True
                    break
            if not found:
                gold_action_reward = 0.0
                gold_action_match = False
            else:
                gold_action_reward = 1.0
                gold_action_match = True
            action_checks.append(
                ActionCheck(
                    action=gold_action,
                    action_match=gold_action_match,
                    action_reward=gold_action_reward,
                )
            )
        return action_checks



class ProhibitedActionsEvaluator(ActionEvaluator):
    """
    Evaluates whether the agent executed prohibited actions.
    """
    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> RewardInfo:
        """
        Calculate the reward based on whether the agent executed prohibited actions.
        """
        # Check for prohibited actions first (for infeasible scenarios)
        # Returns (violated_actions, has_defined_actions)
        result = cls.check_prohibited_actions(task, full_trajectory)
        if result is None:
            # No prohibited actions defined
            return RewardInfo(
                reward=1.0,
                action_checks=[],
                info={"note": "No prohibited actions to evaluate"},
                reward_breakdown={RewardType.PROHIBITED_ACTION: 1.0},
            )
        
        violated_actions, _ = result
        if violated_actions:
            logger.warning(
                f"⚠️  PROHIBITED ACTION VIOLATION detected in task {task.id}: "
                f"{len(violated_actions)} prohibited action(s) were executed"
            )
            return RewardInfo(
                reward=0.0,
                action_checks=[],
                info={
                    "note": "Prohibited action executed",
                    "prohibited_violation": True,
                    "violated_actions": violated_actions
                },
                reward_breakdown={RewardType.PROHIBITED_ACTION: 0.0},
            )
        else:
            # Prohibited actions are defined but no violations found
            return RewardInfo(
                reward=1.0,
                action_checks=[],
                info={"note": "No prohibited actions were violated"},
                reward_breakdown={RewardType.PROHIBITED_ACTION: 1.0},
            )

    
    @classmethod
    def check_prohibited_actions(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> tuple[list[dict], bool] | None:
        """
        Check if any prohibited actions were executed in the trajectory.
        For infeasible scenarios, certain actions should NOT be taken.
        
        Returns:
            None if no prohibited actions are defined,
            (violated_actions, True) if prohibited actions are defined (violated_actions may be empty)
        """
        # Get prohibited actions from task.user_scenario.instructions
        instructions = task.user_scenario.instructions
        # Handle both structured and dict-based instructions
        
        prohibited_actions = instructions.actions_should_not_taken
        
        
        if not prohibited_actions:
            return None
        
        logger.info(f"Prohibited actions: {prohibited_actions}")
        # Get all tool calls from trajectory
        predicted_tool_calls: list[ToolCall] = []
        for message in full_trajectory:
            if (
                isinstance(message, AssistantMessage)
            ) and message.is_tool_call():
                predicted_tool_calls.extend(message.tool_calls)
        logger.info(f"Predicted tool calls: {predicted_tool_calls}")
        # Check if any prohibited action was executed
        violated_actions = []
        for prohibited in prohibited_actions:
            prohibited_tool = prohibited.get('tool_name', '')
            prohibited_args = prohibited.get('arguments', {})
            for pred_tool_call in predicted_tool_calls:
                # Check if tool name matches
                if pred_tool_call.name == prohibited_tool:
                    # If no specific arguments specified, any call to this tool is prohibited
                    if not prohibited_args:
                        violated_actions.append({
                            'tool_name': pred_tool_call.name,
                            'arguments': pred_tool_call.arguments,
                        })
                    else:
                        # Check if all specified prohibited arguments match
                        args_match = all(
                            pred_tool_call.arguments.get(key) == value
                            for key, value in prohibited_args.items()
                        )
                        if args_match:
                            violated_actions.append({
                                'tool_name': pred_tool_call.name,
                                'arguments': pred_tool_call.arguments,
                            })
        logger.info(f"Violated actions: {violated_actions}")
        return (violated_actions, True)




class RequiredActionsEvaluator(ActionEvaluator):
    """
    Evaluates whether the agent executed required actions.
    """
    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> RewardInfo:
        """
        Calculate the reward based on whether the agent executed required actions.
        """
        # Check for required actions
        # Returns (missing_actions, has_defined_actions)
        result = cls.check_required_actions(task, full_trajectory)
        if result is None:
            # No required actions defined
            return RewardInfo(
                reward=1.0,
                action_checks=[],
                info={"note": "No required actions to evaluate"},
                reward_breakdown={RewardType.REQUIRED_ACTION: 1.0},
            )
        
        missing_actions, _ = result
        if missing_actions:
            logger.warning(
                f"⚠️  REQUIRED ACTION VIOLATION detected in task {task.id}: "
                f"{len(missing_actions)} required action(s) were not executed"
            )
            return RewardInfo(
                reward=0.0,
                action_checks=[],
                info={
                    "note": "Required action not executed",
                    "required_violation": True,
                    "not_executed_actions": missing_actions
                },
                reward_breakdown={RewardType.REQUIRED_ACTION: 0.0},
            )
        else:
            # Required actions are defined and all were executed
            return RewardInfo(
                reward=1.0,
                action_checks=[],
                info={"note": "All required actions were executed"},
                reward_breakdown={RewardType.REQUIRED_ACTION: 1.0},
            )

    
    @classmethod
    def check_required_actions(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> tuple[list[dict], bool] | None:
        """
        Check if any required actions were not executed in the trajectory.
        For infeasible scenarios, certain actions should be taken.
        
        Returns:
            None if no required actions are defined,
            (missing_actions, True) if required actions are defined (missing_actions may be empty)
        """
        # Get required actions from task.user_scenario.instructions
        instructions = task.user_scenario.instructions
        
        # Handle both structured and dict-based instructions
        required_actions = instructions.actions_should_be_taken
        
        if not required_actions:
            return None
        
        logger.info(f"Required actions: {required_actions}")
        # Get all tool calls from trajectory
        predicted_tool_calls: list[ToolCall] = []
        for message in full_trajectory:
            if (
                isinstance(message, AssistantMessage)
            ) and message.is_tool_call():
                predicted_tool_calls.extend(message.tool_calls)
        logger.info(f"Predicted tool calls: {predicted_tool_calls}")
        # Check if any required action was not executed in the trajectory
        missing_actions = []
        for required in required_actions:
            required_tool = required.get('tool_name', '')
            required_args = required.get('arguments', {})
            
            # Check if this required action was executed in the trajectory
            found_match = False
            for pred_tool_call in predicted_tool_calls:
                # Check if tool name matches
                if pred_tool_call.name == required_tool:
                    # If no specific arguments specified, any call to this tool satisfies the requirement
                    if not required_args:
                        found_match = True
                        break
                    else:
                        # Check if all specified required arguments match
                        args_match = all(
                            pred_tool_call.arguments.get(key) == value
                            for key, value in required_args.items()
                        )
                        if args_match:
                            found_match = True
                            break
            
            # If no match was found, this required action was not executed
            if not found_match:
                missing_actions.append({
                    'tool_name': required_tool,
                    'arguments': required_args,
                })
        
        logger.info(f"Missing required actions: {missing_actions}")
        return (missing_actions, True)