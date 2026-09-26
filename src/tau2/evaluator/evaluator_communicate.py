from tau2.data_model.message import AssistantMessage, Message
from tau2.data_model.simulation import CommunicateCheck, RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator_base import EvaluatorBase


class CommunicateEvaluator(EvaluatorBase):
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
                info={"notes": "No evaluation criteria"},
                reward_breakdown={RewardType.COMMUNICATE: 1.0},
            )
        communicate_info = task.evaluation_criteria.communicate_info
        if not communicate_info:
            return RewardInfo(
                reward=1.0,
                info={"note": "No communicate_info to evaluate"},
                reward_breakdown={RewardType.COMMUNICATE: 1.0},
            )

        communicate_info_checks = cls.evaluate_communicate_info(
            full_trajectory, communicate_info
        )
        
        # Check if extra_communicate_info exists (optional attribute)
        extra_communicate_info = getattr(task.evaluation_criteria, 'extra_communicate_info', None)
        if extra_communicate_info:
            extra_communicate_info_checks = cls.evaluate_extra_communicate_info(full_trajectory, extra_communicate_info)
            communicate_info_checks.extend(extra_communicate_info_checks)

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.met for result in communicate_info_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            communicate_checks=communicate_info_checks,
            reward_breakdown={RewardType.COMMUNICATE: reward},
        )

    @classmethod
    def evaluate_communicate_info(
        cls,
        full_trajectory: list[Message],
        communicate_info: list[str],
    ) -> list[CommunicateCheck]:
        """
        Evaluate whether the agent communicates the information correctly.
        """
        if len(communicate_info) == 0:
            return []

        outputs = []
        for info_str in communicate_info:
            found = False
            for message in full_trajectory:
                if not isinstance(message, AssistantMessage):
                    continue
                if not message.has_text_content():
                    continue
                if info_str.lower() in message.content.lower().replace(
                    ",", ""
                ):  # TODO: This could be improved!
                    found = True
                    break
            if found:
                met = True
                justification = f"Information '{info_str}' communicated in the message:\n '{message.content}'"
            else:
                met = False
                justification = f"Information '{info_str}' not communicated."
            outputs.append(
                CommunicateCheck(
                    info=info_str,
                    met=met,
                    justification=justification,
                )
            )
        return outputs

    @classmethod
    def evaluate_extra_communicate_info(
        cls,
        full_trajectory: list[Message],
        extra_communicate_info: list[str],
    ) -> list[CommunicateCheck]:
        """
        Evaluate whether the agent communicates the information correctly.
        """
        if len(extra_communicate_info) == 0:
            return []

        outputs = []
        # extra_communicate_info={
        #     "info_key": [info_value1, info_value2, ...],
        # }
        for info_key, info_value_list in extra_communicate_info.items():
            found = False
            for message in full_trajectory:
                if not isinstance(message, AssistantMessage):
                    continue
                if not message.has_text_content():
                    continue
                if any(info_value.lower() in message.content.lower().replace(",", "") for info_value in info_value_list):
                    found = True
                    break
            if found:
                met = True
                justification = f"Information '{info_key}' communicated in the message:\n '{message.content}'"
            else:
                met = False
                justification = f"Information '{info_key}' not communicated."
            outputs.append(
                CommunicateCheck(
                    info=info_key,
                    met=met,
                    justification=justification,
                )
            )
        return outputs
