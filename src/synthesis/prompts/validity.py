"""Phase-3 prompts: an LLM judge scores each (task, exploration trajectory) pair.

Each template is filled with ``str.format``; literal braces are doubled.
Scores are 1-10 per criterion; ``synthesis.validate`` averages them into
``overall_score``, which ``synthesis.filter_tasks`` thresholds.

  general    - realism, necessity, correctness
  changing   - realism, necessity, correctness (+ changing_scenario_relevance gate)
  infeasible - realism, necessity, correctness, infeasibility_reasonability
"""

# Inserted into general/changing prompts for the airline domain only.
AIRLINE_TIME_BLOCK = """

# DOMAIN TIME

The current time in the airline domain is fixed at **2024-05-15 15:00 EST**.
The user is calling at this moment, so:
- Flights dated before 2024-05-15 are in the past — the user cannot legitimately ask to change them or cancel them ahead of departure; they can only ask about what happened (status, refund, compensation for cancellation/delay).
- Flights on 2024-05-15 are today; flights after 2024-05-15 are upcoming.
- A user-narrated motivation that contradicts the call-time anchor (e.g., asking what will happen with a flight whose status is already `landed`) should be flagged as a REALISM issue.
"""

GENERAL_JUDGE_PROMPT = """You are an expert judge evaluating the validity of a customer service task and its verification trajectory.

IMPORTANT CONTEXT:
- The trajectory was generated via **exploration**, and may include redundant, trial-and-error, or noisy actions.
- You should NOT penalize unnecessary intermediate actions **as long as**:
  1) The actions does not change the FINAL DATABASE STATE
  2) All REQUIRED USER-FACING COMMUNICATION is correctly provided
- Your evaluation should focus on **final effects**, not on whether the agent followed an optimal or minimal path.

{domain_time_block}
# TASK DESCRIPTION

Domain: {domain}

User Scenario (from user's perspective):
- Reason for Call: {reason_for_call}
- Known Information: {known_info}
- Unknown Information: {unknown_info}
- Task Instructions: {task_instructions}


# TRAJECTORY (Exploration Execution Trace)

This is the full execution trace produced by the agent during exploration.
It may include redundant or exploratory actions.

You should evaluate correctness and necessity **based on the FINAL STATE** implied by this trajectory, not on the exact action path.

{trajectory_text}


# REQUIRED COMMUNICATION WITH USER

This specifies the information that MUST be communicated to the user (e.g., price difference, refund amount).

Only evaluate whether this information is:
- Required by the task
- Correctly supported by the execution trace

{communication_text}


# EVALUATION TASK

Please evaluate this task-trajectory pair on three criteria:

## 1. REALISM (Score 1-10)
Is the user's request realistic?
- Would a real user plausibly make this request?
- Is the motivation clear and reasonable?
- Are the details (names, order numbers, product types) enough to uniquely identify what needs to be done?
- Is the user's narration consistent with the domain's current time (see DOMAIN TIME above, when present)? Down-score if the user talks about a past flight as if it is still upcoming, or about a future flight as if it has already happened.

## 2. NECESSITY (Score 1-10)
Does the trajectory achieve all the effects required by the task?
- Are there missing critical actions that should have been taken?
- Are there any **database write effects ** (e.g., modify order information) that contradict or exceed what the task instructions allow?
- Is the communication information explicitly mentioned in the user's request? (e.g. if the user asks for the price difference or total price, the agent should tell the number)

## 3. CORRECTNESS (Score 1-10)
Does the task description match what the trajectory accomplishes?
- Does the trajectory solve the user's stated problem?
- Are the actions aligned with the task description?
- Is the information flow logical?
- Do the tool call arguments match the task context?
- If there is any communication information related to price, is it correctly supported by the execution trace and does it match the task instructions?

# OUTPUT FORMAT

Provide your evaluation as a JSON object with the following structure:

{{
  "evaluations": {{
    "realism": {{
      "score": <1-10>,
      "explanation": "<why you gave this score>",
      "issues": [<list of specific issues found, or empty array>]
    }},
    "necessity": {{
      "score": <1-10>,
      "explanation": "<why you gave this score>",
      "issues":
      [
        {{
          "missing_critical_actions": <list of missing required actions to achieve the task>
        }},
        {{
          "unauthorized_write_actions": <list of actions that write to the database but are not mentioned in the task instructions>
        }},
        {{
          "unmotivated_communication": <list of communication items not asked by the task scenario>
        }}
      ]    
      }},
    "correctness": {{
      "score": <1-10>,
      "explanation": "<why you gave this score>",
      "issues": 
      [
        {{
          "unsupported_communication": <list of communication items not match the task instructions>
        }},
        {{
            "other_issues": <list of other issues found>
        }}
      ]
    }},
  }},
  "suggestions": [<list of suggestions for improvement>],
}}

Return ONLY the JSON object, no additional text or explanation outside the JSON."""

CHANGING_JUDGE_PROMPT = """You are an expert judge evaluating the validity of a customer service task and its verification trajectory.

IMPORTANT CONTEXT:
- The trajectory was generated via **exploration**, and may include redundant, trial-and-error, or noisy actions.
- You should NOT penalize unnecessary intermediate actions **as long as**:
  1) The actions does not change the FINAL DATABASE STATE
  2) All REQUIRED USER-FACING COMMUNICATION is correctly provided
- Your evaluation should focus on **final effects**, not on whether the agent followed an optimal or minimal path.

{domain_time_block}
# TASK DESCRIPTION

Domain: {domain}

User Scenario (from user's perspective):
- Reason for Call: {reason_for_call}
- Known Information: {known_info}
- Unknown Information: {unknown_info}
- Task Instructions: {task_instructions}


# TRAJECTORY (Exploration Execution Trace)

This is the full execution trace produced by the agent during exploration.
It may include redundant or exploratory actions.

You should evaluate correctness and necessity **based on the FINAL STATE** implied by this trajectory, not on the exact action path.

{trajectory_text}


# REQUIRED COMMUNICATION WITH USER

This specifies the information that MUST be communicated to the user (e.g., price difference, refund amount).

Only evaluate whether this information is:
- Required by the task
- Correctly supported by the execution trace

{communication_text}


# EVALUATION TASK

Please evaluate this task-trajectory pair on four criteria:

## 1. REALISM (Score 1-10)
Is the user's request realistic?
- Would a real user plausibly make this request?
- Is the motivation clear and reasonable?
- Are the details (names, order numbers, product types) enough to uniquely identify what needs to be done?
- Is the user's narration consistent with the domain's current time (see DOMAIN TIME above, when present)? Down-score if the user talks about a past flight as if it is still upcoming, or about a future flight as if it has already happened.

## 2. NECESSITY (Score 1-10)
Does the trajectory achieve all the effects required by the task?
- Are there missing critical actions that should have been taken?
- Are there any **database write effects ** (e.g., modify order information) that contradict or exceed what the task instructions allow?
- Is the communication information explicitly mentioned in the user's request? (e.g. if the user asks for the price difference or total price, the agent should tell the number)

## 3. CORRECTNESS (Score 1-10)
Does the task description match what the trajectory accomplishes?
- Does the trajectory solve the user's stated problem?
- Are the actions aligned with the task description?
- Is the information flow logical?
- Do the tool call arguments match the task context?
- If there is any communication information related to price, is it correctly supported by the execution trace and does it match the task instructions?

## 4. CHANGING_SCENARIO_RELEVANCE (True/False)
Does the user task involve changing of user intent? This includes but is not limited to:
- Change of mind when dealing with a specific issue (e.g., initially wants to cancel the order, but during confirmation decides to exchange it instead).
- Dealing with multiple issues (e.g., the user wants to cancel the order and change the payment method, or the user found out that the order is not eligible for exchange and ask for human help)
- The user have multiple requests that are not related to each other (e.g., the user wants to cancel the order and modify the address)
Mark as False when the user expresses a single stable intent, even if it reflects a changed personal preference (e.g., the user says they want to return a delivered item because they changed their mind, this is still one intent: returning the item).


# OUTPUT FORMAT

Provide your evaluation as a JSON object with the following structure:

{{
  "evaluations": {{
    "realism": {{
      "score": <1-10>,
      "explanation": "<why you gave this score>",
      "issues": [<list of specific issues found, or empty array>]
    }},
    "necessity": {{
      "score": <1-10>,
      "explanation": "<why you gave this score>",
      "issues":
      [
        {{
          "missing_critical_actions": <list of missing required actions to achieve the task>
        }},
        {{
          "unauthorized_write_actions": <list of actions that write to the database but are not mentioned in the task instructions>
        }},
        {{
          "unmotivated_communication": <list of communication items not asked by the task scenario>
        }}
      ]    
      }},
    "correctness": {{
      "score": <1-10>,
      "explanation": "<why you gave this score>",
      "issues": 
      [
        {{
          "unsupported_communication": <list of communication items not match the task instructions>
        }},
        {{
            "other_issues": <list of other issues found>
        }}
      ]
    }},
    "changing_scenario_relevance": {{
      "score": <True/False>,
      "explanation": "<why you gave this score>",
      "issues": [<list of specific issues found, or empty array>]
    }},
  }},
  "suggestions": [<list of suggestions for improvement>],
}}

Return ONLY the JSON object, no additional text or explanation outside the JSON."""

INFEASIBLE_JUDGE_PROMPT = """You are an expert judge evaluating the validity of a customer service task. This scenario is an INFEASIBLE scenario, where the user's request is infeasible to be handled by the agent. The user might have several intentions in the task. The task might terminate after the infeasible request is made during simulation. You are provided with the user scenario, the domain policy, the prohibited actions, the required actions, the required communication information, and an toolcall trajectory. Note: The toolcall trajectory was generated via **exploration**, and may include redundant, trial-and-error, or noisy actions. It's not the final trajectory that the agent will follow. You can use the trajectory to understand the user's intentions, the agent's actions, and scenario context, but you should not use the trajectory to directly evaluate the correctness and necessity of the task. 


# TASK DESCRIPTION

Domain: {domain}

User Scenario (from user's perspective):
- Reason for Call: {reason_for_call}
- Known Information: {known_info}
- Unknown Information: {unknown_info}
- Task Instructions: {task_instructions}



# DOMAIN POLICY
{domain_policy}


# Why This Task Is Infeasible:
{infeasible_reason}

# Forbidden Actions:
{forbidden_actions_text}

# Required Actions:
{required_actions_text}


# REQUIRED COMMUNICATION WITH USER

This specifies the information that MUST be communicated to the user (e.g., price difference, refund amount).

Only evaluate whether this information is:
- Required by the task
- Correctly supported by the execution trace

{communication_text}


# TRAJECTORY (Exploration Execution Trace)

This is the full execution trace produced by the agent during exploration.
It may include redundant or exploratory actions.

You should evaluate correctness and necessity **based on the FINAL STATE** implied by this trajectory, not on the exact action path.

{trajectory_text}

# EVALUATION TASK

For INFEASIBLE scenarios, evaluate whether the user's request cannot be fulfilled and the prohibited actions and required actions are reasonable and well-specified. Please evaluate this task-trajectory pair on four criteria:


## 1. REALISM (Score 1-10)
- Would a real user plausibly make this request?
- Is the motivation clear and reasonable?
- Are the details (names, order numbers, product types) enough to uniquely identify what needs to be done?
- Does the user's lack of policy knowledge seem natural?

## 2. NECESSITY (Score 1-10)
- If any part of the task is feasible, are all the required actions and communication information needed to achieve the feasible part of the task before the infeasible request is made?
- Are there any **database write effects ** (e.g., modify order information) in the required actions that contradict or exceed what the task instructions allow?
- Is the communication information explicitly mentioned in the user's request? (e.g. if the user asks for the price difference or total price, the agent should tell the number)

## 3. CORRECTNESS (Score 1-10)
- Are the required actions well defined (tool call names / arguments)?
- Are the tool call arguments in the required actions reasonable?
- If there is any communication information related to price, is it correctly supported by the toolcall trajectory?

## 4. INFEASIBILITY_REASONABILITY (Score 1-10)
- Do the forbidden actions genuinely need to be avoided (align with the stated policy and the infeasibility reason)?
- Are forbidden actions well defined (tool call names / arguments)?
- Are the tool call arguments in the forbidden actions reasonable?
- Is the infeasible_reason coherent with the domain and the user request?

# OUTPUT FORMAT

Provide your evaluation as a JSON object with the following structure:

{{
  "evaluations": {{
    "realism": {{
      "score": <1-10>,
      "explanation": "<why you gave this score>",
      "issues": [<list of specific issues found, or empty array>]
    }},
    "necessity": {{
      "score": <1-10>,
      "explanation": "<why you gave this score>",
      "issues":
      [
        {{
          "unreasonable_required_actions": <list of required actions that are not reasonable>
        }},
        {{
          "unmotivated_communication": <list of communication items not asked by the task scenario>
        }}
      ]    
      }},
    "correctness": {{
      "score": <1-10>,
      "explanation": "<why you gave this score>",
      "issues": 
      [
        {{
          "underdefined_required_actions": <list of required actions that are not well defined>
        }},
        {{
          "other_issues": <list of other issues found>
        }}
      ]
    }},
    "infeasibility_reasonability": {{
      "score": <1-10>,
      "explanation": "<why you gave this score>",
      "issues": 
      [
        {{
          "underdefined_forbidden_actions": <list of forbidden actions that are not well defined>
        }},
        {{
          "other_issues": <list of other issues found>
        }}
      ]
    }},
  }},
  "suggestions": [<list of suggestions for improvement>],
}}
"""

JUDGE_PROMPTS = {
    "general": GENERAL_JUDGE_PROMPT,
    "changing": CHANGING_JUDGE_PROMPT,
    "infeasible": INFEASIBLE_JUDGE_PROMPT,
}

# Criteria averaged into ``overall_score`` for each task type.
SCORED_CRITERIA = {
    "general": ["realism", "necessity", "correctness"],
    "changing": ["realism", "necessity", "correctness"],
    "infeasible": ["realism", "necessity", "correctness", "infeasibility_reasonability"],
}
