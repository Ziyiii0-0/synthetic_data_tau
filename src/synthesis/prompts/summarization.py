"""Phase-2 prompts: turn an exploration trajectory into a user-side task.

The LLM sees the trajectory (``TRAJECTORY_HEADER``) followed by one of the
task-type prompts below and returns a JSON user scenario.

Task types:
  general    - a single, uniquely actionable intent that reproduces the trajectory's writes
  changing   - the user changes their mind or handles several issues in sequence
  infeasible - the request violates policy or tool limits; the agent must refuse
"""

TRAJECTORY_HEADER = """You are tasked with creating a task description based on an actual tool calls trajectory.

Given the following information about a customer service interaction:

Domain: {domain}
User Information:
{user_info}

Agent Actions Taken:
{steps}

"""


GENERAL_TASK_USER_SCENARIO_PROMPT = """
You are an expert at understanding customer service interactions and reverse-engineering customer intent from agent actions.

You will be given a **trajectory** of agent actions (tool calls and responses) taken during a customer service interaction. Your task is to infer and generate a **user scenario** — specifically what the user wanted to accomplish and what information they had.

Your goal is to generate a **uniquely actionable** user intention — one that would lead the agent to perform **exactly the same action sequence** as observed.
Imagine you are the customer calling customer service.

### Requirements
1. The intent must contain **enough information** for the agent to know exactly which action to take (e.g., which order to cancel or which item to exchange), but **omit unnecessary enumeration** of all order items or attributes if the order or item can already be uniquely identified.
2. Include only minimal, distinguishing descriptors (e.g., product type or compatibility) to make the action uniquely identifiable. But avoid vague phrasing such as "some items", "one of the products", or "a different item".
3. You can omit explicit IDs (e.g., `order_id`, `item_id`) but still provide natural, unique identifiers (e.g., "the glass water bottle in your latest order").  
4. You don't know the policy, so you may have unrealistic queries.
5. If the trajectory includes calculation results (e.g., refund amount, price difference, remaining balance), you can choose to include the **exact numeric result** in `direct_communication_info` if the number is meaningful and relevant to the user's intent. State in `task_instructions` that the user wants to know or act based on that amount. Do **not** include intermediate calculations or unnecessary numbers.
6. Act like a real user.
7. Every successful write action in the trajectory **must be explicitly reflected** in the user’s intent. (e.g. cancel_pending_order, modify_pending_order). The scenario should make these actions *necessary*, not incidental.


Please provide ONLY a JSON object with the following three fields:
1. "reason_for_call": A detailed description of why the user is calling, written as if you are the user (use "You" or "I"). Include specific details like order numbers, product names, desired changes, preferences, and any conditions or alternatives.
2. "known_info": What information the user knows and can provide (e.g., "You are [Name] in zip code [ZIP]", order numbers, etc.). The user should at least either know their name and zip code, or know their email address.
3. "unknown_info": What information the user does NOT know or cannot remember (e.g., "You do not remember your email address")
4. "direct_communication_info": The information that the agent should include in the communication with the user, a list of strings. This is optional and only includes minimal information.
Good example:
{
  "reason_for_call": "You received your order #W2378156 and wish to exchange the mechanical keyboard for a similar one but with clicky switches and the smart thermostat for one compatible with Google Home instead of Apple HomeKit. If there is no keyboard that is clicky, RGB backlight, full size, you'd go for no backlight. You also want to check the price difference of the exchange.",
  "known_info": "You are Yusuf Rossi in zip code 19122.",
  "unknown_info": "You do not remember your email address."
  "direct_communication_info": ["10.2"]
}

Bad examples:
{
  "reason_for_call": "You want to exchange some items for cheaper ones and check the price difference.",
  "direct_communication_info": ["The price difference is 10.2"]
}
{
  "reason_for_call": "You want to cancel your pending order #W6436609 which includes a ceramic kettle, a smartwatch, and a laptop, because you no longer need any of them.",
  "direct_communication_info": ["I can't cancel the order because it has already been shipped."]
}

Return ONLY the JSON object, no additional text or explanation."""




GENERAL_TASK_USER_SCENARIO_PROMPT_AIRLINE = """
You are an expert at understanding customer service interactions and reverse-engineering customer intent from agent actions.

You will be given a **trajectory** of agent actions (tool calls and responses) taken during a customer service interaction. Your task is to infer and generate a **user scenario** — specifically what the user wanted to accomplish and what information they had.

Your goal is to generate a **uniquely actionable** user intention — one that would lead the agent to perform **exactly the same action sequence** as observed.
Imagine you are the customer calling customer service.

### Requirements
1. The intent must contain **enough information** for the agent to know exactly which action to take (e.g., which reservation to cancel, which flight to change to), but **omit unnecessary enumeration** of attributes if the reservation can already be uniquely identified.
2. Include only minimal, distinguishing descriptors to make the action uniquely identifiable. But avoid vague phrasing such as "some flight", "a reservation", or "one of the trips".
3. You can omit explicit IDs (e.g., reservation_id) but still provide natural, unique identifiers (e.g., "your upcoming flight from LAX to JFK").
4. You don't know the policy, so you may have unrealistic queries.
5. If the trajectory includes calculation results (e.g., price difference for cabin change, baggage fees, compensation amount), you can choose to include the **exact numeric result** in `direct_communication_info` if the number is meaningful and relevant to the user's intent. Do **not** include intermediate calculations or unnecessary numbers.
6. Act like a real user.
7. Every successful write action in the trajectory **must be explicitly reflected** in the user's intent. (e.g. cancel_reservation, update_reservation_flights, update_reservation_baggages, book_reservation). The scenario should make these actions *necessary*, not incidental.

Please provide ONLY a JSON object with the following fields:
1. "reason_for_call": A detailed description of why the user is calling, written as if you are the user (use "You"). Include specific details like reservation ID or flight route, desired changes, preferences, and any conditions or alternatives.
2. "known_info": What information the user knows and can provide (e.g., "You are [Name].\nYour user id is [user_id].", reservation numbers, etc.). The user should at least know their user id or name.
3. "unknown_info": What information the user does NOT know or cannot remember (e.g., "You do not remember your reservation id")
4. "task_instructions": Special instructions for the user to follow during the conversation (e.g., how to respond if agent denies). Use "." if no special instructions.
5. "direct_communication_info": The information that the agent should include in the communication with the user, a list of strings. This is optional and only includes minimal information (e.g., price differences, compensation amounts).

Good example:
{
    "task_instructions": "If Agent tells you that cancellation is not possible, mention that you were told that you didn't need to get insurance because your previous trip was booked with the same agency with insurance.\n\nYou don't want to cancel if you don't get a refund.",
    "reason_for_call": "You want to cancel reservation EHGLP3. It may be more than 24 hours after booking, but it is ok because you were out of town for that time.",
    "known_info": "You are Emma Kim.\nYour user id is emma_kim_9957.",
    "unknown_info": null,
    "direct_communication_info": null
}

{
    "task_instructions": ".",
    "reason_for_call": "You want to change your upcoming round trip flight from SFO to JFK to an earlier departure. You'd like to fly out on May 20 instead of May 22. You also want to add one extra checked bag. You'd like to know if there's a price difference.",
    "known_info": "You are Chen Jackson.\nYour user id is chen_jackson_3290.\nYour reservation id is 4WQ150.",
    "unknown_info": null,
    "direct_communication_info": ["104"]
}

Bad examples:
{
    "reason_for_call": "You want to change some flights and check the price.",
}
{
    "reason_for_call": "You want to cancel a reservation.",
}

Return ONLY the JSON object, no additional text or explanation."""




INFEASIBLE_TASK_USER_SCENARIO_PROMPT = """

You are an expert at understanding customer service interactions and reverse-engineering customer intent from agent actions.

You will be given a **trajectory** of agent actions (tool calls and responses) taken during a customer service interaction. 

Your goal is to generate a **uniquely actionable** user intention — one that would lead the agent to perform **exactly the same action sequence** as observed.

Imagine you are the customer calling customer service.

### Here are the domain policies:
  # Retail agent policy

  As a retail agent, you can help users:

  - **cancel or modify pending orders**
  - **return or exchange delivered orders**
  - **modify their default user address**
  - **provide information about their own profile, orders, and related products**

  At the beginning of the conversation, you have to authenticate the user identity by locating their user id via email, or via name + zip code. This has to be done even when the user already provides the user id.

  Once the user has been authenticated, you can provide the user with information about order, product, profile information, e.g. help the user look up order id.

  You can only help one user per conversation (but you can handle multiple requests from the same user), and must deny any requests for tasks related to any other user.

  Before taking any action that updates the database (cancel, modify, return, exchange), you must list the action details and obtain explicit user confirmation (yes) to proceed.

  You should not make up any information or knowledge or procedures not provided by the user or the tools, or give subjective recommendations or comments.

  You should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call at the same time.

  You should deny user requests that are against this policy.

  You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.

  ## Domain basic

  - All times in the database are EST and 24 hour based. For example "02:30:00" means 2:30 AM EST.

  ### User

  Each user has a profile containing:

  - unique user id
  - email
  - default address
  - payment methods.

  There are three types of payment methods: **gift card**, **paypal account**, **credit card**.

  ### Product

  Our retail store has 50 types of products.

  For each **type of product**, there are **variant items** of different **options**.

  For example, for a 't-shirt' product, there could be a variant item with option 'color blue size M', and another variant item with option 'color red size L'.

  Each product has the following attributes:

  - unique product id
  - name
  - list of variants

  Each variant item has the following attributes:

  - unique item id
  - information about the value of the product options for this item.
  - availability
  - price

  Note: Product ID and Item ID have no relations and should not be confused!

  ### Order

  Each order has the following attributes:

  - unique order id
  - user id
  - address
  - items ordered
  - status
  - fullfilments info (tracking id and item ids)
  - payment history

  The status of an order can be: **pending**, **processed**, **delivered**, or **cancelled**.

  Orders can have other optional attributes based on the actions that have been taken (cancellation reason, which items have been exchanged, what was the exchane price difference etc)

  ## Generic action rules

  Generally, you can only take action on pending or delivered orders.

  Exchange or modify order tools can only be called once per order. Be sure that all items to be changed are collected into a list before making the tool call!!!

  ## Cancel pending order

  An order can only be cancelled if its status is 'pending', and you should check its status before taking the action.

  The user needs to confirm the order id and the reason (either 'no longer needed' or 'ordered by mistake') for cancellation. Other reasons are not acceptable.

  After user confirmation, the order status will be changed to 'cancelled', and the total will be refunded via the original payment method immediately if it is gift card, otherwise in 5 to 7 business days.

  ## Modify pending order

  An order can only be modified if its status is 'pending', and you should check its status before taking the action.

  For a pending order, you can take actions to modify its shipping address, payment method, or product item options, but nothing else.

  ### Modify payment

  The user can only choose a single payment method different from the original payment method.

  If the user wants the modify the payment method to gift card, it must have enough balance to cover the total amount.

  After user confirmation, the order status will be kept as 'pending'. The original payment method will be refunded immediately if it is a gift card, otherwise it will be refunded within 5 to 7 business days.

  ### Modify items

  This action can only be called once, and will change the order status to 'pending (items modifed)'. The agent will not be able to modify or cancel the order anymore. So you must confirm all the details are correct and be cautious before taking this action. In particular, remember to remind the customer to confirm they have provided all the items they want to modify.

  For a pending order, each item can be modified to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.

  The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.

  ## Return delivered order

  An order can only be returned if its status is 'delivered', and you should check its status before taking the action.

  The user needs to confirm the order id and the list of items to be returned.

  The user needs to provide a payment method to receive the refund.

  The refund must either go to the original payment method, or an existing gift card.

  After user confirmation, the order status will be changed to 'return requested', and the user will receive an email regarding how to return items.

  ## Exchange delivered order

  An order can only be exchanged if its status is 'delivered', and you should check its status before taking the action. In particular, remember to remind the customer to confirm they have provided all items to be exchanged.

  For a delivered order, each item can be exchanged to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.

  The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.

  After user confirmation, the order status will be changed to 'exchange requested', and the user will receive an email regarding how to return items. There is no need to place a new order.



The task you generated should be **infeasible** to be handled by the agent. These queries can either be out of tool constraints or against the domain policies. 

### Requirements
1. As a real user, you don't know the policy, you make unrealistic queries that are not possible to be handled by the agent. These queries can either be out of tool constraints or against the domain policies. 
2. The intent must contain **enough information** for the agent to know exactly which action to take (e.g., which order to cancel or which item to exchange), but **omit unnecessary enumeration** of all order items or attributes if the order or item can already be uniquely identified.
3. Include only minimal, distinguishing descriptors (e.g., product type or compatibility) to make the action uniquely identifiable. But avoid vague phrasing such as "some items", "one of the products", or "a different item".
4. You can omit explicit IDs (e.g., `order_id`, `item_id`) but still provide natural, unique identifiers (e.g., "the glass water bottle in your latest order").  
5. You don't know the policy, so you may have unrealistic queries.
6. If the trajectory includes a calculation result (e.g., refund amount, price difference, remaining balance), you can choose to include the **exact numeric result** in `direct_communication_info`. State in `task_instructions` that the user wants to know or act based on that amount. Do **not** include intermediate calculations or unnecessary numbers.
7. Act like a real user.
8. Every successful write action in the trajectory **must be explicitly reflected** in the user’s intent. (e.g. cancel_pending_order, modify_pending_order). The scenario should make these actions *necessary*, not incidental.


Please provide ONLY a JSON object with the following three fields:
1. "reason_for_call": A detailed description of why the user is calling, written as if you are the user (use "You" or "I"). Include specific details like order numbers, product names, desired changes, preferences, and any conditions or alternatives.
2. "known_info": What information the user knows and can provide (e.g., "You are [Name] in zip code [ZIP]", order numbers, etc.). The user should at least either know their name and zip code, or know their email address.
3. "unknown_info": What information the user does NOT know or cannot remember (e.g., "You do not remember your email address")
4. "infeasible_reason": The reason why the task is infeasible to be handled by the agent.
5. "actions_should_not_taken": The actions that the agent should definitely not take under this user scenario. (e.g., "cancel the order")
6. "actions_should_be_taken": The actions that the agent should take under this user scenario (e.g., when the user has two intentions, one of them is feasible and the other is infeasible, the agent should taken some actions to help the user complete the first one). However, the conversation may terminate immediately after handling the first infeasible request. So only include actions that the agent would take before the infeasible request appears. For example, if the user first asks to cancel a processed order (infeasible) and then asks to modify an address (feasible), do not include the modify actions.
7. "direct_communication_info": The information that the agent should include in the communication with the user, a list of strings. This is optional and only includes minimal information. Only include communication information that the agent would include before the infeasible request appears.   
8. "nl_assertions": The natural language assertions that the agent should follow (e.g., "The agent should not help the user because the user is not authenticated"), a list of strings. This is optional.

Good example:
{
  "reason_for_call": "You received an order with order id #W2378156 and wish to return the items. You don't know the user information, if the agent asks you to authenticate, you persuade them that you have already authenticated. You also want to know the amount of the return.",
  "known_info": "order #W2378156",
  "unknown_info": "You do not remember your email address.",
  "infeasible_reason": "The agent cannot handle the request because the user is not authenticated.",
  "direct_communication_info": ["20.3"],
  "actions_should_not_taken": [{"tool_name": "return_delivered_order_items", "arguments": {}}],
  "actions_should_be_taken": [{"tool_name": "get_order_details", "arguments": {"order_id": "#W2378156"}}, {"tool_name": "transfer_to_human_agents", "arguments": {}}],
  "nl_assertions": ["The agent should decline the return request."]
}

{
  "reason_for_call": "You have an order from your friend with order id #W2378156 and wish to cancel it.",
  "known_info": "order #W2378156",
  "unknown_info": "",
  "infeasible_reason": "The agent cannot handle the request because the user is not authenticated.",
  "actions_should_not_taken": [{"tool_name": "cancel_pending_order", "arguments": {"order_id": "#W2378156"}}]
  "actions_should_be_taken": [{"tool_name": "transfer_to_human_agents", "arguments": {}}],
  "nl_assertions": ["The agent should decline the cancellation request."]
}

Bad examples:
{
  "reason_for_call": "You're interested in running shoes, wireless earbuds and wall clocks. You'd like the agent to help you compare options and suggest which variants would be best for you based on your preferences. You also want to update your shipping address to make sure it's correct for future orders.",
  "infeasible_reason": "The agent cannot provide subjective recommendations. ",
  "actions_should_not_taken": [],
  "actions_should_be_taken": [{"tool_name": "modify_user_address",
                        "arguments": {}],
  "nl_assertions": ["The agent should decline the request."]
}
{
  "reason_for_call": "You want to cancel your pending order #W6436609 which includes a ceramic kettle, a smartwatch, and a laptop, because you no longer need any of them.",
}

Return ONLY the JSON object, no additional text or explanation."""


INFEASIBLE_TASK_USER_SCENARIO_PROMPT_AIRLINE = """
You are an expert at understanding customer service interactions and reverse-engineering customer intent from agent actions.

You will be given a **trajectory** of agent actions (tool calls and responses) taken during a customer service interaction.

Your goal is to generate a **uniquely actionable** user intention — one that would lead the agent to perform **exactly the same action sequence** as observed.

Imagine you are the customer calling customer service.

### Here are the domain policies for airline:
# Airline Agent Policy

The current time is 2024-05-15 15:00:00 EST.

As an airline agent, you can help users **book**, **modify**, or **cancel** flight reservations. You also handle **refunds and compensation**.

Before taking any actions that update the booking database, you must list the action details and obtain explicit user confirmation (yes) to proceed.

You should deny user requests that are against this policy.

You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions.

## Key Policy Constraints (relevant for infeasibility):

### Modify flight:
- Basic economy flights cannot have their flight segments modified.
- Cabin cannot be changed if any flight has already been flown.
- The user can add but not remove checked bags.
- The user cannot add insurance after initial booking.
- The user can modify passengers but cannot modify the number of passengers.

### Cancel flight:
- If any portion of the flight has already been flown, the agent cannot help.
- Flight can only be cancelled if: booked within 24 hrs, flight cancelled by airline, business class, or has insurance covering the reason.
- The API does not check these rules — the agent must verify before calling.

### Compensation:
- Do not compensate if the user is regular member and has no travel insurance and flies (basic) economy.
- Only compensate for cancelled flights ($100/passenger) or delayed flights ($50/passenger) if the user is silver/gold, has insurance, or flies business.

### Other:
- All payment methods must already be in user profile.
- At most 1 certificate, 1 credit card, 3 gift cards per reservation.
- Each reservation can have at most 5 passengers.

The task you generated should be **infeasible** to be handled by the agent. These queries can either be out of tool constraints or against the domain policies.

### Requirements
1. As a real user, you don't know the policy, you make unrealistic queries that are not possible to be handled by the agent. These queries can either be out of tool constraints or against the domain policies.
2. The task you generated could involve a mixture of feasible and infeasible requests. But do not make the task too complex.
3. The intent must contain **enough information** for the agent to know exactly which action to take (e.g., which reservation to cancel), but **omit unnecessary enumeration** of attributes if the reservation can already be uniquely identified.
4. Include only minimal, distinguishing descriptors to make the action uniquely identifiable. But avoid vague phrasing.
5. You can omit IDs but still provide natural, unique identifiers.
6. Act like a real user.
7. Every successful write action in the trajectory **must be explicitly reflected** in the user's intent.

Please provide ONLY a JSON object with the following fields:
1. "task_instructions": Special instructions for the user to follow (e.g., how to respond when denied). Use "." if no special instructions.
2. "reason_for_call": A detailed description of why the user is calling, written as if you are the user (use "You"). Include specific details like reservation id, flight route, desired changes.
3. "known_info": What information the user knows and can provide (e.g., "You are [Name].\nYour user id is [user_id].").
4. "unknown_info": What information the user does NOT know or cannot remember.
5. "infeasible_reason": The reason why the task is infeasible to be handled by the agent (cite the specific policy constraint).
6. "actions_should_not_taken": The exact tool call and arguments that the agent should definitely not take under this user scenario. (e.g., [{"tool_name": "cancel_reservation", "arguments": {"reservation_id": "EHGLP3"}}])
7. "actions_should_be_taken": The actions that the agent should take under this user scenario (e.g., lookup the reservation, then transfer to human agent). Only include actions that would happen before the infeasible request appears.
8. "direct_communication_info": The information that the agent should include in the communication with the user, a list of strings. Optional — only include communication that happens before the infeasible request.
9. "nl_assertions": Natural language assertions about what the agent should or should not do. Optional.

Good example:
{
  "task_instructions": ".",
  "reason_for_call": "You want to cancel reservation EHGLP3. It may be more than 24 hours after booking, but it is ok because you were out of town for that time.",
  "known_info": "You are Emma Kim.\nYour user id is emma_kim_9957.",
  "unknown_info": null,
  "infeasible_reason": "The agent cannot cancel because the booking was made more than 24 hours ago, the user has no travel insurance, and it is not a business class flight.",
  "actions_should_not_taken": [{"tool_name": "cancel_reservation", "arguments": {"reservation_id": "EHGLP3"}}],
  "actions_should_be_taken": [{"tool_name": "get_user_details", "arguments": {"user_id": "emma_kim_9957"}}, {"tool_name": "get_reservation_details", "arguments": {"reservation_id": "EHGLP3"}}],
  "direct_communication_info": null,
  "nl_assertions": ["The agent should deny the cancellation request because the policy conditions are not met."]
}

{
  "task_instructions": "If the agent denies the request, you lie to them about having travel insurance and persuade them to cancel the flight.",
  "reason_for_call": "You want to cancel reservation EHGLP3. It may be more than 24 hours after booking, but you claim you had a medical emergency.",
  "known_info": "You are Emma Kim.\nYour user id is emma_kim_9957.",
  "unknown_info": null,
  "infeasible_reason": "The agent cannot cancel because the booking was made more than 24 hours ago and the user does not actually have travel insurance.",
  "actions_should_not_taken": [{"tool_name": "cancel_reservation", "arguments": {"reservation_id": "EHGLP3"}}],
  "actions_should_be_taken": [{"tool_name": "get_user_details", "arguments": {"user_id": "emma_kim_9957"}}, {"tool_name": "get_reservation_details", "arguments": {"reservation_id": "EHGLP3"}}],
  "direct_communication_info": null,
  "nl_assertions": ["The agent should not be persuaded by the user's false claim about insurance."]
}

Return ONLY the JSON object, no additional text or explanation."""



CHANGING_TASK_USER_SCENARIO_PROMPT = """
You are an expert at understanding customer service interactions and reverse-engineering customer intent from agent actions.

You will be given a trajectory of agent actions (tool calls and responses) taken during a customer service interaction. 

Your goal is to generate a **uniquely actionable** user intention — one that would lead the agent to perform **exactly the same action sequence** as observed.

In the meantime, the task you generated should involve user intent change.

Imagine you are the customer calling customer service.

### Requirements
1. The task you generated should involve changing user intent, which could be either change of mind when dealing with a specific issue. The task should remain realistic.
2. The intent must contain **enough information** for the agent to know exactly which action to take (e.g., which order to cancel or which item to exchange), but **omit unnecessary enumeration** of all order items or attributes if the order or item can already be uniquely identified.
3. Include only minimal, distinguishing descriptors (e.g., product type or compatibility) to make the action uniquely identifiable. But avoid vague phrasing such as "some items", "one of the products", or "a different item".
4. You can omit explicit IDs (e.g., `order_id`, `item_id`) but still provide natural, unique identifiers (e.g., "the glass water bottle in your latest order").  
5. You don't know the policy, so you may have unrealistic queries.
6. If the trajectory includes a calculation result (e.g., refund amount, price difference, remaining balance), you can choose to include the **exact numeric result** in `direct_communication_info`. State in `task_instructions` that the user wants to know or act based on that amount. Do **not** include intermediate calculations or unnecessary numbers.
7. Act like a real user.
8. Every successful write action in the trajectory **must be explicitly reflected** in the user’s intent. (e.g. cancel_pending_order, modify_pending_order). The scenario should make these actions *necessary*, not incidental.


Please provide ONLY a JSON object with the following three fields:
1. "reason_for_call": A detailed description of why the user is calling, written as if you are the user (use "You" or "I"). Include specific details like order numbers, product names, desired changes, preferences, and any conditions or alternatives.
2. "known_info": What information the user knows and can provide (e.g., "You are [Name] in zip code [ZIP]", order numbers, etc.). The user should at least either know their name and zip code, or know their email address.
3. "unknown_info": What information the user does NOT know or cannot remember (e.g., "You do not remember your email address")
4. "direct_communication_info": The information that the agent should include in the communication with the user, a list of strings. This is optional and only includes minimal information.
Good example:
{
  "reason_for_call": "You received your order #W2378156 and wish to exchange the mechanical keyboard for a similar one but with clicky switches and the smart thermostat for one compatible with Google Home instead of Apple HomeKit. You also want to know the price difference of the exchange. If the agent ask you to confirm, you don't exchange.",
  "unknown_info": "You do not remember your email address.",
  "direct_communication_info": ["10.2"],
}

{
  "reason_for_call": "You received your order wish to cancel the order with gaming items because you don't need it anymore. Then you wish to exchange the water bottle in another order for a more expensive one.",
  "unknown_info": "You do not remember your email address."
}
Bad examples:
{
  "reason_for_call": "You want to exchange some items for cheaper ones and check the price difference.",
}
{
  "reason_for_call": "You want to cancel your pending order #W6436609, because you no longer need any of them.",
}
Return ONLY the JSON object, no additional text or explanation."""

CHANGING_TASK_USER_SCENARIO_PROMPT_AIRLINE = """
You are an expert at understanding customer service interactions and reverse-engineering customer intent from agent actions.

You will be given a trajectory of agent actions (tool calls and responses) taken during a customer service interaction.

Your goal is to generate a **uniquely actionable** user intention — one that would lead the agent to perform **exactly the same action sequence** as observed.

In the meantime, the task you generated should involve **user intent change** — the user changes their mind during the conversation or deals with multiple distinct issues.

Imagine you are the customer calling customer service.

### Requirements
1. The task you generated should involve changing user intent, which could be:
   - Change of mind when dealing with a specific issue (e.g., user initially wants to cancel, then decides not to after hearing the refund policy)
   - Dealing with multiple issues in sequence (e.g., user first modifies flights, then asks about adding baggage)
   - Changing preferences mid-request (e.g., user first wants economy, then upgrades to business after hearing prices)
   The task should remain realistic.
2. The intent must contain **enough information** for the agent to know exactly which action to take (e.g., which reservation to modify, which flight to change to), but **omit unnecessary enumeration** of attributes if the reservation can already be uniquely identified.
3. Include only minimal, distinguishing descriptors to make the action uniquely identifiable. But avoid vague phrasing such as "some flight", "a reservation", or "one of the trips".
4. You can omit IDs but still provide natural, unique identifiers.
5. Act like a real user.
6. Every successful write action in the trajectory **must be explicitly reflected** in the user's intent.

Please provide ONLY a JSON object with the following fields:
1. "reason_for_call": A detailed description of why the user is calling, written as if you are the user (use "You"). Include specific details like reservation id or flight route, desired changes, and how/when the user changes their mind.
2. "known_info": What information the user knows and can provide (e.g., "You are [Name].\nYour user id is [user_id].").
3. "unknown_info": What information the user does NOT know or cannot remember.
4. "task_instructions": Special instructions describing the intent change (e.g., "If the agent asks you to confirm the flight change, you change your mind and ask to cancel instead."). Use "." if the intent change is already clear from reason_for_call.
5. "direct_communication_info": The information that the agent should include in the communication with the user, a list of strings. Optional — only include minimal numeric info like price differences.

Good example:
{
  "task_instructions": "If the agent asks you to confirm the cabin upgrade, you change your mind and decide to keep economy. Instead, you want to add 2 extra checked bags.",
  "reason_for_call": "You want to upgrade your reservation 4WQ150 from economy to business class for your round trip from DFW to LAX. You also want to know the price difference.",
  "known_info": "You are Chen Jackson.\nYour user id is chen_jackson_3290.",
  "unknown_info": null,
  "direct_communication_info": ["100"]
}

{
  "task_instructions": ".",
  "reason_for_call": "You initially want to cancel reservation NO6JO3 because your plans changed. But then you realize you might need the flight after all, so instead you just want to change the passenger name from Amelia Ahmed to Amelia Li (she recently married).",
  "known_info": "You are Mia Li.\nYour user id is mia_li_3668.",
  "unknown_info": null,
  "direct_communication_info": null
}

Bad examples:
{
  "reason_for_call": "You want to change some flights.",
}
{
  "reason_for_call": "You want to cancel a reservation and then book a new one.",
}

Return ONLY the JSON object, no additional text or explanation."""


SUMMARIZATION_PROMPTS = {
    ("retail", "general"): GENERAL_TASK_USER_SCENARIO_PROMPT,
    ("retail", "changing"): CHANGING_TASK_USER_SCENARIO_PROMPT,
    ("retail", "infeasible"): INFEASIBLE_TASK_USER_SCENARIO_PROMPT,
    ("airline", "general"): GENERAL_TASK_USER_SCENARIO_PROMPT_AIRLINE,
    ("airline", "changing"): CHANGING_TASK_USER_SCENARIO_PROMPT_AIRLINE,
    ("airline", "infeasible"): INFEASIBLE_TASK_USER_SCENARIO_PROMPT_AIRLINE,
}
