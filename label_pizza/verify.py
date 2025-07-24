from typing import Dict

# Below are three examples of verification functions.
def verify_non_empty(answers: Dict[str, str]) -> None:
    """Verify that answers are non-empty.
    
    Args:
        answers: Dictionary mapping question text to answer value
        
    Raises:
        ValueError: If any answer is empty
    """
    for question_text, answer in answers.items():
        if not answer.strip():
            raise ValueError(f"Answer for '{question_text}' cannot be empty")

def check_human_description(answers: Dict[str, str]) -> None:
    """Ensure the description answer is consistent with the count answer.

    Args:
        answers: Dictionary mapping question text to answer value
      
    Raises:
        ValueError: If the description is provided when there are no people, or if the description is not provided when there are people.
    """
    num_people = answers.get("Number of people?")
    description = answers.get("If there are people, describe them.")
    
    if num_people == "0" and description:
        raise ValueError("Description cannot be provided when there are no people")
    if num_people != "0" and not description:
        raise ValueError("Description must be provided when there are people")

def check_pizza_description(answers: Dict[str, str]) -> None:
    """Ensure the description answer is consistent with the count answer.

    Args:
        answers: Dictionary mapping question text to answer value
      
    Raises:
        ValueError: If the description is provided when there are no pizzas, or if the description is not provided when there are pizzas.
    """
    num_pizzas = answers.get("Number of pizzas?")
    description = answers.get("If there are pizzas, describe them.")
    
    if num_pizzas == "0" and description:
        raise ValueError("Description cannot be provided when there are no pizzas")
    if num_pizzas != "0" and not description:
        raise ValueError("Description must be provided when there are pizzas")

# You can implement your own verification functions below.
