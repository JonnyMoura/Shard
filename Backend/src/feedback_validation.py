def feedback_matches_generation(feedback_data, generation, expected_count):
    if not isinstance(expected_count, int) or expected_count <= 0:
        return False
    if not isinstance(feedback_data, dict):
        return False
    if type(feedback_data.get("generation")) is not int:
        return False
    if feedback_data["generation"] != generation:
        return False

    feedback = feedback_data.get("feedback")
    if not isinstance(feedback, list) or len(feedback) != expected_count:
        return False

    positions = set()
    for item in feedback:
        if not isinstance(item, dict):
            return False

        solution_id = item.get("solution_id")
        if not isinstance(solution_id, str):
            return False

        parts = solution_id.split("_")
        if len(parts) != 3 or parts[0] != "solution":
            return False
        try:
            solution_generation = int(parts[1])
            position = int(parts[2])
        except ValueError:
            return False

        if solution_generation != generation or position < 0 or position in positions:
            return False
        positions.add(position)

    return positions == set(range(expected_count))
