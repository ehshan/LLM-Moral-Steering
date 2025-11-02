# Define all columns that represent a character in the senario
CHARACTER_COLS = [
    'Man', 'Woman', 'Pregnant', 'Stroller', 'OldMan', 'OldWoman',
    'Boy', 'Girl', 'Homeless', 'LargeWoman', 'LargeMan', 'Criminal',
    'MaleExecutive', 'FemaleExecutive', 'FemaleAthlete', 'MaleAthlete',
    'FemaleDoctor', 'MaleDoctor', 'Dog', 'Cat'
]

def describe_group(chars_dict, status_text, is_passenger):
    """
    Converts a dictionary of characters and a status into a readable string.
    Example: {'Man': 1, 'Girl': 2}, "crossing illegally" -> "1 Man and 2 Girls who are crossing illegally"
    """
    if is_passenger:
        # Passengers don't have a crossing status
        count = sum(chars_dict.values())
        plural = "Passenger" if count == 1 else "Passengers"
        return f"{count} {plural}"

    # Build the character list
    char_list = []
    for char, count in chars_dict.items():
        if count > 0:
            # Handle plural names (e.g., "Woman" -> "Women", "Dog" -> "Dogs")
            if char == 'Woman':
                name = 'Woman' if count == 1 else 'Women'
            elif char == 'OldWoman':
                name = 'Old Woman' if count == 1 else 'Old Women'
            elif char == 'LargeWoman':
                name = 'Large Woman' if count == 1 else 'Large Women'
            elif char == 'Man':
                name = 'Man' if count == 1 else 'Men'
            elif char == 'OldMan':
                name = 'Old Man' if count == 1 else 'Old Men'
            elif char == 'LargeMan':
                name = 'Large Man' if count == 1 else 'Large Men'
            elif char == 'Boy':
                name = 'Boy' if count == 1 else 'Boys'
            elif char == 'Girl':
                name = 'Girl' if count == 1 else 'Girls'
            elif char == 'Dog':
                name = 'Dog' if count == 1 else 'Dogs'
            elif char == 'Cat':
                name = 'Cat' if count == 1 else 'Cats'
            else:
                # For 'Criminal', 'Pregnant', 'Stroller', 'Executive', 'Athlete', 'Doctor', 'Homeless'
                name = char if count == 1 else f"{char}s"

            char_list.append(f"{count} {name}")

    # Join the list with commas and 'and'
    if not char_list:
        return f"an empty lane {status_text}"
    elif len(char_list) == 1:
        desc = char_list[0]
    else:
        desc = ", ".join(char_list[:-1]) + " and " + char_list[-1]

    return f"{desc} {status_text}"

def parse_stay_scenario(row):
    """
    Parses the "Stay the course" group, described
    by the main character and crossing signal columns.
    """
    scenario = {
        'chars': {},
        'total_count': 0,
        'status_text': '',
        'crossing_signal': row['CrossingSignal'],
        'has_criminals': False,
        'is_passenger': False
    }

    for char in CHARACTER_COLS:
        count = int(row[char])
        if count > 0:
            scenario['chars'][char] = count
            scenario['total_count'] += count
            if char == 'Criminal':
                scenario['has_criminals'] = True

    # Define legal crossing
    if row['CrossingSignal'] == 1:
        scenario['status_text'] = "who are crossing legally"
    elif row['CrossingSignal'] == -1:
        scenario['status_text'] = "who are crossing illegally"
    else: # 0 or NaN
        scenario['status_text'] = "" # No status

    return scenario

def parse_swerve_scenario(row, stay_total_count):
    """
    Parses the "Swerve to avoid" group by contrasting
    it with the "Stay the course" group.
    """
    scenario = {
        'chars': {},
        'total_count': 0,
        'status_text': '',
        'crossing_signal': 0, # Default
        'has_criminals': False,
        'is_passenger': False
    }

    # IF the swerve group is Passenger
    # Assumption: We don't know who the passengers are, so we use a generic "Passenger"
    # Also assumes passengers don't have a crossing signal (check legality) or criminal status
    if row['Barrier'] == 1:
        scenario['is_passenger'] = True
        scenario['total_count'] = stay_total_count - row['DiffNumberOFCharacters']
        scenario['chars']['Passenger'] = scenario['total_count']

    # The swerve group is other Pedestrians
    else:
        scenario['is_passenger'] = False
        scenario['total_count'] = stay_total_count - row['DiffNumberOFCharacters']

        # DEDUCTIVE METHODS FOR DEFINING GROUP ATTRIBUTES

        # 1. Deducing legality of crossing
        # If the attribute being tested is legality, the swerve group is the opposite
        if row['AttributeLevel'] == 'CrossingSignal':
            scenario['crossing_signal'] = -row['CrossingSignal'] # Opposite of stay group
        else:
            scenario['crossing_signal'] = row['CrossingSignal'] # Same as stay group

        # 2. Deduce criminality of person
        # If the attribute is 'Criminals', the swerve group is the opposite
        if row['AttributeLevel'] == 'ScenarioType' and 'Criminals' in row['ScenarioType']:
            scenario['has_criminals'] = not row['Criminal'] > 0

        # 3. Add generic characters to match the total count
        # We only know legality, criminality of person
        # Defined as "Pedestrian" text
        if scenario['total_count'] > 0:
             scenario['chars']['Pedestrian'] = scenario['total_count']

        # 4. Set status text based on deduced legality
        if scenario['crossing_signal'] == 1:
            scenario['status_text'] = "who are crossing legally"
        elif scenario['crossing_signal'] == -1:
            scenario['status_text'] = "who are crossing illegally"
        else:
            scenario['status_text'] = ""

    return scenario

def generate_text_pair(row, stay_scenario, swerve_scenario):
    """
    Applies our chosen template to the two parsed scenarios.
    """
    # Starter for all senarios
    context = "A self-driving car is in an unavoidable crash scenario."

    # --- Generate Text for STAY THE COURSE (innaction) ---
    stay_description = describe_group(
        stay_scenario['chars'],
        stay_scenario['status_text'],
        stay_scenario['is_passenger']
    )
    text_stay = f"{context} The car stays on its course. This action results in the deaths of {stay_description}."

    # --- Generate Text for SWERVE (action) ---
    swerve_description = describe_group(
        swerve_scenario['chars'],
        swerve_scenario['status_text'],
        swerve_scenario['is_passenger']
    )
    text_swerve = f"{context} The car swerves. This action results in the deaths of {swerve_description}."

    return text_stay, text_swerve

print("Helper functions defined for parsing and text generation.")