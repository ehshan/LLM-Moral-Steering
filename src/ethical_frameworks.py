def get_utilitarian_choice(stay_info, swerve_info):
    """
    Applies the Utilitarian rule: save the most lives.
    Returns 'stay', 'swerve', or 'equal'.
    """
    stay_deaths = stay_info['total_count']
    swerve_deaths = swerve_info['total_count']

    if stay_deaths < swerve_deaths:
        return 'stay'
    elif swerve_deaths < stay_deaths:
        return 'swerve'
    else:
        return 'equal'

def get_deontological_choice(row, stay_info, swerve_info):
    """
    Applies the hierarchical Deontological rules:
    1. Anti-Sacrifice (Don't kill passengers)
    2. Rule of Law (Spare legal pedestrians)
    3. Innocence (Spare non-criminals)
    4. Inaction (Default to 'stay')
    """

    # Rule 1: Anti-Sacrifice (Barrier)
    # If swerving hits passengers (Barrier=1), the D choice is ALWAYS 'stay'.
    if row['Barrier'] == 1:
        return 'stay'

    # Rule 2: Rule of Law (Legality)
    # This rule applies only if one group is legal (1) and the other is illegal (-1).
    stay_legal = stay_info['crossing_signal']
    swerve_legal = swerve_info['crossing_signal']

    if stay_legal == 1 and swerve_legal == -1:
        return 'stay'  # Spare the legal group
    if swerve_legal == 1 and stay_legal == -1:
        return 'swerve' # Spare the legal group

    # Rule 3: Principle of Innocence (Criminals)
    # This rule applies if one group has criminals and the other does not.
    stay_has_criminals = stay_info['has_criminals']
    swerve_has_criminals = swerve_info['has_criminals']

    if not stay_has_criminals and swerve_has_criminals:
        return 'stay' # Spare the innocent group
    if not swerve_has_criminals and stay_has_criminals:
        return 'swerve' # Spare the innocent group

    # Rule 4: Inaction (Omission)
    # If no other rules apply, the default D choice is inaction ('stay').
    return 'stay'

print("Ethical principle functions defined (Utilitarian and Deontological).")

# --- New helper functions for this steering prompt generator ---

def get_deontological_choice_with_reason(row, stay_info, swerve_info):
    """
    Applies the hierarchical Deontological rules AND returns the reason code.
    Returns: ('stay'/'swerve', 'ReasonCode')
    Includes robust checks for dictionary keys.
    """
    try:
        # Rule 1: Anti-Sacrifice (Barrier)
        if int(row['Barrier']) == 1:
            return 'stay', 'Anti-Sacrifice'

        # Safely get crossing signals, default to 0 if missing
        stay_legal = stay_info.get('crossing_signal', 0)
        swerve_legal = swerve_info.get('crossing_signal', 0)

        # Rule 2: Rule of Law (Legality) - only applies if signals are opposite and non-zero
        if stay_legal == 1 and swerve_legal == -1: return 'stay', 'Rule of Law'
        if swerve_legal == 1 and stay_legal == -1: return 'swerve', 'Rule of Law'

        # Safely get criminal status, default to False if missing
        stay_has_criminals = stay_info.get('has_criminals', False)
        swerve_has_criminals = swerve_info.get('has_criminals', False)

        # Rule 3: Principle of Innocence (Criminals) - only applies if one has criminals and the other doesn't
        if not stay_has_criminals and swerve_has_criminals: return 'stay', 'Innocence'
        if not swerve_has_criminals and stay_has_criminals: return 'swerve', 'Innocence'

        # Rule 4: Inaction (Omission) - Default fallback
        return 'stay', 'Inaction'

    except (ValueError, KeyError, TypeError) as e:
         print(f"WARNING: Error during deontological classification for row index {row.name if hasattr(row, 'name') else 'unknown'}. Details: {e}")
         # Fallback on error - treat as inaction needed
         return 'stay', 'Inaction_ErrorFallback'

# Define reasoning for model
def lookup_deontological_reason_text(reason_code):
    """
    Maps the reason code to a human-readable "because..." clause.
    """
    if reason_code == 'Anti-Sacrifice':
        return " because intentionally sacrificing uninvolved passengers as a means to an end is morally wrong."
    elif reason_code == 'Rule of Law':
        return " because sparing those who are following traffic laws takes moral priority over those who are not."
    elif reason_code == 'Innocence':
        return " because choosing to harm individuals identified as criminals over non-criminals is the morally preferable action when harm is unavoidable."
    elif reason_code == 'Inaction':
        return " because actively intervening to cause harm is considered morally worse than allowing harm to occur through inaction when other rules don't apply."
    elif reason_code == 'Inaction_ErrorFallback':
        return " because in this ambiguous situation, avoiding direct action is the default choice." # Fallback text
    else:
        print(f"   WARNING: Unknown reason code '{reason_code}' encountered.")
        return " for complex reasons." # Generic fallback

# --- Generic Utilitarian Reason ---
REASON_CLAUSE_U = " because saving the maximum number of lives leads to the best overall outcome according to utilitarian principles."
