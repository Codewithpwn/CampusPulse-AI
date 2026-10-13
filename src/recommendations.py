
"""Generate transparent, rule-based actions for detected complaint themes."""

RECOMMENDATIONS = {
    "Lab equipment and practical learning": (
        "Review laboratory equipment, practical-session quality, and whether "
        "activities use relevant, up-to-date technologies."
    ),
    "Library books and availability": (
        "Review book availability against student demand and identify frequently "
        "requested books that may need additional copies."
    ),
    "Library systems and rules": (
        "Investigate book-issuing system reliability and review whether library "
        "rules create avoidable difficulties for students."
    ),
    "Course material and content": (
        "Ask course coordinators to review learning materials, syllabus relevance, "
        "and opportunities for conceptual rather than rote learning."
    ),
    "Assessment and grading fairness": (
        "Review assessment rubrics, communicate grading criteria, and provide "
        "a clear process for students to query assessment decisions."
    ),
    "Teaching quality and subject knowledge": (
        "Review student feedback on lecture delivery and consider appropriate "
        "support or professional development for teaching staff."
    ),
    "Teacher interaction and student support": (
        "Review opportunities for students to ask questions, raise concerns, "
        "and receive timely responses from teaching staff."
    ),
    "General dissatisfaction": (
        "Review the original comments to determine the specific concern before "
        "deciding on an improvement action."
    ),
    "Other / Review": (
        "Manually review these comments and assign a theme before recommending "
        "a specific action."
    ),
}


def get_recommendation(theme):
    """Return a suggested action for a theme, or a safe review fallback."""
    return RECOMMENDATIONS.get(
        theme,
        "Review the original feedback and determine an appropriate action."
    )
