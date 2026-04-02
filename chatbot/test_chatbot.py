from chatbot.chatbot_nlp import get_chatbot_response

def run_test_case(title, messages, user_id="benz"):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for msg in messages:
        result = get_chatbot_response(msg, user_id=user_id)
        print(f"User: {msg}")
        print("Bot:", result)
        print("-" * 80)


if __name__ == "__main__":
    # 1. Greeting
    run_test_case(
        "TEST 1 - GREETING",
        ["hello"],
        user_id="user_greeting"
    )

    # 2. Thanks
    run_test_case(
        "TEST 2 - THANKS",
        ["thanks"],
        user_id="user_thanks"
    )

    # 3. Required inputs
    run_test_case(
        "TEST 3 - REQUIRED INPUTS",
        ["what inputs are required"],
        user_id="user_required"
    )

    # 4. How to predict
    run_test_case(
        "TEST 4 - HOW TO PREDICT",
        ["how can I predict a lead"],
        user_id="user_how_predict"
    )

    # 5. Model used
    run_test_case(
        "TEST 5 - MODEL USED",
        ["which model is used"],
        user_id="user_model"
    )

    # 6. Lead priority
    run_test_case(
        "TEST 6 - LEAD PRIORITY",
        ["which leads should I prioritize"],
        user_id="user_priority"
    )

    # 7. Follow-up strategy
    run_test_case(
        "TEST 7 - FOLLOW-UP STRATEGY",
        ["what follow up strategy should I use"],
        user_id="user_followup"
    )

    # 8. Score interpretation
    run_test_case(
        "TEST 8 - SCORE INTERPRETATION",
        ["what does the score mean"],
        user_id="user_score"
    )

    # 9. Model trust
    run_test_case(
        "TEST 9 - MODEL TRUST",
        ["can we trust this model"],
        user_id="user_trust"
    )

    # 10. Database info
    run_test_case(
        "TEST 10 - DATABASE INFO",
        ["is chatbot data stored in database"],
        user_id="user_db"
    )

    # 11. Dashboard help
    run_test_case(
        "TEST 11 - DASHBOARD HELP",
        ["what does dashboard show"],
        user_id="user_dashboard"
    )

    # 12. Source performance
    run_test_case(
        "TEST 12 - SOURCE PERFORMANCE",
        ["which lead source performs best"],
        user_id="user_source"
    )

    # 13. System usage
    run_test_case(
        "TEST 13 - SYSTEM USAGE",
        ["how can I use this system"],
        user_id="user_usage"
    )

    # 14. Goodbye
    run_test_case(
        "TEST 14 - GOODBYE",
        ["bye"],
        user_id="user_goodbye"
    )

    # 15. Fallback / unknown query
    run_test_case(
        "TEST 15 - FALLBACK",
        ["banana laptop river sky"],
        user_id="user_fallback"
    )

    # 16. Step-by-step prediction flow
    run_test_case(
        "TEST 16 - STEP-BY-STEP PREDICTION",
        [
            "predict this lead",
            "Sales Person 1",
            "Data Science",
            "Ernakulam",
            "Digital Marketing",
            "MBA",
            "Data Science",
            "Male"
        ],
        user_id="user_prediction"
    )

    # 17. Prediction cancel flow
    run_test_case(
        "TEST 17 - PREDICTION CANCEL",
        [
            "predict this lead",
            "Sales Person 2",
            "cancel"
        ],
        user_id="user_cancel"
    )

    # 18. Restart prediction after cancel
    run_test_case(
        "TEST 18 - RESTART AFTER CANCEL",
        [
            "predict this lead",
            "Sales Person 3",
            "stop",
            "predict this lead",
            "Sales Person 3",
            "MBA",
            "Kozhikode",
            "Website Enquiry",
            "MBA",
            "Finance",
            "Female"
        ],
        user_id="user_restart"
    )