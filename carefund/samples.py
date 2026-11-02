"""Demo cases (fictional people and documents)."""

CLEAN = {
    "form": {
        "organiser_name": "Rohan Patil",
        "relation": "mother",
        "patient_name": "Sunita Patil",
        "patient_age": 54,
        "city": "Pune",
        "hospital_name": "Sahyadri Care Hospital",
        "doctor_name": "Dr. Meera Kulkarni",
        "condition": "End-stage kidney failure",
        "treatment": "Kidney transplant",
        "estimated_cost": 750000,
        "amount_arranged": 250000,
        "goal_amount": 500000,
        "description": "My mother Sunita has been on dialysis three times a week for the last 14 months. Doctors at "
                       "Sahyadri Care Hospital have advised a kidney transplant within 45 days, and my father is a "
                       "matching donor. My father retired last year and I work as a sales executive, so the family "
                       "has put together 2.5 lakh from savings and insurance. We need help with the remaining amount.",
    },
    "documents": ["patient_id", "organiser_id", "medical_estimate", "diagnosis_report", "transplant_approval", "doctor_letter"],
    "estimate_text": "SAHYADRI CARE HOSPITAL, Pune. Cost estimate for patient Mrs. Sunita Patil (54 F). Procedure: live "
                     "related renal transplant. Estimated package: Rs. 7,50,000 including surgery, ICU stay and "
                     "first-month immunosuppressants. Dr. Meera Kulkarni, Nephrology.",
}

RISKY = {
    "form": {
        "organiser_name": "Vikram S",
        "relation": "friend",
        "patient_name": "Arjun Mehta",
        "patient_age": 29,
        "city": "Mumbai",
        "hospital_name": "Lifeline Heart Institute",
        "doctor_name": "",
        "condition": "Heart valve disease",
        "treatment": "Valve replacement surgery",
        "estimated_cost": 4500000,
        "amount_arranged": 0,
        "goal_amount": 6000000,
        "description": "URGENT URGENT. My friend Arjun needs heart surgery costing 60 lakh, only today we can save him, "
                       "last chance! Please also send money directly to my UPI vikram@upi so it reaches faster.",
    },
    "documents": ["patient_id"],
    "estimate_text": "City Care Clinic. Estimate for valve replacement: Rs 3,20,000.",
}

SAMPLES = {"clean": CLEAN, "risky": RISKY}
