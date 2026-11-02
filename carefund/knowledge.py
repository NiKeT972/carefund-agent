"""Static domain knowledge: document requirements, cost benchmarks, risky phrases."""

CATEGORIES = ["organ_transplant", "cancer", "cardiac", "neonatal", "accident_trauma", "neuro", "other"]

CATEGORY_LABELS = {
    "organ_transplant": "Organ transplant",
    "cancer": "Cancer treatment",
    "cardiac": "Heart surgery",
    "neonatal": "Newborn / NICU care",
    "accident_trauma": "Accident or trauma",
    "neuro": "Brain or spine",
    "other": "Other medical",
}

# Keyword rules used for classification when no LLM is available.
CATEGORY_KEYWORDS = {
    "organ_transplant": ["transplant", "kidney failure", "liver failure", "dialysis", "cirrhosis"],
    "cancer": ["cancer", "leukemia", "leukaemia", "lymphoma", "tumour", "tumor", "chemo", "carcinoma", "bone marrow", "oncolog"],
    "cardiac": ["heart", "cardiac", "bypass", "cabg", "valve", "angioplasty", "hole in the heart"],
    "neonatal": ["nicu", "premature", "newborn", "neonatal", "preterm"],
    "accident_trauma": ["accident", "fracture", "burn", "trauma", "injury", "fell"],
    "neuro": ["brain", "spine", "spinal", "neuro", "stroke", "haemorrhage", "hemorrhage"],
}

# Illustrative upper bounds of typical treatment cost in India (INR), used only to flag outliers for review.
COST_BENCHMARKS = {
    "organ_transplant": (300_000, 4_000_000),
    "cancer": (150_000, 4_000_000),
    "cardiac": (150_000, 1_500_000),
    "neonatal": (100_000, 1_500_000),
    "accident_trauma": (100_000, 1_500_000),
    "neuro": (200_000, 1_500_000),
    "other": (50_000, 2_000_000),
}

DOCUMENT_LABELS = {
    "patient_id": "Patient ID proof (Aadhaar / PAN)",
    "organiser_id": "Organiser ID proof",
    "medical_estimate": "Cost estimate on hospital letterhead",
    "diagnosis_report": "Diagnosis report",
    "doctor_letter": "Treating doctor's letter",
    "transplant_approval": "Transplant authorisation committee approval",
    "biopsy_report": "Biopsy / histopathology report",
    "birth_certificate": "Birth certificate or hospital birth record",
    "authorisation_letter": "Patient's authorisation letter for the organiser",
}

BASE_DOCUMENTS = ["patient_id", "organiser_id", "medical_estimate", "diagnosis_report"]
CATEGORY_DOCUMENTS = {
    "organ_transplant": ["transplant_approval", "doctor_letter"],
    "cancer": ["biopsy_report", "doctor_letter"],
    "cardiac": ["doctor_letter"],
    "neonatal": ["birth_certificate"],
    "accident_trauma": ["doctor_letter"],
    "neuro": ["doctor_letter"],
    "other": ["doctor_letter"],
}

CLOSE_RELATIONS = {"self", "parent", "mother", "father", "spouse", "wife", "husband", "child", "son",
                   "daughter", "sibling", "brother", "sister", "guardian"}

PRESSURE_PHRASES = ["only today", "within hours", "will die tonight", "last chance", "act now", "urgent urgent",
                    "or else", "no time left", "guaranteed cure", "100% cure", "miracle"]
PAYMENT_PHRASES = ["upi", "gpay", "google pay", "paytm", "phonepe", "my account", "bank account",
                   "account number", "ifsc", "send money directly", "transfer to"]

STORY_BANNED = ["guaranteed", "miracle", "100%", "will surely die", "only today", "last chance", "upi", "ifsc",
                "account number"]

PLATFORM_POLICIES = [
    "CareFund pays raised funds directly to the hospital against the cost estimate, not to the organiser's personal account.",
    "Every campaign is reviewed by the CareFund verification team before it goes live; medical documents are checked with the hospital.",
    "Donors receive an 80G-style donation receipt by email when the campaign is run through a registered partner NGO; personal campaigns are not tax-deductible.",
    "If the patient no longer needs the funds, unused money is refunded to donors or, with donor consent, transferred to a similar verified patient.",
    "Organisers must post treatment updates; donors can report a campaign from the campaign page if something looks wrong.",
]
