"""Threshold instances for ACS / folktables tasks.
"""
from folktexts.threshold import Threshold

# ACSIncome task
acs_income_threshold = Threshold(50_000, ">")

# ACSPublicCoverage task
acs_public_coverage_threshold = Threshold(1, "==")

# ACSMobility task
acs_mobility_threshold = Threshold(1, "==")

# ACSEmployment task
acs_employment_threshold = Threshold(1, "==")

# ACSTravelTime task
acs_travel_time_threshold = Threshold(20, ">")

# ACSIncomePovertyRatio task
acs_income_poverty_ratio_threshold = Threshold(250, "<")

# ACSHealthInsurance task
acs_health_insurance_threshold = Threshold(1, "==")
