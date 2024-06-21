"""Threshold instances for ACS / folktables tasks.
"""
from folktexts.threshold import Threshold


# ACSIncome task
acs_income_threshold = Threshold(50_000, ">")

# ACSPublicCoverage task
acs_publiccoverage_threshold = Threshold(1, "==")

# ACSMobility task
acs_mobility_threshold = Threshold(1, "==")

# ACSEmployment task
acs_employment_threshold = Threshold(1, "==")

# ACSTravelTime task
acs_traveltime_threshold = Threshold(20, ">")
