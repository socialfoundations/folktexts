"""Module to hold ACS column mappings from values to natural text."""

import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from folktexts.acs.acs_columns import acs_place_of_birth

from ._utils import parse_pums_code

# Path to ACS codebook files
ACS_CODEBOOK_DIR = Path(__file__).parent / "data"


def transform_age(x, bin_width=10, min_age=18, max_age=90):
    # Create age bins including min and max
    # Find the first bin edge >= min_age that is a multiple of bin_width
    first_bin_edge = ((min_age + bin_width - 1) // bin_width) * bin_width
    # skip if too close to min age
    if first_bin_edge - min_age > bin_width / 2:
        age_bins = [min_age, first_bin_edge]
    else:
        age_bins = [min_age]
    age_bins += list(np.arange(first_bin_edge + bin_width, max_age, bin_width))
    age_bins.append(max_age)

    if x < min_age:
        return f"Less than {min_age} years old"
    elif x >= max_age:
        return f"{max_age} or more years old"

    # Find the index of the bin this age falls into
    idx = np.searchsorted(age_bins, x, side="right") - 1
    lower = age_bins[idx]
    upper = age_bins[idx + 1] - 1
    return f"{lower}-{upper} years old"


def transform_cow(x):
    map_to_lower_res = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 8, 9: 9}
    simplified_cow_map = {
        1: "Employed",
        2: "Self-employed",
        8: "Unpaid worker",
        9: "Unemployed or not in the labor force",
    }
    return simplified_cow_map.get(map_to_lower_res.get(x))


def transform_schooling(x):
    simplified_schl_map = {
        1: "No formal education",
        2: "Early childhood education",
        3: "Primary education (Grades 1-5)",
        4: "Middle school education (Grades 6-8)",
        5: "Some high school, no diploma",
        6: "High school graduate or equivalent",
        7: "Some college, no degree",
        8: "Associate's degree",
        9: "Bachelor's degree",
        10: "Graduate or professional degree",
    }
    map_to_lower_res = {
        1: 1,  # No schooling -> No formal education
        2: 2,  # Nursery school -> Early childhood education
        3: 2,  # Kindergarten -> Early childhood education
        4: 3,  # 1st grade -> Primary education
        5: 3,  # 2nd grade -> Primary education
        6: 3,  # 3rd grade -> Primary education
        7: 3,  # 4th grade -> Primary education
        8: 3,  # 5th grade -> Primary education
        9: 4,  # 6th grade -> Middle school education
        10: 4,  # 7th grade -> Middle school education
        11: 4,  # 8th grade -> Middle school education
        12: 5,  # 9th grade -> Some high school, no diploma
        13: 5,  # 10th grade -> Some high school, no diploma
        14: 5,  # 11th grade -> Some high school, no diploma
        15: 5,  # 12th grade, no diploma -> Some high school, no diploma
        16: 6,  # High school diploma -> High school graduate or equivalent
        17: 6,  # GED -> High school graduate or equivalent
        18: 7,  # Some college, less than 1 year -> Some college, no degree
        19: 7,  # Some college, 1+ years -> Some college, no degree
        20: 8,  # Associate’s degree -> Associate’s degree
        21: 9,  # Bachelor's degree -> Bachelor's degree
        22: 10,  # Master's degree -> Graduate or professional degree
        23: 10,  # Professional degree -> Graduate or professional degree
        24: 10,  # Doctorate degree -> Graduate or professional degree
    }
    return simplified_schl_map.get(map_to_lower_res.get(x))


def transform_occp(x):
    simplified_occp_map = {
        "MGR": "Management Occupations",
        "BUS": "Business and Financial Operations",
        "FIN": "Finance and Accounting",
        "CMM": "Computer and Mathematical Occupations",
        "ENG": "Engineering and Architecture",
        "SCI": "Science and Research",
        "CMS": "Community and Social Services",
        "LGL": "Legal Occupations",
        "EDU": "Education and Training",
        "ENT": "Arts, Design, Entertainment, Sports, and Media",
        "MED": "Healthcare Practitioners and Technicians",
        "HLS": "Healthcare Support",
        "PRT": "Protective Services",
        "EAT": "Food Preparation and Serving",
        "CLN": "Building and Grounds Cleaning and Maintenance",
        "PRS": "Personal Care and Service",
        "SAL": "Sales and Related Occupations",
        "OFF": "Office and Administrative Support",
        "FFF": "Farming, Fishing, and Forestry",
        "CON": "Construction",
        "EXT": "Mining and Extraction",
        "RPR": "Installation, Maintenance, and Repair",
        "PRD": "Production and Manufacturing",
        "TRN": "Transportation and Material Moving",
        "MIL": "Military Occupations",
        "UNEMPL": "Unemployed, Not worked for at least 5 years or Never Worked",
    }

    def get_prefix(occp):
        if len(occp.split("-", 1)) == 2:
            cat, desc = occp.split("-", 1)
            return cat
        else:
            # catch cases in OCCP.txt not following the format
            if occp.startswith("Engineering"):
                return "ENG"
            elif occp.startswith("Grinding"):
                return "PRD"
            elif occp.startswith("Unemployed"):
                return "UNEMPL"
            else:
                return occp

    map_to_lower_res = partial(
        parse_pums_code,
        file=ACS_CODEBOOK_DIR / "OCCP.txt",
        postprocess=get_prefix,
    )
    return simplified_occp_map.get(map_to_lower_res(x), "N/A")


def transform_pobp(x):
    simplified_pobp_map = {
        1: "Northeast USA",
        2: "Midwest USA",
        3: "South USA",
        4: "West USA",
        5: "US Territories",
        6: "European Union",
        7: "Europe, non EU",
        8: "Asia",
        9: "North America (excluding USA)",
        10: "Central America & Caribbean",
        11: "South America",
        12: "Africa",
        13: "Oceania",
        14: "Other/Unspecified",
    }

    map_to_lower_res = {
        # Northeast USA
        **{i: 1 for i in [9, 23, 25, 33, 34, 36, 42, 44, 50]},
        # CT, ME, MA, NH, NJ, NY, PA, RI, VT
        # Midwest USA
        **{i: 2 for i in [17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55]},
        # IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, WI
        # South USA
        **{i: 3 for i in [1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54]},
        # AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN, TX, VA, WV
        # West USA
        **{i: 4 for i in [2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56]},
        # AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY
        # US Territories
        60: 5,
        66: 5,
        69: 5,
        72: 5,
        78: 5,  # American Samoa, Guam, N. Mariana Islands, Puerto Rico, US Virgin Islands
        # European Union (EU)
        **{
            i: 6
            for i in [
                102,
                103,
                104,
                151,
                208,
                148,
                106,
                108,
                109,
                110,
                116,
                117,
                119,
                120,
                156,
                157,
                126,
                128,
                129,
                130,
                132,
                149,
                134,
                136,
            ]
        },
        # Austria, Belgium, Bulgaria, Croatia, Cyprus (2016 or earlier), Czech Republic, Denmark,
        # Finland, France, Germany, Greece, Hungary, Ireland, Italy, Latvia, Lithuania,
        # Netherlands, Poland, Portugal, Azores Islands, Romania, Slovakia, Spain, Sweden
        # missing: Estonia, Luxembourg, Slovenia missing
        # Non-EU Europe or unclear
        **{
            i: 7
            for i in [
                100,
                105,
                118,
                127,
                137,
                138,
                139,
                140,
                142,
                147,
                150,
                152,
                154,
                158,
                159,
                160,
                161,
                162,
                163,
                164,
                165,
                166,
                167,
                168,
                169,
            ]
        },
        # Albania, Czechoslovakia, Iceland, Norway, Switzerland, United Kingdom, Not Specified, England,
        # Scotland, Northern Ireland (2017 or later), Yugoslavia, Bosnia and Herzegovina, Macedonia,
        # Serbia, Armenia, Azerbaijan, Belarus, Georgia, Moldova, Russia, Ukraine, USSR,
        # Europe (2017 or later), Kosovo (2017 or later), Montenegro, Other Europe, Not Specified
        # Asia
        **{i: 8 for i in range(200, 250) if i != 208},  # Asian countries (200-249)
        253: 8,
        254: 8,
        # North America (excluding USA)
        300: 9,
        301: 9,
        303: 9,  # Bermuda, Canada, Mexico
        # Central America & Caribbean
        **{i: 10 for i in range(310, 345)},  # Central America & Caribbean (310-344)
        # South America
        **{i: 11 for i in range(360, 375)},  # South America (360-374)
        # Africa
        **{i: 12 for i in range(400, 470)},  # Africa (400-469)
        462: 14,
        # Oceania
        **{i: 13 for i in range(501, 528)},  # Australia, New Zealand, Pacific Islands
        # Other / Unspecified
        399: 14,
        554: 14,  # Misc. unspecified regions
    }

    return simplified_pobp_map.get(map_to_lower_res.get(x, 14))


def transform_pobp_unsd(x):
    original_value_map = acs_place_of_birth.value_map
    unsd_data = pd.read_csv(ACS_CODEBOOK_DIR / "UNSD.csv", sep=";")

    manually_matched_area_names = {
        "Commonwealth of the Northern Mariana Islands": "Northern Mariana Islands",
        "US Virgin Islands": "United States Virgin Islands",
        "Azores Islands": "Portugal",
        "United Kingdom, Not Specified": "United Kingdom of Great Britain and Northern Ireland",
        "England": "United Kingdom of Great Britain and Northern Ireland",
        "Scotland": "United Kingdom of Great Britain and Northern Ireland",
        "Czech Republic": "Czechia",
        "Laos": "Lao People's Democratic Republic",
        "Turkey": "Türkiye",
        "Vietnam": "Viet Nam",
        "St. Kitts-Nevis": "Saint Kitts and Nevis",
        "St. Lucia": "Saint Lucia",
        "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
        "Ivory Coast": "Côte d’Ivoire",
        "Democratic Republic of Congo": "Democratic Republic of the Congo",
    }
    manually_matched_name_to_subregion = {
        "Kosovo": "Southern Europe",
        "Yugoslavia": "Southern Europe",
        "Taiwan": "Eastern Asia",
        "West Indies": "Latin America and the Caribbean",
        "USSR": "Eastern Europe",
        "Czechoslovakia": "Eastern Europe",
        # Region Names
        "Europe": "Europe, not specified",
        "Other Europe, Not Specified": "Europe, not specified",
        "Caribbean, Not Specified": "Latin America and the Caribbean",
        "Northern Africa, Not Specified": "Northern Africa",
        "Western Africa, Not Specified": "Sub-Saharan Africa",
        "Eastern Africa, Not Specified": "Sub-Saharan Africa",
        "Other Africa, Not Specified": "Africa, not specified",
        "Asia": "Asia, not specified",
        "South Central Asia, Not Specified": "Asia, not specified",
        "Other Asia, Not Specified": "Asia, not specified",
        "South America": "Latin America and the Caribbean",
        "Other US Island Areas, Oceania, Not Specified, or at Sea": "Not specified",
        "Americas, Not Specified": "Not specified",
    }

    if x in range(1, 57):  # US states
        return "United States of America"
    else:
        name = original_value_map(x)
        if "(" in name:
            name = name[: name.find("(")].strip()
        if name in manually_matched_area_names.keys():
            name = manually_matched_area_names[name]
        for idx, (region, area) in unsd_data[["Sub-region Name", "Country or Area"]].iterrows():
            if name == area:
                return region
            if name in area:
                return region
        if name in manually_matched_name_to_subregion.keys():
            return manually_matched_name_to_subregion[name]
        else:
            logging.warning(f"Could not find code '{x}' or name '{name}' in file '{ACS_CODEBOOK_DIR / 'UNSD.csv'}'")
            return "N/A"


def transform_relp(x):
    simplified_relp_map = {
        0: "Reference person",
        1: "Spouse or partner",
        2: "Child (biological, adpoted or stepchild)",
        3: "Siblings",
        4: "Parent",
        5: "Grandchild",
        6: "Parent-in-law",
        7: "Son-in-law or daughter-in-law",
        8: "Other relative",
        9: "Non-relative (e.g., roommate, boarder)",
        10: "Group quarters population",
    }
    map_to_lower_res = {
        0: 0,  # "The reference person itself" -> "Reference person"
        1: 1,  # "Husband/wife" -> "Spouse/Partner"
        2: 2,  # "Biological son or daughter" -> "Children"
        3: 2,  # "Adopted son or daughter" -> "Children"
        4: 2,  # "Stepson or stepdaughter" -> "Children"
        5: 3,  # "Brother or sister" -> "Siblings"
        6: 4,  # "Father or mother" -> "Parents"
        7: 5,  # "Grandchild" -> "Grandchild"
        8: 6,  # "Parent-in-law" -> "Parent-in-law"
        9: 7,  # "Son-in-law or daughter-in-law" -> "Son-in-law or daughter-in-law"
        10: 8,  # "Other relative" -> "Other relatives"
        11: 9,  # "Roomer or boarder" -> "Non-relatives (e.g., roommate, boarder)"
        12: 9,  # "Housemate or roommate" -> "Non-relatives (e.g., roommate, boarder)"
        13: 1,  # "Unmarried partner" -> "Spouse/Partner"
        14: 2,  # "Foster child" -> "Children"
        15: 9,  # "Other non-relative" -> "Non-relatives (e.g., roommate, boarder)"
        16: 10,  # "Institutionalized group quarters population" -> "Group quarters population"
        17: 10,  # "Non-institutionalized group quarters population" -> "Group quarters population"
    }
    return simplified_relp_map.get(map_to_lower_res.get(x))


def transform_wkhp(x, bin_width=10, max_hours=60):
    # Generate bin edges incl max_hours
    bins = np.arange(0, max_hours + bin_width, bin_width)

    # Handle non-working or underage responses
    if x <= 0:
        return "N/A (less than 16 years old, or did not work during the past 12 months)"
    elif x >= bins[-1]:
        return f"more than {bins[-1]} hours"

    # Find appropriate bin
    idx = np.searchsorted(bins, x, side="right") - 1
    lower = bins[idx]
    upper = bins[idx + 1] - 1
    return f"{lower}-{upper} hours"


def transform_rac1p_binary(x):
    # binarize
    binary_value_map = {1: "White", 2: "Non-White"}

    def map_to_binary(x):
        return 1 if x == 1 else 2

    return binary_value_map.get(map_to_binary(x))


def transform_rac1p(x):
    simplified_rac1p_map = {
        1: "White",
        2: "Black or African American",
        3: "Indigenous (American Indian or Alaska Native)",
        4: "Asian",
        5: "Pacific Islander",
        6: "Some other race alone",
        7: "Two or more races",
    }
    map_to_lower_res = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7}
    return simplified_rac1p_map.get(map_to_lower_res.get(x))


simplified_value_maps = {
    "AGEP": transform_age,
    "COW": transform_cow,
    "SCHL": transform_schooling,
    "OCCP": transform_occp,
    "POBP": transform_pobp,
    "RELP": transform_relp,
    "WKHP": transform_wkhp,
    "RAC1P": transform_rac1p,
}
