import folktexts
import pandas as pd

from folktexts.col_to_text import ColumnToText
from folktexts.task import TaskMetadata
from folktexts.qa_interface import DirectNumericQA
from folktexts.qa_interface import MultipleChoiceQA, Choice
from folktexts.classifier import WebAPILLMClassifier
from folktexts.benchmark import BenchmarkConfig, Benchmark
from folktexts.dataset import Dataset
import os

# os.chdir('/users/bryanwilder/Dropbox/llm_preds')


TASK_DESCRIPTION = """\
The following data corresponds to US Census data on US residents. \
The data is from states throughout the US in the year 2018. \
Please answer each question based on the information provided. \
The data provided is enough to reach an approximate answer for each person.
"""

occp_dict={
        160: 'MGR-Transportation, Storage, And Distribution Managers', 
        820: 'FIN-Budget Analysts', 
        1105: 'CMM-Network And Computer Systems Administrators', 
        2300: 'EDU-Preschool And Kindergarten Teachers', 
        3220: 'MED-Respiratory Therapists', 
        5340: 'OFF-New Accounts Clerks', 
        5730: 'OFF-Medical Secretaries and Administrative Assistants', 
        5840: 'OFF-Insurance Claims And Policy Processing Clerks', 
        8610: 'PRD-Stationary Engineers And Boiler Operators', 
        7120: 'RPR-Audiovisual Equipment Installers And Repairers', 
        640: 'BUS-Compensation, Benefits, And Job Analysis Specialists', 
        3545: 'MED-Miscellaneous Health Technologists and Technicians', 
        3647: 'HLS-Pharmacy Aides', 3945: 'PRT-Transportation Security Screeners', 
        4110: 'EAT-Waiters And Waitresses', 
        5630: 'OFF-Weighers, Measurers, Checkers, And Samplers, Recordkeeping', 
        8710: 'PRD-Cutting Workers', 
        3910: 'PRT-Private Detectives And Investigators', 
        4130: 'EAT-Dining Room And Cafeteria Attendants And Bartender Helpers', 
        4540: 'PRS-Tour And Travel Guides', 
        5330: 'OFF-Loan Interviewers And Clerks', 
        7730: 'PRD-Engine And Other Machine Assemblers', 
        7925: 'PRD-Forming Machine Setters, Operators, And Tenders, Metal And Plastic', 
        8250: 'PRD-Prepress Technicians And Workers', 9040: 'TRN-Air Traffic Controllers And Airfield Operations Specialists', 
        9570: 'TRN-Conveyor, Dredge, And Hoist and Winch Operators', 
        9: 'N/A (less than 16 years old/NILF who last worked more than 5 years ago or never worked)', 
        705: 'BUS-Project Management Specialists', 
        1010: 'CMM-Computer Programmers', 
        1305: 'ENG-Architects, Except Landscape And Naval', 
        1450: 'ENG-Materials Engineers', 
        1541: 'ENG-Architectural And Civil Drafters', 
        3324: 'MED-Magnetic Resonance Imaging Technologists', 
        3430: 'MED-Dietetic Technicians And Ophthalmic Medical Technicians', 
        4340: 'PRS-Animal Trainers', 
        6530: 'CON-Structural Iron And Steel Workers', 
        7140: 'RPR-Aircraft Mechanics And Service Technicians', 
        7750: 'PRD-Other Assemblers And Fabricators', 
        425: 'MGR-Emergency Management Directors', 
        500: 'BUS-Agents And Business Managers Of Artists, Performers, And Athletes', 
        1720: 'SCI-Chemists And Materials Scientists', 3603: 'HLS-Nursing Assistants', 
        3802: 'PRT-Correctional Officers and Jailers', 
        5420: 'OFF-Other Information And Records Clerks', 8025: 'PRD-Other Machine Tool Setters, Operators, And Tenders, Metal and Plastic', 
        8600: 'PRD-Power Plant Operators, Distributors, And Dispatchers', 2003: 'CMS-Marriage And Family Therapists', 2310: 'EDU-Elementary And Middle School Teachers', 
        2865: 'ENT-Media And Communication Workers, All Other', 3421: 'MED-Pharmacy Technicians', 4940: 'SAL-Telemarketers', 8810: 'PRD-Painting Workers', 
        9630: 'TRN-Machine Feeders And Offbearers', 8950: 'PRD-Helpers--Production Workers', 2636: 'ENT-Merchandise Displayers And Window Trimmers', 
        1410: 'ENG-Electrical And Electronics Engineers', 420: 'MGR-Social And Community Service Managers', 1800: 'SCI-Economists', 2060: 'CMS-Religious Workers, All Other', 
        2640: 'ENT-Other Designers', 4465: 'PRS-Morticians, Undertakers, And Funeral Arrangers', 4800: 'SAL-Advertising Sales Agents', 5250: 'OFF-Eligibility Interviewers, Government Programs', 
        7130: 'RPR-Security And Fire Alarm Systems Installers', 8750: 'PRD-Jewelers And Precious Stone And Metal Workers', 8930: 'PRD-Paper Goods Machine Setters, Operators, And Tenders', 
        1032: 'CMM-Web And Digital Interface Designers', 2320: 'EDU-Secondary School Teachers', 2905: 'ENT-Other Media And Communication Equipment Workers', 
        3620: 'HLS-Physical Therapist Assistants And Aides', 3646: 'HLS-Medical Transcriptionists', 3870: 'PRT-Police Officers', 4020: 'EAT-Cooks', 4240: 'CLN-Pest Control Workers', 
        5020: 'OFF-Telephone Operators', 5140: 'OFF-Payroll And Timekeeping Clerks', 5150: 'OFF-Procurement Clerks', 5610: 'OFF-Shipping, Receiving, And Inventory Clerks', 
        8225: 'PRD-Other Metal Workers And Plastic Workers', 2810: 'ENT-News Analysts, Reporters, And Journalists', 3720: 'PRT-First-Line Supervisors Of Firefighting And Prevention Workers', 
        10: 'MGR-Chief Executives And Legislators', 2011: 'CMS-Child, Family, And School Social Workers', 3200: 'MED-Radiation Therapists', 3640: 'HLS-Dental Assistants', 
        3655: 'HLS-Other Healthcare Support Workers', 4200: 'CLN-First-Line Supervisors Of Housekeeping And Janitorial Workers', 4522: 'PRS-Skincare Specialists', 4622: 'PRS-Recreation Workers', 
        8140: 'PRD-Welding, Soldering, And Brazing Workers', 3725: 'PRT-Miscellaneous First-Line Supervisors, Protective Service Workers', 335: 'MGR-Entertainment and Recreation Managers', 
        2633: 'ENT-Floral Designers', 3750: 'PRT-Fire Inspectors', 5860: 'OFF-Office Clerks, General', 6240: 'CON-Carpet, Floor, And Tile Installers And Finishers', 
        7020: 'RPR-Radio And Telecommunications Equipment Installers And Repairers', 7260: 'RPR-Miscellaneous Vehicle And Mobile Equipment Mechanics, Installers, And Repairers', 
        9350: 'TRN-Parking Attendants', 102: 'MGR-Facilities Managers', 630: 'BUS-Human Resources Workers', 1022: 'CMM-Software Quality Assurance Analysts and Testers', 
        2180: 'LGL-Legal Support Workers, All Other', 2755: 'ENT-Disc Jockeys, Except Radio', 2850: 'ENT-Writers And Authors', 3150: 'MED-Occupational Therapists', 3630: 'HLS-Massage Therapists', 
        4055: 'EAT-Fast Food And Counter Workers', 4160: 'EAT-Food Preparation and Serving Related Workers, All Other', 4720: 'SAL-Cashiers', 6010: 'FFF-Agricultural Inspectors', 
        6825: 'EXT-Surface Mining Machine Operators And Earth Drillers', 960: 'FIN-Other Financial Specialists', 1860: 'SCI-Other Social Scientists', 
        1900: 'SCI-Agricultural And Food Science Technicians', 4120: 'EAT-Food Servers, Nonrestaurant', 7200: 'RPR-Automotive Service Technicians And Mechanics', 
        7350: 'RPR-Maintenance Workers, Machinery', 1007: 'CMM-Information Security Analysts', 2006: 'CMS-Counselors, All Other', 2721: 'ENT-Athletes and Sports Competitors', 
        3840: 'PRT-Fish And Game Wardens And Parking Enforcement Officers', 3946: 'PRT-School Bus Monitors', 8030: 'PRD-Machinists', 9415: 'TRN-Passenger Attendants', 4600: 'PRS-Childcare Workers', 
        2015: 'CMS-Probation Officers And Correctional Treatment Specialists', 5040: 'OFF-Communications Equipment Operators, All Other', 5710: 'OFF-Executive Secretaries And Executive Administrative Assistants', 7360: 'RPR-Millwrights', 2555: 'EDU-Other Educational Instruction And Library Workers', 340: 'MGR-Lodging Managers', 3649: 'HLS-Phlebotomists', 4252: 'CLN-Tree Trimmers and Pruners', 5400: 'OFF-Receptionists And Information Clerks', 6740: 'CON-Rail-Track Laying And Maintenance Equipment Operators', 8310: 'PRD-Pressers, Textile, Garment, And Related Materials', 1005: 'CMM-Computer And Information Research Scientists', 2910: 'ENT-Photographers', 6355: 'CON-Electricians', 6850: 'EXT-Underground Mining Machine Operators', 8555: 'PRD-Other Woodworkers', 510: 'BUS-Buyers And Purchasing Agents, Farm Products', 1420: 'ENG-Environmental Engineers', 3160: 'MED-Physical Therapists', 4210: 'CLN-First-Line Supervisors Of Landscaping, Lawn Service, And Groundskeeping Workers', 5500: 'OFF-Cargo And Freight Agents', 7150: 'RPR-Automotive Body And Related Repairers', 9122: 'TRN-Bus Drivers, Transit And Intercity', 9141: 'TRN-Shuttle Drivers And Chauffeurs', 6220: 'CON-Brickmasons, Blockmasons, Stonemasons, And Reinforcing Iron And Rebar Workers', 2330: 'EDU-Special Education Teachers', 4010: 'EAT-First-Line Supervisors Of Food Preparation And Serving Workers', 4965: 'SAL-Sales And Related Workers, All Other', 6600: 'CON-Helpers, Construction Trades', 110: 'MGR-Computer And Information Systems Managers', 2631: 'ENT-Commercial And Industrial Designers', 2861: 'ENT-Interpreters and Translators', 4810: 'SAL-Insurance Sales Agents', 5300: 'OFF-Hotel, Motel, And Resort Desk Clerks', 9265: 'TRN-Other Rail Transportation Workers', 9365: 'TRN-Transportation Service Attendants', 1750: 'SCI-Geoscientists And Hydrologists, Except Geographers', 2004: 'CMS-Mental Health Counselors', 2205: 'EDU-Postsecondary Teachers', 3960: 'PRT-Other Protective Service Workers', 5320: 'OFF-Library Assistants, Clerical', 5521: 'OFF-Public Safety Telecommunicators', 6005: 'FFF-First-Line Supervisors Of Farming, Fishing, And Forestry Workers', 6130: 'FFF-Logging Workers', 7315: 'RPR-Heating, Air Conditioning, And Refrigeration Mechanics And Installers', 850: 'FIN-Personal Financial Advisors', 1530: 'ENG-Other Engineers', 2723: 'ENT-Umpires, Referees, And Other Sports Officials', 3740: 'PRT-Firefighters', 5160: 'OFF-Tellers', 5510: 'OFF-Couriers And Messengers', 8320: 'PRD-Sewing Machine Operators', 1822: 'SCI-School Psychologists', 3140: 'MED-Audiologists', 3605: 'HLS-Orderlies and Psychiatric Aides', 3820: 'PRT-Detectives And Criminal Investigators', 5240: 'OFF-Customer Service Representatives', 9410: 'TRN-Transportation Inspectors', 137: 'MGR-Training And Development Managers', 140: 'MGR-Industrial Production Managers', 150: 'MGR-Purchasing Managers', 726: 'BUS-Fundraisers', 1310: 'ENG-Surveyors, Cartographers, And Photogrammetrists', 845: 'FIN-Financial And Investment Analysts', 3256: 'MED-Nurse Anesthetists', 3422: 'MED-Psychiatric Technicians', 5540: 'OFF-Postal Service Clerks', 6050: 'FFF-Other Agricultural Workers', 9510: 'TRN-Crane And Tower Operators', 650: 'BUS-Training And Development Specialists', 2545: 'EDU-Teaching Assistants', 4030: 'EAT-Food Preparation Workers', 4750: 'SAL-Parts Salespersons', 5820: 'OFF-Word Processors And Typists', 6115: 'FFF-Fishing And Hunting Workers', 6120: 'FFF-Forest And Conservation Workers', 6305: 'CON-Construction Equipment Operators', 6410: 'CON-Painters and Paperhangers', 7560: 'RPR-Riggers', 8620: 'PRD-Water And Wastewater Treatment Plant And System Operators', 9600: 'TRN-Industrial Truck And Tractor Operators', 1710: 'SCI-Atmospheric And Space Scientists', 1825: 'SCI-Other Psychologists', 3310: 'MED-Dental Hygienists', 3550: 'MED-Other Healthcare Practitioners and Technical Occupations', 4850: 'SAL-Sales Representatives, Wholesale And Manufacturing', 6260: 'CON-Construction Laborers', 7640: 'RPR-Other Installation, Maintenance, And Repair Workers', 7720: 'PRD-Electrical, Electronics, And Electromechanical Assemblers', 9760: 'TRN-Other Material Moving Workers', 6800: 'EXT-Derrick, Rotary Drill, And Service Unit Operators, And Roustabouts, Oil And Gas', 700: 'BUS-Logisticians', 1610: 'SCI-Biological Scientists', 2025: 'CMS-Other Community and Social Service Specialists', 2710: 'ENT-Producers And Directors', 3424: 'MED-Veterinary Technologists and Technicians', 3940: 'PRT-Crossing Guards And Flaggers', 5260: 'OFF-File Clerks', 5360: 'OFF-Human Resources Assistants, Except Payroll And Timekeeping', 9150: 'TRN-Motor Vehicle Operators, All Other', 440: 'MGR-Other Managers', 1460: 'ENG-Mechanical Engineers', 2040: 'CMS-Clergy', 2740: 'ENT-Dancers And Choreographers', 3323: 'MED-Radiologic Technologists And Technicians', 4760: 'SAL-Retail Salespersons', 4830: 'SAL-Travel Agents', 5350: 'OFF-Correspondence Clerks And Order Clerks', 8990: 'PRD-Miscellaneous Production Workers, Including Equipment Operators And Tenders', 360: 'MGR-Natural Sciences Managers', 530: 'BUS-Purchasing Agents, Except Wholesale, Retail, And Farm Products', 2005: 'CMS-Rehabilitation Counselors', 2632: 'ENT-Fashion Designers', 3900: 'PRT-Animal Control Workers', 8465: 'PRD-Other Textile, Apparel, And Furnishings Workers', 8640: 'PRD-Chemical Processing Machine Setters, Operators, And Tenders', 8740: 'PRD-Inspectors, Testers, Sorters, Samplers, And Weighers', 8920: 'PRD-Molders, Shapers, And Casters, Except Metal And Plastic', 2805: 'ENT-Broadcast Announcers And Radio Disc Jockeys', 3040: 'MED-Optometrists', 3050: 'MED-Pharmacists', 4621: 'PRS-Exercise Trainers And Group Fitness Instructors', 4820: 'SAL-Securities, Commodities, And Financial Services Sales Agents', 5110: 'OFF-Billing And Posting Clerks', 6515: 'CON-Roofers', 6710: 'CON-Fence Erectors', 9240: 'TRN-Railroad Conductors And Yardmasters', 4640: 'PRS-Residential Advisors', 6230: 'CON-Carpenters', 6400: 'CON-Insulation Workers', 8630: 'PRD-Miscellaneous Plant And System Operators', 9110: 'TRN-Ambulance Drivers And Attendants, Except Emergency Medical Technicians', 900: 'FIN-Financial Examiners', 1430: 'ENG-Industrial Engineers, Including Health And Safety', 1545: 'ENG-Other Drafters', 1821: 'SCI-Clinical And Counseling Psychologists', 2013: 'CMS-Mental Health And Substance Abuse Social Workers', 4350: 'PRS-Animal Caretakers', 7540: 'RPR-Locksmiths And Safe Repairers', 8130: 'PRD-Tool And Die Makers', 8720: 'PRD-Extruding, Forming, Pressing, And Compacting Machine Setters, Operators, And Tenders', 9610: 'TRN-Cleaners Of Vehicles And Equipment', 205: 'MGR-Farmers, Ranchers, And Other Agricultural Managers', 600: 'BUS-Cost Estimators', 2752: 'ENT-Musicians and Singers', 7160: 'RPR-Automotive Glass Installers And Repairers', 7330: 'RPR-Industrial And Refractory Machinery Mechanics', 2920: 'ENT-Television, Video, And Film Camera Operators And Editors', 136: 'MGR-Human Resources Managers', 1006: 'CMM-Computer Systems Analysts', 2014: 'CMS-Social Workers, All Other', 3010: 'MED-Dentists', 3100: 'MED-Surgeons', 4040: 'EAT-Bartenders', 4530: 'PRS-Baggage Porters, Bellhops, And Concierges', 5230: 'OFF-Credit Authorizers, Checkers, And Clerks', 6540: 'CON-Solar Photovoltaic Installers', 7320: 'RPR-Home Appliance Repairers', 8040: 'PRD-Metal Furnace Operators, Tenders, Pourers, And Casters', 2360: 'EDU-Other Teachers and Instructors', 2700: 'ENT-Actors', 2722: 'ENT-Coaches and Scouts', 3648: 'HLS-Veterinary Assistants And Laboratory Animal Caretakers', 4255: 'CLN-Other Grounds Maintenance Workers', 6360: 'CON-Glaziers', 8500: 'PRD-Cabinetmakers And Bench Carpenters', 1555: 'ENG-Other Engineering Technologists And Technicians, Except Drafters', 1320: 'ENG-Aerospace Engineers', 2840: 'ENT-Technical Writers', 3300: 'MED-Clinical Laboratory Technologists And Technicians', 6330: 'CON-Drywall Installers, Ceiling Tile Installers, And Tapers', 7800: 'PRD-Bakers', 7855: 'PRD-Food Processing Workers, All Other', 1340: 'ENG-Biomedical And Agricultural Engineers', 1440: 'ENG-Marine Engineers And Naval Architects', 1745: 'SCI-Environmental Scientists And Specialists, Including Health', 2170: 'LGL-Title Examiners, Abstractors, and Searchers', 3090: 'MED-Physicians', 3645: 'HLS-Medical Assistants', 4900: 'SAL-Models, Demonstrators, And Product Promoters', 8510: 'PRD-Furniture Finishers', 9005: 'TRN-Supervisors Of Transportation And Material Moving Workers', 725: 'BUS-Meeting, Convention, And Event Planners', 2400: 'EDU-Archivists, Curators, And Museum Technicians', 3423: 'MED-Surgical Technologists', 6950: 'EXT-Other Extraction Workers', 7040: 'RPR-Electric Motor, Power Tool, And Related Repairers', 9300: 'TRN-Sailors And Marine Oilers, And Ship Engineers', 9640: 'TRN-Packers And Packagers, Hand', 3322: 'MED-Diagnostic Medical Sonographers', 4220: 'CLN-Janitors And Building Cleaners', 4230: 'CLN-Maids And Housekeeping Cleaners', 5522: 'OFF-Dispatchers, Except Police, Fire, And Ambulance', 5900: 'OFF-Office Machine Operators, Except Computer', 7430: 'RPR-Precision Instrument And Equipment Repairers', 8940: 'PRD-Tire Builders', 9142: 'TRN-Taxi Drivers', 9825: 'MIL-Military Enlisted Tactical Operations And Air/Weapons Specialists And Crew Members', 3930: 'PRT-Security Guards And Gambling Surveillance Officers', 1220: 'CMM-Operations Research Analysts', 1600: 'SCI-Agricultural And Food Scientists', 1520: 'ENG-Petroleum, Mining And Geological Engineers, Including Mining Safety Engineers', 2440: 'EDU-Library Technicians', 2830: 'ENT-Editors', 3610: 'HLS-Occupational Therapy Assistants And Aides', 4251: 'CLN-Landscaping And Groundskeeping Workers', 4655: 'PRS-Personal Care and Service Workers, All Other', 5100: 'OFF-Bill And Account Collectors', 6520: 'CON-Sheet Metal Workers', 8530: 'PRD-Sawing Machine Setters, Operators, And Tenders, Wood', 9645: 'TRN-Stockers And Order Fillers', 7100: 'RPR-Other Electrical And Electronic Equipment Mechanics, Installers, And Repairers', 830: 'FIN-Credit Analysts', 
        5550: 'OFF-Postal Service Mail Carriers', 7220: 'RPR-Heavy Vehicle And Mobile Equipment Service Technicians And Mechanics', 1551: 'ENG-Electrical And Electronic Engineering Technologists And Technicians', 1700: 'SCI-Astronomers And Physicists', 3000: 'MED-Chiropractors', 3801: 'PRT-Bailiffs', 7950: 'PRD-Cutting, Punching, And Press Machine Setters, Operators, And Tenders, Metal And Plastic', 9310: 'TRN-Ship And Boat Captains And Operators', 350: 'MGR-Medical And Health Services Managers', 1106: 'CMM-Computer Network Architects', 1360: 'ENG-Civil Engineers', 1980: 'SCI-Occupational Health And Safety Specialists and Technicians', 2001: 'CMS-Substance Abuse And Behavioral Disorder Counselors', 2635: 'ENT-Interior Designers', 3110: 'MED-Physician Assistants', 7700: 'PRD-First-Line Supervisors Of Production And Operating Workers', 60: 'MGR-Public Relations And Fundraising Managers', 120: 'MGR-Financial Managers', 3210: 'MED-Recreational Therapists', 5410: 'OFF-Reservation And Transportation Ticket Agents And Travel Clerks', 7240: 'RPR-Small Engine Mechanics', 9130: 'TRN-Driver/Sales Workers And Truck Drivers', 101: 'MGR-Administrative Services Managers', 3030: 'MED-Dietitians And Nutritionists', 4400: 'PRS-Gambling Services Workers', 7740: 'PRD-Structural Metal Fabricators And Fitters', 9210: 'TRN-Locomotive Engineers And Operators', 9810: 'MIL-First-Line Enlisted Military Supervisors', 1050: 'CMM-Computer Support Specialists', 3230: 'MED-Speech-Language Pathologists', 4330: 'PRS-Supervisors Of Personal Care And Service Workers', 4510: 'PRS-Hairdressers, Hairstylists, And Cosmetologists', 4740: 'SAL-Counter And Rental Clerks', 5165: 'OFF-Other Financial Clerks', 6460: 'CON-Plasterers And Stucco Masons', 6765: 'CON-Other Construction And Related Workers', 8650: 'PRD-Crushing, Grinding, Polishing, Mixing, And Blending Workers', 3401: 'MED-Emergency Medical Technicians', 3700: 'PRT-First-Line Supervisors Of Correctional Officers', 5600: 'OFF-Production, Planning, And Expediting Clerks', 5720: 'OFF-Legal Secretaries and Administrative Assistants', 8540: 'PRD-Woodworking Machine Setters, Operators, And Tenders, Except Sawing', 8730: 'PRD-Furnace, Kiln, Oven, Drier, And Kettle Operators And Tenders', 8910: 'PRD-Etchers And Engravers', 9050: 'TRN-Flight Attendants', 20: 'MGR-General And Operations Managers', 750: 'BUS-Business Operations Specialists, All Other', 810: 'FIN-Property Appraisers and Assessors', 2012: 'CMS-Healthcare Social Workers', 3321: 'MED-Cardiovascular Technologists and Technicians', 5810: 'OFF-Data Entry Keyers', 6441: 'CON-Pipelayers', 8255: 'PRD-Printing Press Operators', 1031: 'CMM-Web Developers', 1240: 'CMM-Other Mathematical Science Occupations', 1650: 'SCI-Other Life Scientists', 1840: 'SCI-Urban And Regional Planners', 3270: 'MED-Healthcare Diagnosing Or Treating Practitioners, All Other', 9720: 'TRN-Refuse And Recyclable Material Collectors', 310: 'MGR-Food Service Managers', 3520: 'MED-Opticians, Dispensing', 4140: 'EAT-Dishwashers', 4950: 'SAL-Door-To-Door Sales Workers, News And Street Vendors, And Related Workers', 7210: 'RPR-Bus And Truck Mechanics And Diesel Engine Specialists', 9920: 'Unemployed, With No Work Experience In The Last 5 Years Or Earlier Or Never Worked', 5740: 'OFF-Secretaries And Administrative Assistants, Except Legal, Medical, And Executive', 51: 'MGR-Marketing Managers', 300: 'MGR-Architectural And Engineering Managers', 2016: 'CMS-Social And Human Service Assistants', 3120: 'MED-Podiatrists', 3255: 'MED-Registered Nurses', 4500: 'PRS-Barbers', 4920: 'SAL-Real Estate Brokers And Sales Agents', 6200: 'CON-First-Line Supervisors Of Construction Trades And Extraction Workers', 6730: 'CON-Highway Maintenance Workers', 7030: 'RPR-Avionics Technicians', 860: 'FIN-Insurance Underwriters', 1560: 'ENG-Surveying And Mapping Technicians', 2634: 'ENT-Graphic Designers', 3710: 'PRT-First-Line Supervisors Of Police And Detectives', 4000: 'EAT-Chefs And Head Cooks', 9430: 'TRN-Other Transportation Workers', 800: 'FIN-Accountants And Auditors', 2825: 'ENT-Public Relations Specialists', 3602: 'HLS-Personal Care Aides', 4840: 'SAL-Sales Representatives Of Services, Except Advertising, Insurance, Financial Services, And Travel', 5120: 'OFF-Bookkeeping, Accounting, And Auditing Clerks', 5220: 'OFF-Court, Municipal, And License Clerks', 7010: 'RPR-Computer, Automated Teller, And Office Machine Repairers', 7340: 'RPR-Maintenance And Repair Workers, General', 7410: 'RPR-Electrical Power-Line Installers And Repairers', 8256: 'PRD-Print Binding And Finishing Workers', 9620: 'TRN-Laborers And Freight, Stock, And Material Movers, Hand', 52: 'MGR-Sales Managers', 540: 'BUS-Claims Adjusters, Appraisers, Examiners, And Investigators', 2600: 'ENT-Artists And Related Workers', 2770: 'ENT-Entertainers And Performers, Sports and Related Workers, All Other', 4150: 'EAT-Hosts And Hostesses, Restaurant, Lounge, And Coffee Shop', 4930: 'SAL-Sales Engineers', 7510: 'RPR-Coin, Vending, And Amusement Machine Servicers And Repairers', 2751: 'ENT-Music Directors and Composers', 4420: 'PRS-Ushers, Lobby Attendants, And Ticket Takers', 5000: 'OFF-First-Line Supervisors Of Office And Administrative Support Workers', 5920: 'OFF-Statistical Assistants', 6835: 'EXT-Explosives Workers, Ordnance Handling Experts, and Blasters', 7830: 'PRD-Food And Tobacco Roasting, Baking, And Drying Machine Operators And Tenders', 8350: 'PRD-Tailors, Dressmakers, And Sewers', 40: 'MGR-Advertising And Promotions Managers', 940: 'FIN-Tax Preparers', 1021: 'CMM-Software Developers', 1400: 'ENG-Computer Hardware Engineers', 735: 'BUS-Market Research Analysts And Marketing Specialists', 1350: 'ENG-Chemical Engineers', 2050: 'CMS-Directors, Religious Activities And Education', 2435: 'EDU-Librarians And Media Collections Specialists', 3500: 'MED-Licensed Practical And Licensed Vocational Nurses', 6250: 'CON-Cement Masons, Concrete Finishers, And Terrazzo Workers', 6442: 'CON-Plumbers, Pipefitters, And Steamfitters', 7840: 'PRD-Food Batchmakers', 8800: 'PRD-Packaging And Filling Machine Operators And Tenders', 8000: 'PRD-Grinding, Lapping, Polishing, And Buffing Machine Tool Setters, Operators, And Tenders, Metal And Plastic', 1760: 'SCI-Physical Scientists, All Other', 2100: 'LGL-Lawyers, And Judges, Magistrates, And Other Judicial Workers', 2105: 'LGL-Judicial Law Clerks', 3330: 'MED-Nuclear Medicine Technologists and Medical Dosimetrists', 3515: 'MED-Medical Records Specialists', 4700: 'SAL-First-Line Supervisors Of Retail Sales Workers', 5010: 'OFF-Switchboard Operators, Including Answering Service', 6210: 'CON-Boilermakers', 7000: 'RPR-First-Line Supervisors Of Mechanics, Installers, And Repairers', 9800: 'MIL-Military Officer Special And Tactical Operations Leaders', 2862: 'ENT-Court Reporters and Simultaneous Captioners', 4435: 'PRS-Other Entertainment Attendants And Related Workers', 5910: 'OFF-Proofreaders And Copy Markers', 8335: 'PRD-Shoe And Leather Workers', 8450: 'PRD-Upholsterers', 8760: 'PRD-Dental And Ophthalmic Laboratory Technicians And Medical Appliance Technicians', 4461: 'PRS-Embalmers, Crematory Operators And Funeral Attendants', 520: 'BUS-Wholesale And Retail Buyers, Except Farm Products', 1200: 'CMM-Actuaries', 1910: 'SCI-Biological Technicians', 1920: 'SCI-Chemical Technicians', 2350: 'EDU-Tutors', 3601: 'HLS-Home Health Aides', 4521: 'PRS-Manicurists And Pedicurists', 5940: 'OFF-Other Office And Administrative Support Workers', 8850: 'PRD-Adhesive Bonding Machine Operators And Tenders', 6700: 'CON-Elevator and Escalator Installers And Repairers', 230: 'MGR-Education And Childcare Administrators', 3402: 'MED-Paramedics', 5560: 'OFF-Postal Service Mail Sorters, Processors, And Processing Machine Operators', 6660: 'CON-Construction And Building Inspectors', 7850: 'PRD-Food Cooking Machine Operators And Tenders', 7905: 'PRD-Computer Numerically Controlled Tool Operators And Programmers', 9030: 'TRN-Aircraft Pilots And Flight Engineers', 410: 'MGR-Property, Real Estate, And Community Association Managers', 1108: 'CMM-Computer Occupations, All Other', 1640: 'SCI-Conservation Scientists And Foresters', 3245: 'MED-Other Therapists', 6040: 'FFF-Graders And Sorters, Agricultural Products', 7810: 'PRD-Butchers And Other Meat, Poultry, And Fish Processing Workers', 9650: 'TRN-Pumping Station Operators', 9830: 'MIL-Military, Rank Not Specified', 565: 'BUS-Compliance Officers', 1065: 'CMM-Database Administrators and Architects', 2002: 'CMS-Educational, Guidance, And Career Counselors And Advisors', 8365: 'PRD-Textile Machine Setters, Operators, And Tenders', 2145: 'LGL-Paralegals And Legal Assistants', 3250: 'MED-Veterinarians', 4710: 'SAL-First-Line Supervisors Of Non-Retail Sales Workers', 5530: 'OFF-Meter Readers, Utilities', 7610: 'RPR-Helpers--Installation, Maintenance, And Repair Workers', 9121: 'TRN-Bus Drivers, School', 135: 'MGR-Compensation And Benefits Managers', 220: 'MGR-Construction Managers', 910: 'FIN-Credit Counselors And Loan Officers', 710: 'BUS-Management Analysts', 1306: 'ENG-Landscape Architects', 3258: 'MED-Nurse Practitioners, And Nurse Midwives', 3261: 'MED-Acupuncturists', 4525: 'PRS-Other Personal Appearance Workers', 5310: 'OFF-Interviewers, Except Eligibility And Loan', 5850: 'OFF-Mail Clerks And Mail Machine Operators, Except Postal Service', 6720: 'CON-Hazardous Materials Removal Workers', 7300: 'RPR-Control And Valve Installers And Repairers', 7420: 'RPR-Telecommunications Line Installers And Repairers', 8100: 'PRD-Model Makers, Patternmakers, And Molding Machine Setters, Metal And Plastic', 930: 'FIN-Tax Examiners And Collectors, And Revenue Agents', 1935: 'SCI-Environmental Science and Geoscience Technicians, And Nuclear Technicians', 1970: 'SCI-Other Life, Physical, And Social Science Technicians', 8300: 'PRD-Laundry And Dry-Cleaning Workers', 8830: 'PRD-Photographic Process Workers And Processing Machine Operators'
        }

pobp_dict={
        300: 'Bermuda', 120: 'Italy', 211: 'Indonesia', 501: 'Australia', 515: 'New Zealand', 54: 'West Virginia/WV', 130: 'Azores Islands', 154: 'Serbia', 44: 'Rhode Island/RI', 368: 'Guyana', 341: 'Trinidad and Tobago', 34: 'New Jersey/NJ', 42: 'Pennsylvania/PA', 249: 'Asia', 425: 'Ivory Coast (2017 or later)', 508: 'Fiji', 10: 'Delaware/DE', 36: 'New York/NY', 421: 'Ghana', 157: 'Lithuania', 205: 'Myanmar', 219: 'Kyrgyzstan (2017 or later)', 429: 'Liberia', 373: 'Venezuela', 370: 'Peru', 159: 'Azerbaijan', 449: 'South Africa', 360: 'Argentina', 333: 'Jamaica', 460: 'Zambia', 5: 'Arkansas/AR', 245: 'United Arab Emirates', 469: 'Eastern Africa, Not Specified', 108: 'Finland', 447: 'Sierra Leone', 235: 'Saudi Arabia', 399: 'Americas, Not Specified', 451: 'Sudan', 208: 'Cyprus (2016 or earlier)', 109: 'France', 365: 'Ecuador', 119: 'Ireland', 16: 'Idaho/ID', 468: 'Other Africa, Not Specified', 158: 'Armenia', 21: 'Kentucky/KY', 127: 'Norway', 136: 'Sweden', 457: 'Uganda', 323: 'Bahamas', 15: 'Hawaii/HI', 72: 'Puerto Rico', 137: 'Switzerland', 328: 'Dominica', 233: 'Philippines', 238: 'Sri Lanka', 161: 'Georgia', 18: 'Indiana/IN', 228: 'Mongolia (2017 or later)', 369: 'Paraguay', 169: 'Other Europe, Not Specified', 45: 'South Carolina/SC', 200: 'Afghanistan', 372: 'Uruguay', 454: 'Togo', 459: 'Democratic Republic of Congo (Zaire)', 117: 'Hungary', 242: 'Thailand', 28: 'Mississippi/MS', 202: 'Bangladesh', 314: 'Honduras', 321: 'Antigua and Barbuda', 239: 'Syria', 222: 'Kuwait', 9: 'Connecticut/CT', 48: 'Texas/TX', 102: 'Austria', 248: 'Yemen', 168: 'Montenegro', 206: 'Cambodia', 66: 'Guam', 511: 'Marshall Islands', 11: 'District of Columbia/DC', 13: 'Georgia/GA', 462: 'Africa', 126: 'Netherlands', 338: 'St. Kitts-Nevis (2017 or later)', 442: 'Rwanda (2017 or later)', 110: 'Germany', 316: 'Panama', 156: 'Latvia', 30: 'Montana/MT', 301: 'Canada', 312: 'El Salvador', 22: 'Louisiana/LA', 311: 'Costa Rica', 361: 'Bolivia', 27: 'Minnesota/MN', 53: 'Washington/WA', 374: 'South America', 224: 'Lebanon', 209: 'Hong Kong', 453: 'Tanzania', 448: 'Somalia', 329: 'Dominican Republic', 218: 'Kazakhstan', 25: 'Massachusetts/MA', 253: 'South Central Asia, Not Specified', 167: 'Kosovo (2017 or later)', 140: 'Scotland', 416: 'Ethiopia', 340: 'St. Vincent and the Grenadines', 243: 'Turkey', 163: 'Russia', 2: 'Alaska/AK', 26: 'Michigan/MI', 554: 'Other US Island Areas, Oceania, Not Specified, or at Sea', 313: 'Guatemala', 150: 'Bosnia and Herzegovina', 364: 'Colombia', 8: 'Colorado/CO', 129: 'Portugal', 134: 'Spain', 162: 'Moldova', 1: 'Alabama/AL', 105: 'Czechoslovakia', 212: 'Iran', 332: 'Haiti', 139: 'England', 210: 'India', 440: 'Nigeria', 400: 'Algeria', 217: 'Korea', 160: 'Belarus', 33: 'New Hampshire/NH', 37: 'North Carolina/NC', 78: 'US Virgin Islands', 467: 'Western Africa, Not Specified', 152: 'Macedonia', 444: 'Senegal', 339: 'St. Lucia', 223: 'Laos', 423: 'Guinea', 19: 'Iowa/IA', 39: 'Ohio/OH', 55: 'Wisconsin/WI', 463: 'South Sudan (2017 or later)', 207: 'China', 203: 'Bhutan', 151: 'Croatia', 38: 'North Dakota/ND', 527: 'Samoa', 417: 'Eritrea', 46: 'South Dakota/SD', 142: 'Northern Ireland (2017 or later)', 4: 'Arizona/AZ', 254: 'Other Asia, Not Specified', 106: 'Denmark', 213: 'Iraq', 12: 'Florida/FL', 17: 'Illinois/IL', 330: 'Grenada', 303: 'Mexico', 247: 'Vietnam', 50: 'Vermont/VT', 56: 'Wyoming/WY', 215: 'Japan', 165: 'USSR', 148: 'Czech Republic', 412: 'Congo', 49: 'Utah/UT', 216: 'Jordan', 324: 'Barbados', 24: 'Maryland/MD', 464: 'Northern Africa, Not Specified', 128: 'Poland', 436: 'Morocco', 461: 'Zimbabwe', 246: 'Uzbekistan', 420: 'Gambia', 456: 'Tunisia (2017 or later)', 344: 'Caribbean, Not Specified', 327: 'Cuba', 236: 'Singapore', 100: 'Albania', 430: 'Libya', 41: 'Oregon/OR', 164: 'Ukraine', 23: 'Maine/ME', 103: 'Belgium', 427: 'Kenya', 60: 'American Samoa', 118: 'Iceland', 343: 'West Indies', 408: 'Cabo Verde', 363: 'Chile', 47: 'Tennessee/TN', 414: 'Egypt', 149: 'Slovakia', 512: 'Micronesia', 523: 'Tonga', 226: 'Malaysia', 6: 'California/CA', 20: 'Kansas/KS', 407: 'Cameroon', 315: 'Nicaragua', 229: 'Nepal', 132: 'Romania', 214: 'Israel', 310: 'Belize', 104: 'Bulgaria', 51: 'Virginia/VA', 35: 'New Mexico/NM', 69: 'Commonwealth of the Northern Mariana Islands', 166: 'Europe (2017 or later)', 32: 'Nevada/NV', 231: 'Pakistan', 31: 'Nebraska/NE', 138: 'United Kingdom, Not Specified', 116: 'Greece', 147: 'Yugoslavia', 362: 'Brazil', 240: 'Taiwan', 29: 'Missouri/MO', 40: 'Oklahoma/OK'
        }

relp_dict={
    8: "Parent-in-law",
    14: "Foster child",
    11: "Roomer or boarder",
    2: "Biological son or daughter",
    4: "Stepson or stepdaughter",
    17: "Noninstitutionalized group quarters population",
    10: "Other relative",
    7: "Grandchild",
    5: "Brother or sister",
    3: "Adopted son or daughter",
    15: "Other nonrelative",
    0: "Reference person",
    6: "Father or mother",
    1: "Husband/wife",
    12: "Housemate or roommate",
    16: "Institutionalized group quarters population",
    9: "Son-in-law or daughter-in-law",
    13: "Unmarried partner"
}

age_col = ColumnToText(
    "AGEP",
    short_description="age",
    value_map=lambda x: f"{int(x)} years old",
)

education_col = ColumnToText(
    "SCHL",
    short_description="education",
    value_map={
          4: "Grade 1",
          1: "No schooling completed",
          7: "Grade 4",
          16: "Regular high school diploma",
          3: "Kindergarten",
          23: "Professional degree beyond a bachelor's degree",
          19: "1 or more years of college credit, no degree",
          10: "Grade 7",
          22: "Master's degree",
          20: "Associate's degree",
          0: "N/A (less than 3 years old)",
          2: "Nursery school, preschool",
          21: "Bachelor's degree",
          8: "Grade 5",
          24: "Doctorate degree",
          14: "Grade 11",
          6: "Grade 3",
          17: "GED or alternative credential",
          12: "Grade 9",
          13: "Grade 10",
          9: "Grade 6",
          5: "Grade 2",
          15: "12th grade - no diploma",
          11: "Grade 8",
          18: "Some college, but less than 1 year"
    },
)

race_col = ColumnToText(
    "RAC1P",
    short_description="race",
    value_map={
          3: "American Indian alone",
          1: "White alone",
          8: "Some Other Race alone",
          6: "Asian alone",
          9: "Two or More Races",
          2: "Black or African American alone",
          4: "Alaska Native alone",
          7: "Native Hawaiian and Other Pacific Islander alone",
          5: "American Indian and Alaska Native tribes; or not specified and no other races"
    },
)

gender_col = ColumnToText(
    "SEX",
    short_description="gender",
    value_map={
        1: "Male",
        2: "Female",
    },
)

# cow_col = ColumnToText(
#     "COW",
#     short_description="class of worker",
#     value_map={
#         9: "Unemployed and last worked 5 years ago or earlier or never worked",
#         1: "Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions",
#         7: "Self-employed in own incorporated business, professional practice or farm",
#         8: "Working without pay in family business or farm",
#         3: "Local government employee (city, county, etc.)",
#         6: "Self-employed in own not incorporated business, professional practice, or farm",
#         4: "State government employee",
#         5: "Federal government employee",
#         2: "Employee of a private not-for-profit, tax-exempt, or charitable organization",
#         0: "Not in universe (less than 16 years old/NILF who last worked more than 5 years ago or never worked)"
#     }
# )

mar_col = ColumnToText(
    "MAR",
    short_description="marriage status",
    value_map={
        1: "Married",
        5: "Never married or under 15 years old",
        4: "Separated",
        3: "Divorced",
        2: "Widowed"
    }
)

# occp_col = ColumnToText(
#     "OCCP",
#     short_description="job type",
#     value_map={
#         k: "works as " + v for k, v in occp_dict.items()
#     }
# )

# pobp_col = ColumnToText(
#     "POBP",
#     short_description="place of birth",
#     value_map={
#         k: "born in " + v for k, v in pobp_dict.items()
#     }
# )

relp_col = ColumnToText(
    "RELP",
    short_description="relationship to head of household",
    value_map={
        k: v + " of head of household" for k, v in relp_dict.items()
    }
)

# wkhp_col = ColumnToText(
#     "WKHP",
#     short_description="hours worked per week",
#     value_map=lambda x: f"works {x} hours per week"
# )

state_col = ColumnToText(
    "STATE",
    short_description="state of residence",
    value_map=lambda x: f"resident of {x}"
)

disability_col = ColumnToText(
    "DIS",
    short_description="disability status",
    value_map={
        1: "With a disability",
        2: "Without a disability"
    }
)

parent_col = ColumnToText(
    "ESP",
    short_description="employment status of parents",
    value_map={
        8: "Living with mother: Mother not in labor force",
        0: "N/A (not own child of householder, and not child in subfamily)",
        2: "Living with two parents: Father only in labor force",
        5: "Living with father: Father in the labor force",
        4: "Living with two parents: Neither parent in labor force",
        7: "Living with mother: Mother in the labor force",
        6: "Living with father: Father not in labor force",
        1: "Living with two parents: Both parents in labor force",
        3: "Living with two parents: Mother only in labor force"
    }
)

citizen_col = ColumnToText(
    "CIT",
    short_description="citizenship status",
    value_map={
        3: "Born abroad of American parent(s)",
        2: "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas",
        4: "U.S. citizen by naturalization",
        1: "Born in the U.S.",
        5: "Not a citizen of the U.S."
    }
)

mobility_col = ColumnToText(
    "MIG",
    short_description="mobility status (lived here 1 year ago)",
    value_map={
        2: "No, outside US and Puerto Rico",
        3: "No, different house in US or Puerto Rico",
        1: "Yes, same house (nonmovers)",
        0: "N/A(less than 1 year old)"        
    }
)

military_col = ColumnToText(
    "MIL",
    short_description="military service",
    value_map={
        0: "N/A (less than 17 years old)",
        1: "Now on active duty",
        4: "Never served in the military",
        2: "On active duty in the past, but not now",
        3: "Only on active duty for training in Reserves/National Guard"        
    }
)

ancestry_col = ColumnToText(
    "ANC",
    short_description="number ancestries reported",
    value_map={
        4: "Not reported",
        8: "Suppressed for data year 2018 for select PUMAs",
        3: "Unclassified",
        2: "Multiple",
        1: "Single"
    }
)

nativity_col = ColumnToText(
    "NATIVITY",
    short_description="nativity for US",
    value_map={
        2: "Foreign born",
        1: "Native"        
    }
)

hearing_col = ColumnToText(
    "DEAR",
    short_description="hearing difficulty",
    value_map={
        2: "No",
        1: "Yes"        
    }
)

eye_col = ColumnToText(
    "DEYE",
    short_description="vision difficulty",
    value_map={
        2: "No",
        1: "Yes"        
    }
)

cognitive_col = ColumnToText(
    "DREM",
    short_description="cognitive difficulty",
    value_map={
        2: "No",
        1: "Yes",
        0: "N/A (less than 5 years old)",
    }
)

employment_col = ColumnToText(
    "ESR",
    short_description="employment status",
    value_map={
        1: "employed",
        0: "not employed"
    }
)

# acdhs_race_col = ColumnToText(
#     "RACE",
#     short_description="race",
#     value_map={
#         '1~White': "White",
#         '2~Black/African American': "Black",
#         '4~Asian': "Asian",
#         '3~American Indian/Alaskan Native': "Native American",
#         '7~Other Single Race': "Other (Single Race)",
#         '5~Native Hawaiian/Pacific Islander': "Pacific Islander",
#         '6~Two or More Races': "Two or More Races",
#         '99~Unknown': "Unknown",
#     },
# )


# acdhs_bh_ever_col = ColumnToText(
#     "BH_DIAGNOSIS_EVER_DUMMY",
#     short_description="indicator for any previous behavioral health diagnosis",
#     value_map={
#         0: "No",
#         1: "Yes",
#     },
# )


# acdhs_bh_inpatient_col = ColumnToText(
#     "BH_INPATIENT_REHAB_DAYS_CT_3_YEARS",
#     short_description="days spent in inpatient behavioral health rehabilitation in the last 3 years",
#     value_map=lambda x: f"{int(x)} days",
# )

# acdhs_acj_months_col = ColumnToText(
#     "ACJ_MONTHS_CT_3_YEARS",
#     short_description="months spent in jail in the last three years",
#     value_map=lambda x: f"{int(x)} months",
# )

# acdhs_court_all_col = ColumnToText(
#     "COURT_ALL_CT_EVER",
#     short_description="number of previous court appearances",
#     value_map=lambda x: f"{int(x)}",
# )

# acdhs_probation_col = ColumnToText(
#     "COURT_PROBATION_CT_2_YEARS",
#     short_description="number of times on probation in the last two years",
#     value_map=lambda x: f"{int(x)}",
# )

# acdhs_bh_count_col = ColumnToText(
#     "BH_ALL_SERVICES_CT",
#     short_description="number of behavioral health services previously received:",
#     value_map=lambda x: f"{int(x)}",
# )

# acdhs_housing_count_col = ColumnToText(
#     "HOUSING_ALL_SERVICES_CT",
#     short_description="number of housing services previously received:",
#     value_map=lambda x: f"{int(x)}",
# )

# acdhs_shelter_count_col = ColumnToText(
#     "HOUS_SHELTER_EE_DAYS_1YR",
#     short_description="number of days spent in homeless shelters in the last year:",
#     value_map=lambda x: f"{int(x)}",
# )

# acdhs_jailstays_col = ColumnToText(
#     "N_JAILSTAYS_LAST_YEAR",
#     short_description="number of stays in jail in the last year:",
#     value_map=lambda x: f"{int(x)}",
# )

# acdhs_out_da_count_col = ColumnToText(
#     "BH_OUTPT_DRUG_ALC_DAYS_CT_EVER",
#     short_description="number of days previously spent in outpatient drug and alcohol treatment:",
#     value_map=lambda x: f"{int(x)}",
# )

# acdhs_bh_emergency_count_col = ColumnToText(
#     "BH_EMEGCY_MNTL_CT_EVER",
#     short_description="number of previous behavioral health emergency visits:",
#     value_map=lambda x: f"{int(x)}",
# )

# acdhs_bh_inpatient_count_col = ColumnToText(
#     "BH_INPT_MNTL_CT_EVER",
#     short_description="number of previous behavioral health inpatient visits:",
#     value_map=lambda x: f"{int(x)}",
# )



reentry_numeric_qa = DirectNumericQA(
    column='ESR',
    text=(
        "Is this person employed?"
    ),
)


reentry_qa = MultipleChoiceQA(
    column='ESR',
    text="Is this person employed?",
    choices=(
        Choice("Yes, they are employed", 1),
        Choice("No, they are not employed", 0),
    ),
)

# shelter_qa = MultipleChoiceQA(
#     column='EMERG_SHLTR',
#     text="Will this person use a homeless shelter in the next year?",
#     choices=(
#         Choice("Yes, they will use a homeless shelter in the next year", 1),
#         Choice("No, they will not use a homeless shelter in the next year", 0),
#     ),
# )

# shelter_numeric_qa = DirectNumericQA(
#     column='ONE_YEAR_SHELTER',
#     text=(
#         "Will this person use a homeless shelter in the next year?"
#     ),
# )

# mhip_qa = MultipleChoiceQA(
#     column='MHIP',
#     text="Will this person have inpatient mental health treatment in the next year?",
#     choices=(
#         Choice("Yes, they will have inpatient mental health treatment in the next year", 1),
#         Choice("No, they will not have inpatient mental health treatment in the next year", 0),
#     ),
# )

# mhip_numeric_qa = DirectNumericQA(
#     column='MHIP',
#     text=(
#         "Will this person have inpatient mental health treatment in the next year?"
#     ),
# )

# ed_qa = MultipleChoiceQA(
#     column='FOUR_ER',
#     text="Will this person have at least four emergency department visits in the next year?",
#     choices=(
#         Choice("Yes, they will have at least four emergency department visits in the next year", 1),
#         Choice("No, they will not have at least four emergency department visits in the next year", 0),
#     ),
# )

# ed_numeric_qa = DirectNumericQA(
#     column='FOUR_ER',
#     text=(
#         "Will this person have at least four emergency department visits in the next year?"
#     ),
# )



# reentry_outcome_col = ColumnToText(
#     'JAIL',
#     short_description="reentry within one year",
#     question=reentry_qa,
# )

# shelter_outcome_col = ColumnToText(
#     'EMERG_SHLTR',
#     short_description="shelter useage within one year",
#     question=shelter_qa,
# )

# invol_outcome_col = ColumnToText(
#     'MHIP',
#     short_description="inpatient mental health treatment within one year",
#     question=mhip_qa,
# )

# mortality_outcome_col = ColumnToText(
#     'FOUR_ER',
#     short_description="at least four emergency department visits within one year",
#     question=ed_qa,
# )


columns_map: dict[str, object] = {
    col_mapper.name: col_mapper
    for col_mapper in globals().values()
    if isinstance(col_mapper, ColumnToText)
}


# all_outcomes = ['JAIL', 'FOUR_ER', 'EMERG_SHLTR', 'MHIP']
all_outcomes = ["ESR"]

reentry_task = TaskMetadata(
    name="employment prediction",
    description=TASK_DESCRIPTION,
    features=[x for x in columns_map.keys() if x not in all_outcomes],
    target='ESR',
    cols_to_text=columns_map,
    sensitive_attribute=None,
    multiple_choice_qa=reentry_qa,
    direct_numeric_qa=reentry_numeric_qa,
)

# shelter_task = TaskMetadata(
#     name="shelter prediction",
#     description=TASK_DESCRIPTION,
#     features=[x for x in columns_map.keys() if x not in all_outcomes],
#     target='EMERG_SHLTR',
#     cols_to_text=columns_map,
#     sensitive_attribute=None,
#     multiple_choice_qa=reentry_qa,
#     direct_numeric_qa=reentry_numeric_qa,
# )

# mhip_task = TaskMetadata(
#     name="mental health inpatient prediction",
#     description=TASK_DESCRIPTION,
#     features=[x for x in columns_map.keys() if x not in all_outcomes],
#     target='MHIP',
#     cols_to_text=columns_map,
#     sensitive_attribute=None,
#     multiple_choice_qa=mhip_qa,
#     direct_numeric_qa=mhip_numeric_qa,
# )

# ed_task = TaskMetadata(
#     name="emergency department prediction",
#     description=TASK_DESCRIPTION,
#     features=[x for x in columns_map.keys() if x not in all_outcomes],
#     target='FOUR_ER',
#     cols_to_text=columns_map,
#     sensitive_attribute=None,
#     multiple_choice_qa=ed_qa,
#     direct_numeric_qa=ed_numeric_qa,
# )

# shelter_task.use_numeric_qa = False
# reentry_task.use_numeric_qa = False
# mhip_task.use_numeric_qa = False
# ed_task.use_numeric_qa = False



data = pd.read_csv("data/acsunemployment_filtered.csv")
num_data = len(data)
# we want to sample 10k
subsampling = 50000 / num_data

reentry_dataset = Dataset(
    data=data,
    task=reentry_task,
    test_size=0.95,
    val_size=0,
    subsampling=subsampling,   # NOTE: Optional, for faster but noisier results!
)




all_tasks = {
    "reentry": [reentry_task, reentry_dataset]
}


model_name = "openai/gpt-4o-mini"
import os
import json
os.environ["OPENAI_API_KEY"] = json.loads("secrets.txt")["open_ai_key"]

for taskname in all_tasks:
    task, dataset = all_tasks[taskname]
    llm_clf = WebAPILLMClassifier(model_name=model_name, task=task)
    llm_clf.set_inference_kwargs(batch_size=500)
    bench = Benchmark(llm_clf=llm_clf, dataset=dataset)

    RESULTS_DIR = "acsunemployment"
    bench.run(results_root_dir=RESULTS_DIR)



# llm_clf = WebAPILLMClassifier(model_name=model_name, task=shelter_task)
# bench = Benchmark(llm_clf=llm_clf, dataset=shelter_dataset)
# RESULTS_DIR = "res_shelter"
# bench.run(results_root_dir=RESULTS_DIR)
