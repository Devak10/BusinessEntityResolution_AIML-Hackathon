"""
Synthetic Dataset Generator for Business Entity Resolution (Step 1 - Scaled to 1,000 Entities)
=============================================================================================

Project Goal:
-------------
Build a Machine Learning system that identifies whether business records from
different data sources refer to the same real-world business entity.

Target Dataset Size:
--------------------
- Source 1: Exactly 1,000 unique clean businesses (Reference / Golden catalog)
- Source 2: ~800 records (ERP / CRM noise profile: abbreviations, legal variations, typos)
- Source 3: ~750 records (Scraped / Web directory noise profile: casing, landmarks, reordering)
- Ground Truth: Master link table mapping each Source 1 ID to comma-separated matched IDs

Relationship Distribution:
--------------------------
- 65-75% matching entities (700 entities, 70.0%)
- 25-35% singletons (300 entities, 30.0%)
- Matches in BOTH Source 2 and Source 3: 380 entities (38.0%)
- Matches in Source 2 ONLY: 180 entities (18.0%)
- Matches in Source 3 ONLY: 140 entities (14.0%)
- Multiple matching records from same source included for ~130 entities
- Balanced across India (500) and USA (500)

Hard Negatives:
---------------
- 60 explicit Hard Negative pairs (120 businesses) in Source 1
- 100+ Hard Negative distractors in Source 2 & Source 3 (same name/diff country,
  same building/diff company, similar name/diff industry)

Reproducibility:
----------------
All random operations use fixed random seed (SEED = 42).
"""

import os
import re
import random
from pathlib import Path
import pandas as pd

RANDOM_SEED = 42


# =====================================================================
# 1. HARD NEGATIVE BASE PAIRS (60 Pairs = 120 Businesses)
# =====================================================================
# Deliberate pairs designed to challenge ML blocking and classification:
# - Same name in different countries / cities
# - Same commercial building / tech park, adjacent suites
# - Sister conglomerate operating companies
# - Very similar brand name in different domain/specialty
# =====================================================================

HARD_NEGATIVE_PAIRS = [
    # -------------------------------------------------------------
    # 1. Cross-Country Name Collisions (15 pairs = 30 businesses)
    # -------------------------------------------------------------
    (
        {"name": "Apex Logistics Private Limited", "address": "Plot 18, Road No. 9, MIDC Industrial Area, Andheri East, Mumbai, Maharashtra 400093", "country": "India"},
        {"name": "Apex Logistics LLC", "address": "2100 North Stemmons Freeway, Suite 450, Dallas, TX 75207", "country": "USA"}
    ),
    (
        {"name": "Summit Capital Partners Private Limited", "address": "Level 8, Express Towers, Nariman Point, Mumbai, Maharashtra 400021", "country": "India"},
        {"name": "Summit Capital Partners LP", "address": "540 Madison Avenue, 28th Floor, New York, NY 10022", "country": "USA"}
    ),
    (
        {"name": "Pinnacle Health Systems India Pvt Ltd", "address": "Road No. 12, Banjara Hills, Hyderabad, Telangana 500034", "country": "India"},
        {"name": "Pinnacle Health Systems Inc.", "address": "111 South Front Street, Harrisburg, PA 17101", "country": "USA"}
    ),
    (
        {"name": "Vanguard Technologies Private Limited", "address": "Plot 35, Electronic City Phase 1, Hosur Road, Bengaluru, Karnataka 560100", "country": "India"},
        {"name": "Vanguard Technologies Inc.", "address": "100 Vanguard Boulevard, Malvern, PA 19355", "country": "USA"}
    ),
    (
        {"name": "Beacon Media Works Private Limited", "address": "Film City Complex, Sector 16A, Noida, Uttar Pradesh 201301", "country": "India"},
        {"name": "Beacon Media Works LLC", "address": "750 Lexington Avenue, 14th Floor, New York, NY 10022", "country": "USA"}
    ),
    (
        {"name": "Horizon Pharma India Private Limited", "address": "Survey No. 42, Gachibowli, Hyderabad, Telangana 500032", "country": "India"},
        {"name": "Horizon Pharma USA Inc.", "address": "150 South Saunders Road, Lake Forest, IL 60045", "country": "USA"}
    ),
    (
        {"name": "Quantum Software Labs Private Limited", "address": "RMZ Ecospace, Campus 3B, Bellandur, Bengaluru, Karnataka 560103", "country": "India"},
        {"name": "Quantum Software Labs LLC", "address": "100 Pine Street, Suite 1250, San Francisco, CA 94111", "country": "USA"}
    ),
    (
        {"name": "Trident Marine Services Private Limited", "address": "Ballard Estate, Fort, Mumbai, Maharashtra 400001", "country": "India"},
        {"name": "Trident Marine Services Inc.", "address": "1015 Third Avenue, Seattle, WA 98104", "country": "USA"}
    ),
    (
        {"name": "Sterling Engineering Works Limited", "address": "Mundhwa Industrial Estate, Pune Cantonment, Pune, Maharashtra 411036", "country": "India"},
        {"name": "Sterling Engineering Works Inc.", "address": "510 Lake Cook Road, Suite 100, Deerfield, IL 60015", "country": "USA"}
    ),
    (
        {"name": "Nexus Renewable Energy Pvt Ltd", "address": "SG Highway, Prahlad Nagar, Ahmedabad, Gujarat 380015", "country": "India"},
        {"name": "Nexus Renewable Energy Corp.", "address": "500 West 2nd Street, Suite 1900, Austin, TX 78701", "country": "USA"}
    ),
    (
        {"name": "BlueWave Analytics Private Limited", "address": "Cyber Towers, HITEC City, Madhapur, Hyderabad, Telangana 500081", "country": "India"},
        {"name": "BlueWave Analytics Inc.", "address": "1200 17th Street, Suite 500, Denver, CO 80202", "country": "USA"}
    ),
    (
        {"name": "Titan Heavy Engineering Private Limited", "address": "Industrial Estate, Guindy, Chennai, Tamil Nadu 600032", "country": "India"},
        {"name": "Titan Heavy Engineering Corp.", "address": "300 Renaissance Center, Detroit, MI 48265", "country": "USA"}
    ),
    (
        {"name": "GreenLeaf BioTech Private Limited", "address": "Wagle Industrial Estate, Thane West, Maharashtra 400604", "country": "India"},
        {"name": "GreenLeaf BioTech Solutions LLC", "address": "777 Mariners Island Blvd, Suite 300, San Mateo, CA 94404", "country": "USA"}
    ),
    (
        {"name": "Atlas Cloud Solutions Private Limited", "address": "DLF Cyber City, Building 5, Gurugram, Haryana 122002", "country": "India"},
        {"name": "Atlas Cloud Solutions LLC", "address": "2300 Oracle Way, Austin, TX 78741", "country": "USA"}
    ),
    (
        {"name": "Novus Financial Advisory Private Limited", "address": "27 BKC, Bandra Kurla Complex, Mumbai, Maharashtra 400051", "country": "India"},
        {"name": "Novus Financial Advisory LLC", "address": "200 West Street, New York, NY 10282", "country": "USA"}
    ),

    # -------------------------------------------------------------
    # 2. Intra-City Similar Name Collisions (15 pairs = 30 businesses)
    # -------------------------------------------------------------
    (
        {"name": "Vertex Solutions India Private Limited", "address": "Campus 3B, RMZ Ecospace, Outer Ring Road, Bellandur, Bengaluru, Karnataka 560103", "country": "India"},
        {"name": "Vertex Financial Solutions Private Limited", "address": "78 Barton Centre, Mahatma Gandhi Road, Bengaluru, Karnataka 560001", "country": "India"}
    ),
    (
        {"name": "Sai Krupa Enterprises", "address": "Shop No. 4, Ground Floor, Laxmi Complex, Pune-Satara Road, Katraj, Pune, Maharashtra 411046", "country": "India"},
        {"name": "Sai Krupa Traders", "address": "Gala No. 12, Market Yard Commercial Complex, Gultekdi, Pune, Maharashtra 411037", "country": "India"}
    ),
    (
        {"name": "Shree Ganesh Sweets & Snacks", "address": "12 JP Road, Near Navrang Cinema, Andheri West, Mumbai, Maharashtra 400058", "country": "India"},
        {"name": "Shree Ganesh Jewellers Limited", "address": "88 SV Road, Opposite Railway Station, Andheri West, Mumbai, Maharashtra 400058", "country": "India"}
    ),
    (
        {"name": "Samarth Agro Industries Private Limited", "address": "Plot 14, MIDC, Bhosari Industrial Area, Pune, Maharashtra 411026", "country": "India"},
        {"name": "Samarth Engineering Works Private Limited", "address": "Plot 88, MIDC, Pimpri Industrial Zone, Pune, Maharashtra 411018", "country": "India"}
    ),
    (
        {"name": "Kalyan Heritage Hotels Private Limited", "address": "Station Road, Near Railway Terminal, Jaipur, Rajasthan 302006", "country": "India"},
        {"name": "Kalyan Jewellers India Limited", "address": "Tonk Road, Opposite Nehru Garden, Jaipur, Rajasthan 302015", "country": "India"}
    ),
    (
        {"name": "Utkarsh Small Finance Bank Limited", "address": "S-24/1-2 First Floor, Mahavir Nagar, Orderly Bazar, Varanasi, Uttar Pradesh 221002", "country": "India"},
        {"name": "Utkarsh Micro Wealth Advisory Pvt Ltd", "address": "Rath Yatra Crossing, Mahmoorganj Road, Varanasi, Uttar Pradesh 221010", "country": "India"}
    ),
    (
        {"name": "Daksha Cyber Security Solutions Pvt Ltd", "address": "Plot 11, Software Units Layout, Madhapur, Hyderabad, Telangana 500081", "country": "India"},
        {"name": "Daksha Data Systems Private Limited", "address": "Road No. 36, Jubilee Hills, Hyderabad, Telangana 500033", "country": "India"}
    ),
    (
        {"name": "Pratham Software Services Private Limited", "address": "G-1/19, EPIP, Sitapura Industrial Area, Jaipur, Rajasthan 302022", "country": "India"},
        {"name": "Pratham Telecom Infrastructure Pvt Ltd", "address": "B-4, Subhash Nagar Shopping Centre, Jaipur, Rajasthan 302016", "country": "India"}
    ),
    (
        {"name": "Summit Global Logistics Inc.", "address": "333 North Michigan Avenue, Suite 1200, Chicago, IL 60601", "country": "USA"},
        {"name": "Summit Global Technologies Inc.", "address": "180 North LaSalle Street, Suite 2400, Chicago, IL 60601", "country": "USA"}
    ),
    (
        {"name": "Pacific Coast Retailers LLC", "address": "100 Pine Street, Suite 1250, San Francisco, CA 94111", "country": "USA"},
        {"name": "Pacific Coast Maritime LLC", "address": "50 California Street, Suite 800, San Francisco, CA 94111", "country": "USA"}
    ),
    (
        {"name": "Horizon Healthcare Services Inc.", "address": "350 Fifth Avenue, Suite 4100, New York, NY 10118", "country": "USA"},
        {"name": "Horizon Cloud Services Inc.", "address": "575 Fifth Avenue, 22nd Floor, New York, NY 10017", "country": "USA"}
    ),
    (
        {"name": "Golden Gate Software LLC", "address": "425 Market Street, Suite 900, San Francisco, CA 94105", "country": "USA"},
        {"name": "Golden Gate Capital Management LLC", "address": "One Embarcadero Center, 39th Floor, San Francisco, CA 94111", "country": "USA"}
    ),
    (
        {"name": "Austin BioLabs Incorporated", "address": "Medical Arts Pavilion, 200 Medical Parkway, Austin, TX 78738", "country": "USA"},
        {"name": "Austin BioTherapeutics Corporation", "address": "3700 North Capital of Texas Highway, Austin, TX 78746", "country": "USA"}
    ),
    (
        {"name": "Denver Mountain Energy Corp.", "address": "1700 Lincoln Street, Suite 2000, Denver, CO 80203", "country": "USA"},
        {"name": "Denver Mountain Logistics Inc.", "address": "4600 East 48th Avenue, Denver, CO 80216", "country": "USA"}
    ),
    (
        {"name": "Metro Atlanta Financial Group LLC", "address": "3344 Peachtree Road NE, Suite 800, Atlanta, GA 30326", "country": "USA"},
        {"name": "Metro Atlanta Real Estate Advisors LLC", "address": "1100 Peachtree Street NE, Suite 1400, Atlanta, GA 30309", "country": "USA"}
    ),

    # -------------------------------------------------------------
    # 3. Same Building / Tech Park Collisions (15 pairs = 30 businesses)
    # -------------------------------------------------------------
    (
        {"name": "Quantum Labs Cyber Security Pvt Ltd", "address": "Unit 402, 4th Floor, Block A, DLF Cyber City, Gurugram, Haryana 122002", "country": "India"},
        {"name": "NextGen FinTech Solutions Private Limited", "address": "Unit 403, 4th Floor, Block A, DLF Cyber City, Gurugram, Haryana 122002", "country": "India"}
    ),
    (
        {"name": "Prestige Wealth Management Limited", "address": "Level 5, Tower B, Prestige Tech Park, Outer Ring Road, Bengaluru, Karnataka 560103", "country": "India"},
        {"name": "CloudCore Systems Private Limited", "address": "Level 6, Tower B, Prestige Tech Park, Outer Ring Road, Bengaluru, Karnataka 560103", "country": "India"}
    ),
    (
        {"name": "GreenLeaf Organics Private Limited", "address": "Plot 45, Street No. 3, MIDC Industrial Area, Andheri East, Mumbai, Maharashtra 400093", "country": "India"},
        {"name": "Sterling Tooling Works Limited", "address": "Plot 46, Street No. 3, MIDC Industrial Area, Andheri East, Mumbai, Maharashtra 400093", "country": "India"}
    ),
    (
        {"name": "Infotech Global BPO Private Limited", "address": "Module 14, 3rd Floor, Tidel Park, Rajiv Gandhi Salai, Taramani, Chennai, Tamil Nadu 600113", "country": "India"},
        {"name": "SecurePay Systems Private Limited", "address": "Module 15, 3rd Floor, Tidel Park, Rajiv Gandhi Salai, Taramani, Chennai, Tamil Nadu 600113", "country": "India"}
    ),
    (
        {"name": "CyberNet Defense Labs Private Limited", "address": "Suite 801, 8th Floor, Manjeera Trinity Corporate, JNTU Road, Kukatpally, Hyderabad, Telangana 500072", "country": "India"},
        {"name": "ProData Analytics India Pvt Ltd", "address": "Suite 802, 8th Floor, Manjeera Trinity Corporate, JNTU Road, Kukatpally, Hyderabad, Telangana 500072", "country": "India"}
    ),
    (
        {"name": "Aura Cosmetics & Health Private Limited", "address": "Gala 101, First Floor, Marathon Innova, Ganpatrao Kadam Marg, Lower Parel, Mumbai, Maharashtra 400013", "country": "India"},
        {"name": "BlueStone Capital Advisors Pvt Ltd", "address": "Gala 102, First Floor, Marathon Innova, Ganpatrao Kadam Marg, Lower Parel, Mumbai, Maharashtra 400013", "country": "India"}
    ),
    (
        {"name": "Cognitive Cloud AI Solutions Pvt Ltd", "address": "Tower 3, Level 2, Magarpatta Cybercity, Hadapsar, Pune, Maharashtra 411028", "country": "India"},
        {"name": "Swift Logistics Express Private Limited", "address": "Tower 3, Level 3, Magarpatta Cybercity, Hadapsar, Pune, Maharashtra 411028", "country": "India"}
    ),
    (
        {"name": "OmniHealth Diagnostic Center Inc.", "address": "Medical Arts Pavilion, 200 Medical Parkway, Suite 101, Austin, TX 78738", "country": "USA"},
        {"name": "Austin Spine & Orthopedics LLC", "address": "Medical Arts Pavilion, 200 Medical Parkway, Suite 102, Austin, TX 78738", "country": "USA"}
    ),
    (
        {"name": "Beacon Digital Media LLC", "address": "750 Lexington Avenue, 14th Floor, New York, NY 10022", "country": "USA"},
        {"name": "Vanguard Strategic Capital LLC", "address": "750 Lexington Avenue, 18th Floor, New York, NY 10022", "country": "USA"}
    ),
    (
        {"name": "BlueWave Energy Partners Inc.", "address": "1200 17th Street, Suite 700, Denver, CO 80202", "country": "USA"},
        {"name": "HighPlains Natural Resources Inc.", "address": "1200 17th Street, Suite 900, Denver, CO 80202", "country": "USA"}
    ),
    (
        {"name": "BayArea BioGenetics Inc.", "address": "Mission Bay Science Center, 450 16th Street, Suite 300, San Francisco, CA 94158", "country": "USA"},
        {"name": "Cellular Dynamics Laboratories Inc.", "address": "Mission Bay Science Center, 450 16th Street, Suite 400, San Francisco, CA 94158", "country": "USA"}
    ),
    (
        {"name": "Midwest Agricultural Traders LLC", "address": "Board of Trade Building, 141 West Jackson Blvd, Suite 1200, Chicago, IL 60604", "country": "USA"},
        {"name": "Apex Commodity Derivatives LLC", "address": "Board of Trade Building, 141 West Jackson Blvd, Suite 1400, Chicago, IL 60604", "country": "USA"}
    ),
    (
        {"name": "Seattle Cloud Analytics Corp.", "address": "Columbia Center, 701 Fifth Avenue, Suite 4200, Seattle, WA 98104", "country": "USA"},
        {"name": "Cascade Venture Partners LLC", "address": "Columbia Center, 701 Fifth Avenue, Suite 4800, Seattle, WA 98104", "country": "USA"}
    ),
    (
        {"name": "Dallas Commercial Real Estate Group Inc.", "address": "Trammell Crow Center, 2001 Ross Avenue, Suite 2100, Dallas, TX 75201", "country": "USA"},
        {"name": "Lone Star Asset Advisory Corp.", "address": "Trammell Crow Center, 2001 Ross Avenue, Suite 2800, Dallas, TX 75201", "country": "USA"}
    ),
    (
        {"name": "Boston Quantum Technologies Inc.", "address": "One Financial Center, Suite 3100, Boston, MA 02111", "country": "USA"},
        {"name": "Atlantic Marine Underwriters LLC", "address": "One Financial Center, Suite 3700, Boston, MA 02111", "country": "USA"}
    ),

    # -------------------------------------------------------------
    # 4. Conglomerate Sister & Entity Collisions (15 pairs = 30 businesses)
    # -------------------------------------------------------------
    (
        {"name": "Tata Consumer Products Limited", "address": "11/13 Botawala Building, Horniman Circle, Fort, Mumbai, Maharashtra 400001", "country": "India"},
        {"name": "Tata Motors Commercial Vehicles Limited", "address": "Bombay House, 24 Homi Mody Street, Fort, Mumbai, Maharashtra 400001", "country": "India"}
    ),
    (
        {"name": "Reliance Retail Limited", "address": "Court House, 3rd Floor, Lokmanya Tilak Marg, Dhobi Talao, Mumbai, Maharashtra 400002", "country": "India"},
        {"name": "Reliance Jio Infocomm Limited", "address": "Reliance Corporate Park, Thane-Belapur Road, Ghansoli, Navi Mumbai, Maharashtra 400701", "country": "India"}
    ),
    (
        {"name": "Godrej Properties Limited", "address": "Godrej One, 5th Floor, Pirojshanagar, Eastern Express Highway, Vikhroli East, Mumbai, Maharashtra 400079", "country": "India"},
        {"name": "Godrej Agrovet Limited", "address": "Godrej One, 3rd Floor, Pirojshanagar, Eastern Express Highway, Vikhroli East, Mumbai, Maharashtra 400079", "country": "India"}
    ),
    (
        {"name": "Mahindra Lifespace Developers Limited", "address": "Mahindra Towers, 5th Floor, Road No. 13, Worli, Mumbai, Maharashtra 400018", "country": "India"},
        {"name": "Mahindra Logistics Limited", "address": "Arena Space, 10th Floor, Behind Majas Bus Depot, Jogeshwari East, Mumbai, Maharashtra 400060", "country": "India"}
    ),
    (
        {"name": "Adani Green Energy Limited", "address": "Adani Corporate House, Shantigram, Near Vaishno Devi Circle, SG Highway, Ahmedabad, Gujarat 382421", "country": "India"},
        {"name": "Adani Ports and Special Economic Zone Limited", "address": "Adani House, Mithakhali Six Roads, Navrangpura, Ahmedabad, Gujarat 380009", "country": "India"}
    ),
    (
        {"name": "Larsen & Toubro Infotech Limited", "address": "L&T House, Ballard Estate, Mumbai, Maharashtra 400001", "country": "India"},
        {"name": "L&T Finance Holdings Limited", "address": "Brindavan, Plot No. 177, CST Road, Kalina, Santacruz East, Mumbai, Maharashtra 400098", "country": "India"}
    ),
    (
        {"name": "Bajaj Finserv Limited", "address": "Bajaj Finserv Corporate Office, Off Pune-Ahmednagar Road, Viman Nagar, Pune, Maharashtra 411014", "country": "India"},
        {"name": "Bajaj Electricals Limited", "address": "45/47 Veer Nariman Road, Fort, Mumbai, Maharashtra 400001", "country": "India"}
    ),
    (
        {"name": "JPMorgan Chase Bank NA", "address": "383 Madison Avenue, New York, NY 10179", "country": "USA"},
        {"name": "JPMorgan Securities LLC", "address": "277 Park Avenue, New York, NY 10172", "country": "USA"}
    ),
    (
        {"name": "Morgan Stanley Wealth Management LLC", "address": "1585 Broadway, New York, NY 10036", "country": "USA"},
        {"name": "Morgan Stanley Investment Management Inc.", "address": "522 Fifth Avenue, New York, NY 10036", "country": "USA"}
    ),
    (
        {"name": "Johnson & Johnson Consumer Inc.", "address": "199 Grandview Road, Skillman, NJ 08558", "country": "USA"},
        {"name": "Johnson & Johnson Health Care Systems Inc.", "address": "425 Hoes Lane, Piscataway, NJ 08854", "country": "USA"}
    ),
    (
        {"name": "Pfizer International LLC", "address": "66 Hudson Boulevard East, New York, NY 10001", "country": "USA"},
        {"name": "Pfizer BioTherapeutics Inc.", "address": "10555 Science Center Drive, San Diego, CA 92121", "country": "USA"}
    ),
    (
        {"name": "General Electric Capital Corporation", "address": "901 Main Avenue, Norwalk, CT 06851", "country": "USA"},
        {"name": "GE Aerospace Systems Inc.", "address": "One Neumann Way, Cincinnati, OH 45215", "country": "USA"}
    ),
    (
        {"name": "Amazon Web Services Inc.", "address": "410 Terry Avenue North, Seattle, WA 98109", "country": "USA"},
        {"name": "Amazon Logistics Inc.", "address": "500 9th Avenue North, Seattle, WA 98109", "country": "USA"}
    ),
    (
        {"name": "ExxonMobil Chemical Company", "address": "22777 Springwoods Village Parkway, Spring, TX 77389", "country": "USA"},
        {"name": "ExxonMobil Pipeline Company", "address": "800 Bell Street, Houston, TX 77002", "country": "USA"}
    ),
    (
        {"name": "Chevron U.S.A. Inc.", "address": "6001 Bollinger Canyon Road, San Ramon, CA 94583", "country": "USA"},
        {"name": "Chevron Phillips Chemical Company LP", "address": "10001 Six Pines Drive, The Woodlands, TX 77380", "country": "USA"}
    ),
]


# =====================================================================
# 2. RICH PROCEDURAL BUILDERS FOR 880 DIVERSE BUSINESSES
# =====================================================================

INDUSTRY_DOMAINS = [
    # (Industry Category, Name Keywords, Suffix Style)
    ("Information Technology", ["Software Systems", "Data Cloud", "Digital Labs", "Tech Platforms", "Infotech Solutions", "Enterprise Computing"], "tech"),
    ("Cybersecurity", ["Cyber Defense", "InfoSec Labs", "SecureNet", "Threat Shield", "Sentinel Technologies", "CyberVault"], "tech"),
    ("Cloud Services", ["CloudScale", "Skyline Cloud", "CloudGrid Systems", "Virtual Networks", "CloudStack Solutions"], "tech"),
    ("Telecommunications", ["Telecom Networks", "BroadBand Systems", "Fiber Optics", "Wireless Connect", "Cellular Communications"], "telecom"),
    ("Banking", ["Bank Limited", "Commercial Bank", "Merchant Bank", "Trust & Banking", "Universal Bank"], "bank"),
    ("Insurance", ["Life Assurance", "General Insurance", "Risk Underwriters", "Mutual Assurance", "Reinsurance Group"], "insurance"),
    ("FinTech", ["PayTech Solutions", "Capital Flow", "Digital Lending", "WealthTech Labs", "SmartPay Solutions", "FinStack"], "fintech"),
    ("Healthcare", ["Healthcare Services", "Medical Care", "Integrated Health", "CareFirst Systems", "Wellness Health"], "health"),
    ("Hospitals", ["Memorial Hospital", "Super Specialty Hospital", "Healthcare Pavilion", "City General Hospital", "Metro Healthcare"], "health"),
    ("Pharmacies & Clinics", ["Care Pharmacy", "Diagnostics & Clinic", "MedLife Pharmacy", "Family Health Clinic", "Pathology Labs"], "health"),
    ("Manufacturing", ["Precision Engineering", "Industrial Tools", "Heavy Fabrication", "Manufacturing Works", "Casting & Forging"], "mfg"),
    ("Automobile", ["Motors Limited", "Automotive Systems", "Auto Components", "Mobility Solutions", "Electric Drives"], "auto"),
    ("Electronics", ["Semiconductors", "Circuit Systems", "Electronic Components", "Micro Devices", "OptoElectronics"], "tech"),
    ("Logistics", ["Logistics Solutions", "Express Cargo", "Supply Chain Services", "Freight Forwarders", "Global Trans"], "logistics"),
    ("Transportation", ["Interstate Transport", "Fleet Solutions", "Transit Systems", "Overland Express", "Roadways"], "logistics"),
    ("Warehousing", ["Cold Storage Systems", "Integrated Warehousing", "Logistics Hub", "Depot & Storage", "Fulfillment Centers"], "logistics"),
    ("Retail", ["Retail Enterprises", "Value Mart", "Super Stores", "Consumer Trends", "HyperMarket"], "retail"),
    ("Fashion", ["Apparel Works", "Garment Exports", "Textile Mills", "Fashion Trends", "Couture Design"], "fashion"),
    ("Restaurants", ["Bistro & Kitchen", "Gourmet Foods", "Hospitality Dining", "Brew Works", "Culinary House"], "food"),
    ("Food Processing", ["Agro Foods", "Dairy Products", "Organic Organics", "Beverages & Drinks", "Flour & Grain Mills"], "food"),
    ("Construction", ["Infrastructure Projects", "Constructions", "Civil Engineering", "Bridge & Highway Works", "Heavy Projects"], "infra"),
    ("Real Estate", ["Realty Developers", "Properties Limited", "Landmark Estates", "Commercial Towers", "Urban Housing"], "realestate"),
    ("Education", ["Global Academy", "International Learning", "EdTech Labs", "Educational Foundation", "Skills University"], "edu"),
    ("Consulting", ["Management Consulting", "Strategic Advisory", "Business Partners", "Enterprise Advisors", "Growth Strategies"], "consulting"),
    ("Legal Services", ["Legal Associates", "Law Chambers", "Attorneys & Partners", "Corporate Counsel", "Legal Consulting"], "legal"),
    ("Accounting", ["Tax & Audit Advisory", "Accounting Services", "Chartered Accountants", "Financial Consult", "Bookkeeping Solutions"], "consulting"),
    ("Marketing", ["Digital Marketing", "Interactive Media", "Brand Strategies", "Creative Communications", "Growth Labs"], "media"),
    ("Media", ["Broadcasting Corp", "Entertainment Group", "Publishing Media", "Digital Studio", "Motion Pictures"], "media"),
    ("Energy", ["Power Generation", "Energy Utilities", "Thermal Energy", "Grid Solutions", "Energy Resources"], "energy"),
    ("Renewable Energy", ["Solar Power", "Wind Energy Systems", "CleanTech Solutions", "Renewables Corp", "Green Energy"], "energy"),
    ("Agriculture", ["Agro Sciences", "Bio Crops", "Plantations", "Fertilizers & Chemicals", "Seed Technologies"], "agro"),
    ("Pharmaceuticals", ["Pharmaceuticals", "Laboratories Limited", "Life Sciences", "Drug Research", "Therapeutics"], "pharma"),
    ("Travel", ["Travels & Tours", "World Getaways", "Holidays International", "Tourism Services", "Voyage Planners"], "travel"),
    ("Hospitality", ["Suites & Hotels", "Resorts International", "Grand Hotels", "Heritage Hospitality", "Inn & Suites"], "hospitality"),
]

NAME_PREFIXES_INDIA = [
    "Aarav", "Aadya", "Akash", "Aditya", "Alok", "Amba", "Amrit", "Anand", "Ananta", "Ananya",
    "Apex", "Aristo", "Arya", "Ashok", "Asian", "Atul", "Avani", "Balaji", "Bharat", "Bhaskar",
    "Bhavani", "Bright", "Cadence", "Chaitanya", "Crest", "Crown", "Daksha", "Deepak", "Delta",
    "Devi", "Dhan", "Dolphin", "Durga", "Eagle", "Elite", "Everest", "Galaxy", "Ganesh", "Ganga",
    "Garuda", "Gayatri", "Globe", "Gopal", "Green", "Hari", "Himachal", "Himalaya", "Hind", "Horizon",
    "Imperial", "Indus", "Jagannath", "Jai", "Jal", "Kalyan", "Kanha", "Kaveri", "Kavach", "Krishna",
    "Kuber", "Lakshmi", "Lotus", "Mahavir", "Matrix", "Maya", "Meera", "Modern", "Mukesh", "Narmada",
    "National", "Navkar", "Neelam", "Netra", "Nexus", "Nivaan", "Om", "Omega", "Padma", "Pankaj",
    "Paramount", "Pawan", "Pinnacle", "Prabhat", "Prakash", "Pratham", "Premier", "Prime", "Quantum",
    "Radhe", "Rajan", "Rajdhani", "Ramesh", "Ratna", "Riddhi", "Rohan", "Sagar", "Sai", "Samarth",
    "Sanatan", "Saraswati", "Satyam", "Shakti", "Shanti", "Shiva", "Shree", "Shubham", "Siddhivinayak",
    "Skyline", "Smart", "Solaris", "Somnath", "Sterling", "Summit", "Sundar", "Surya", "Suvidha",
    "Swastik", "Tarun", "Tejas", "Titan", "Trishul", "Uday", "Unique", "Utkarsh", "Vaibhav", "Vandana",
    "Vanguard", "Varun", "Vayu", "Vedanta", "Velocity", "Venkateshwara", "Venus", "Veritas", "Vertex",
    "Vibrant", "Vidya", "Vijay", "Vikas", "Vinayak", "Vishnu", "Vishwakarma", "Vistara", "Vital",
    "Vivaan", "Yamuna", "Yash", "Zenith", "Zodiac"
]

NAME_PREFIXES_USA = [
    "Acme", "Advantage", "Allied", "Alpine", "Alta", "American", "Anchor", "Apollo", "Archer",
    "Aspen", "Atlantic", "Atlas", "Aurora", "Avalon", "Bancroft", "Beacon", "Belmont", "Benchmark",
    "Berkshire", "Bluegrass", "BlueRidge", "BlueStone", "BlueWave", "Bowman", "Bridgeport", "Brighton",
    "Broadstone", "Brookfield", "Caliber", "Cambridge", "Canyon", "Capitol", "Cardinal", "Cascade",
    "Cedar", "Centennial", "Charter", "Chesapeake", "Clarion", "Coastal", "Colonial", "Columbia",
    "Commonwealth", "Compass", "Concord", "Cornerstone", "Covenant", "Crescent", "Crown", "Cumberland",
    "Crestview", "Crossroads", "Dakota", "Delta", "Discovery", "Dominion", "Eagle", "Echo", "EdgeWater",
    "Emerald", "Empire", "Endeavor", "Ensemble", "Equinox", "Evergreen", "Falcon", "FirstNational",
    "Frontier", "Gateway", "Gemini", "Glacier", "GoldCoast", "Granite", "GreatLakes", "Greenwood",
    "GulfStream", "Halcyon", "Harbor", "Heritage", "Highland", "HighPlains", "Horizon", "Hudson",
    "Independence", "Infinity", "Ironclad", "Keystone", "Lakeside", "Liberty", "LoneStar", "Magnolia",
    "Maple", "Matrix", "Maverick", "Meridian", "Midwest", "Mission", "Monarch", "MountainView",
    "National", "NewEngland", "Nexus", "NorthStar", "Oakmont", "OldDominion", "Olympus", "Omni",
    "Pacific", "Palmetto", "Paramount", "Parkway", "Patriot", "PebbleBeach", "Peninsula", "Phoenix",
    "Piedmont", "Pinnacle", "Pioneer", "Plymouth", "Prairie", "Premier", "Prosperity", "Providence",
    "Quaker", "Redwood", "Regal", "Reliant", "Ridgeline", "Riverbend", "Rockwell", "Rosewood",
    "Safeguard", "Seaboard", "Sierra", "SilverState", "Skyline", "Sonoma", "Soundview", "SouthernCross",
    "Southland", "Spectrum", "Springfield", "Starboard", "Sterling", "Stonegate", "Summit", "Sunbelt",
    "Sunset", "Superior", "Tahoma", "Tidewater", "Tradewinds", "Trailblazer", "Trillium", "Trinity",
    "TrueNorth", "Union", "United", "Universal", "Valiant", "Valley", "Valor", "Vanguard",
    "Venture", "Veritas", "Victory", "Voyager", "Wasatch", "Westfield", "Westlake", "Windward",
    "Yellowstone", "Zenith"
]

INDIAN_ADDRESS_TEMPLATES = [
    # (Door/Plot, Building, Street, City, State, PIN)
    ("Plot {num}", "{bldg}", "{street}", "Mumbai", "Maharashtra", "400{pin:03d}"),
    ("Unit {num}, {fl} Floor", "{bldg}", "{street}", "Bengaluru", "Karnataka", "560{pin:03d}"),
    ("Gala No. {num}", "{bldg}", "{street}", "Pune", "Maharashtra", "411{pin:03d}"),
    ("Survey No. {num}/A", "{bldg}", "{street}", "Hyderabad", "Telangana", "500{pin:03d}"),
    ("Plot No. {num}, Phase {ph}", "{bldg}", "{street}", "Gurugram", "Haryana", "122{pin:03d}"),
    ("Shop No. {num}, Ground Floor", "{bldg}", "{street}", "Mumbai", "Maharashtra", "400{pin:03d}"),
    ("Office {num}, Level {fl}", "{bldg}", "{street}", "Noida", "Uttar Pradesh", "201{pin:03d}"),
    ("Building {num}, Tower {twr}", "{bldg}", "{street}", "Chennai", "Tamil Nadu", "600{pin:03d}"),
    ("Shed No. {num}, Industrial Estate", "{bldg}", "{street}", "Ahmedabad", "Gujarat", "380{pin:03d}"),
    ("Suite {num}, 2nd Floor", "{bldg}", "{street}", "Kolkata", "West Bengal", "700{pin:03d}"),
]

INDIAN_BUILDINGS = [
    "DLF Cyber City", "Prestige Tech Park", "RMZ Ecospace", "Manyata Embassy Tech Park",
    "Mindspace Cybercity", "EON Free Zone", "International Tech Park", "Bagmane World Tech Center",
    "Wagle Industrial Estate", "MIDC Industrial Area", "Electronic City", "Cyber Gateway",
    "Tidel Park", "Salt Lake Sector V", "Pioneer Urban Square", "Godrej One", "Solitaire Business Hub",
    "Infocity Commercial Complex", "Marathon Futurex", "Peninsula Corporate Park", "Lodha Supremus"
]

INDIAN_STREETS = [
    "Outer Ring Road", "Senapati Bapat Marg", "Western Express Highway", "Hosur Road",
    "Mahatma Gandhi Road", "Nelson Mandela Road", "CST Road, Kalina", "Banjara Hills Road No. 3",
    "Hinjawadi Infotech Park Road", "OMR IT Expressway", "Park Street", "Ashram Road",
    "Golf Course Road", "Sahar Airport Road", "Marathahalli-Sarjapur Road", "Gachibowli Main Road",
    "Udyog Vihar Phase IV", "Chowringhee Road", "SG Highway", "Veer Nariman Road"
]

USA_STREET_NAMES = [
    "Lexington Avenue", "Madison Avenue", "Broadway", "Fifth Avenue", "Market Street",
    "Pine Street", "California Street", "Mission Street", "North Michigan Avenue", "West Wacker Drive",
    "North LaSalle Street", "Congress Avenue", "Colorado Street", "Terry Avenue North", "Second Avenue",
    "Third Avenue", "Peachtree Road NE", "Paces Ferry Road NW", "17th Street", "Lawrence Street",
    "Stemmons Freeway", "Akard Street", "Boylston Street", "Summer Street", "Financial Center Parkway",
    "San Tomas Expressway", "Tasman Drive", "Mission College Boulevard", "Oracle Way", "Hudson Boulevard"
]

USA_CITIES = [
    ("New York", "NY", "100{pin:02d}"),
    ("San Francisco", "CA", "941{pin:02d}"),
    ("Chicago", "IL", "606{pin:02d}"),
    ("Austin", "TX", "787{pin:02d}"),
    ("Seattle", "WA", "981{pin:02d}"),
    ("Boston", "MA", "021{pin:02d}"),
    ("Atlanta", "GA", "303{pin:02d}"),
    ("Denver", "CO", "802{pin:02d}"),
    ("Dallas", "TX", "752{pin:02d}"),
    ("San Jose", "CA", "951{pin:02d}")
]


def clean_tsv_value(val: str) -> str:
    """Ensure value has no tabs or newlines which would break TSV format."""
    return val.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def build_procedural_businesses(rng: random.Random) -> list[dict]:
    """
    Builds exactly 1,000 unique, realistic Source 1 businesses:
    - 120 businesses from 60 explicit Hard Negative pairs (60 India, 60 USA)
    - 440 additional unique India businesses
    - 440 additional unique USA businesses
    Total: Exactly 1,000 businesses (500 India, 500 USA)
    """
    businesses = []
    used_names = set()
    used_addrs = set()

    # 1. Add Hard Negative pairs (60 pairs = 120 businesses)
    for b1, b2 in HARD_NEGATIVE_PAIRS:
        for b in (b1, b2):
            b_clean = {
                "name": clean_tsv_value(b["name"]),
                "address": clean_tsv_value(b["address"]),
                "country": b["country"]
            }
            businesses.append(b_clean)
            used_names.add(b_clean["name"].lower())
            used_addrs.add(b_clean["address"].lower())

    # Count of India and USA so far from hard negatives
    current_india = sum(1 for b in businesses if b["country"] == "India")  # 60
    current_usa = sum(1 for b in businesses if b["country"] == "USA")      # 60

    needed_india = 500 - current_india  # 440
    needed_usa = 500 - current_usa      # 440

    # 2. Generate 440 India Businesses
    india_legal_suffixes = [
        "Private Limited", "Private Limited", "Limited", "Pvt Ltd", "Pvt. Ltd.", "Limited"
    ]
    gen_i = 0
    p_idx = 0
    d_idx = 0
    while gen_i < needed_india:
        prefix = NAME_PREFIXES_INDIA[p_idx % len(NAME_PREFIXES_INDIA)]
        domain_cat, keywords, style = INDUSTRY_DOMAINS[d_idx % len(INDUSTRY_DOMAINS)]
        keyword = rng.choice(keywords)
        suffix = rng.choice(india_legal_suffixes)

        # Assemble business name
        name_choice = rng.randint(1, 4)
        if name_choice == 1:
            name = f"{prefix} {keyword} {suffix}"
        elif name_choice == 2:
            name = f"{prefix} India {keyword} {suffix}"
        elif name_choice == 3:
            name = f"{prefix} Global {keyword} {suffix}"
        else:
            name = f"{prefix} {keyword} Enterprises {suffix}"

        name = re.sub(r"\s+", " ", name).strip()
        # Ensure name doesn't duplicate suffix
        if name.endswith("Limited Limited"):
            name = name.replace("Limited Limited", "Limited")

        # Assemble realistic address
        tpl = rng.choice(INDIAN_ADDRESS_TEMPLATES)
        bldg = rng.choice(INDIAN_BUILDINGS)
        street = rng.choice(INDIAN_STREETS)
        door, b_val, s_val, city, state, pin_pat = tpl
        door_str = door.format(num=rng.randint(1, 150), fl=rng.randint(1, 14), ph=rng.randint(1, 4), twr=rng.choice(["A", "B", "C", "1", "2"]))
        pin_str = pin_pat.format(pin=rng.randint(10, 99))
        addr = f"{door_str}, {bldg}, {street}, {city}, {state} {pin_str}"
        addr = clean_tsv_value(re.sub(r"\s+", " ", addr))

        name_key = name.lower()
        addr_key = addr.lower()
        if name_key not in used_names and addr_key not in used_addrs:
            used_names.add(name_key)
            used_addrs.add(addr_key)
            businesses.append({
                "name": clean_tsv_value(name),
                "address": addr,
                "country": "India"
            })
            gen_i += 1

        p_idx += 1
        if p_idx % len(NAME_PREFIXES_INDIA) == 0:
            d_idx += 1

    # 3. Generate 440 USA Businesses
    usa_legal_suffixes = [
        "Inc.", "Corporation", "LLC", "Co.", "Incorporated", "LP", "LLC"
    ]
    gen_u = 0
    p_idx = 0
    d_idx = 0
    while gen_u < needed_usa:
        prefix = NAME_PREFIXES_USA[p_idx % len(NAME_PREFIXES_USA)]
        domain_cat, keywords, style = INDUSTRY_DOMAINS[d_idx % len(INDUSTRY_DOMAINS)]
        keyword = rng.choice(keywords)
        suffix = rng.choice(usa_legal_suffixes)

        name_choice = rng.randint(1, 4)
        if name_choice == 1:
            name = f"{prefix} {keyword} {suffix}"
        elif name_choice == 2:
            name = f"{prefix} National {keyword} {suffix}"
        elif name_choice == 3:
            name = f"{prefix} American {keyword} {suffix}"
        else:
            name = f"{prefix} {keyword} Group {suffix}"

        name = re.sub(r"\s+", " ", name).strip()

        # Assemble realistic US address
        street_no = rng.randint(10, 3999)
        street_name = rng.choice(USA_STREET_NAMES)
        city, state, pin_pat = rng.choice(USA_CITIES)
        zip_str = pin_pat.format(pin=rng.randint(10, 99))
        ste_type = rng.choice(["Suite", "Suite", "Floor", "#", "Unit"])
        if ste_type == "Floor":
            ste_str = f"{rng.randint(2, 45)}th Floor"
        else:
            ste_str = f"{ste_type} {rng.randint(100, 4800)}"

        addr = f"{street_no} {street_name}, {ste_str}, {city}, {state} {zip_str}"
        addr = clean_tsv_value(re.sub(r"\s+", " ", addr))

        name_key = name.lower()
        addr_key = addr.lower()
        if name_key not in used_names and addr_key not in used_addrs:
            used_names.add(name_key)
            used_addrs.add(addr_key)
            businesses.append({
                "name": clean_tsv_value(name),
                "address": addr,
                "country": "USA"
            })
            gen_u += 1

        p_idx += 1
        if p_idx % len(NAME_PREFIXES_USA) == 0:
            d_idx += 1

    assert len(businesses) == 1000, f"Expected 1,000 businesses, got {len(businesses)}"
    return businesses


# =====================================================================
# 3. ADVANCED NOISE INJECTION ENGINES
# =====================================================================

def apply_casing_noise(text: str, rng: random.Random) -> str:
    """Simulates inconsistent casing (ALL CAPS, lowercase, Title Case)."""
    mode = rng.choice(["upper", "lower", "title", "preserve", "upper"])
    if mode == "upper":
        return text.upper()
    elif mode == "lower":
        return text.lower()
    elif mode == "title":
        return text.title()
    return text


def apply_legal_suffix_noise(name: str, country: str, rng: random.Random) -> str:
    """Variations in company legal suffixes."""
    s = name
    pvt_ltd_patterns = [
        (r"\bPrivate Limited\b", ["Pvt Ltd", "Pvt. Ltd.", "P. Ltd.", "Ltd", "Private Ltd.", "Pvt Limited", "P.LTD."]),
        (r"\bPvt\.? Ltd\.?\b", ["Private Limited", "Ltd.", "Ltd", "Pvt Limited", "P Ltd"]),
        (r"\bLimited\b", ["Ltd", "Ltd.", "LTD"]),
    ]
    for pattern, replacements in pvt_ltd_patterns:
        if re.search(pattern, s, flags=re.IGNORECASE):
            s = re.sub(pattern, rng.choice(replacements), s, count=1, flags=re.IGNORECASE)
            return s

    usa_patterns = [
        (r"\bInc\.\b|\bInc\b|\bIncorporated\b", ["Inc.", "Inc", "Incorporated", "Corp.", "Co."]),
        (r"\bCorporation\b|\bCorp\.\b|\bCorp\b", ["Corp", "Corp.", "Corporation", "Inc.", "Co."]),
        (r"\bLLC\b|\bL\.L\.C\.\b", ["L.L.C.", "LLC", "LLC.", "Limited Liability Company", "Ltd"]),
        (r"\bCompany\b", ["Co.", "Co", "Company"]),
    ]
    for pattern, replacements in usa_patterns:
        if re.search(pattern, s, flags=re.IGNORECASE):
            s = re.sub(pattern, rng.choice(replacements), s, count=1, flags=re.IGNORECASE)
            return s

    return s


def apply_road_street_noise(address: str, rng: random.Random) -> str:
    """Street component normalization & abbreviations."""
    s = address
    replacements = [
        (r"\bRoad\b", ["Rd", "Rd.", "RD"]),
        (r"\bStreet\b", ["St", "St.", "ST"]),
        (r"\bAvenue\b", ["Ave", "Ave.", "AVE"]),
        (r"\bBoulevard\b", ["Blvd", "Blvd."]),
        (r"\bHighway\b", ["Hwy", "Hwy."]),
        (r"\bFloor\b", ["Fl", "Fl.", "Flr", "Level"]),
        (r"\bSuite\b", ["Ste", "Ste.", "Unit", "#"]),
        (r"\bBuilding\b", ["Bldg", "Bldg."]),
        (r"\bTower\b", ["Twr", "Block"]),
        (r"\bExpressway\b", ["Expwy", "Express Way"]),
        (r"\bParkway\b", ["Pkwy", "Pkwy."]),
    ]
    rng.shuffle(replacements)
    applied = 0
    for pat, rep_list in replacements:
        if re.search(pat, s, flags=re.IGNORECASE):
            s = re.sub(pat, rng.choice(rep_list), s, count=1, flags=re.IGNORECASE)
            applied += 1
            if applied >= 2:
                break
    return s


def apply_spelling_and_typo_noise(text: str, rng: random.Random) -> str:
    """Simulates real human clerical typos & phonetic variations."""
    words = text.split()
    if not words:
        return text

    known_typos = {
        "Technology": "Technolgy",
        "Technologies": "Technologes",
        "Solutions": "Solutins",
        "Software": "Softwrae",
        "International": "Internationl",
        "Enterprises": "Enterprizes",
        "Global": "Globle",
        "Center": "Centre",
        "Express": "Expres",
        "National": "Natoinal",
        "Financial": "Finanical",
        "Systems": "Systmes",
        "Services": "Servics",
        "Healthcare": "Health Care",
        "Digital": "Digitel",
        "Logistics": "Logistcs",
        "Engineering": "Enginering",
        "Security": "Securty",
        "Precision": "Precison",
        "Energy": "Enrgy",
    }
    for i, w in enumerate(words):
        clean_w = re.sub(r"[^\w]", "", w)
        if clean_w in known_typos and rng.random() < 0.45:
            words[i] = w.replace(clean_w, known_typos[clean_w])
            return " ".join(words)

    long_indices = [i for i, w in enumerate(words) if len(re.sub(r"[^\w]", "", w)) >= 5]
    if long_indices and rng.random() < 0.5:
        target_idx = rng.choice(long_indices)
        word = words[target_idx]
        pos = rng.randint(1, len(word) - 2)
        if word[pos].isalpha() and word[pos + 1].isalpha():
            transposed = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
            words[target_idx] = transposed

    return " ".join(words)


def apply_abbreviation_noise(name: str, address: str, rng: random.Random) -> tuple[str, str]:
    """Common business word abbreviations."""
    name_abbrevs = [
        (r"\bTechnologies\b|\bTechnology\b", "Tech"),
        (r"\bInternational\b", "Intl"),
        (r"\bManufacturing\b", "Mfg"),
        (r"\bServices\b", "Svcs"),
        (r"\bManagement\b", "Mgmt"),
        (r"\bDepartment\b", "Dept"),
        (r"\bSolutions\b", "Solns"),
        (r"\bCommunications\b", "Comms"),
        (r"\bPharmaceuticals\b", "Pharma"),
    ]
    new_name = name
    for pat, rep in name_abbrevs:
        if rng.random() < 0.35 and re.search(pat, new_name, flags=re.IGNORECASE):
            new_name = re.sub(pat, rep, new_name, count=1, flags=re.IGNORECASE)

    addr_abbrevs = [
        (r"\bIndustrial Area\b", "Indl Area"),
        (r"\bSquare\b", "Sq."),
        (r"\bBasement\b", "Bsmt"),
        (r"\bSector\b", "Sec."),
        (r"\bPhase\b", "Ph."),
    ]
    new_addr = address
    for pat, rep in addr_abbrevs:
        if rng.random() < 0.35 and re.search(pat, new_addr, flags=re.IGNORECASE):
            new_addr = re.sub(pat, rep, new_addr, count=1, flags=re.IGNORECASE)

    return new_name, new_addr


def apply_punctuation_and_space_noise(text: str, rng: random.Random) -> str:
    """Inconsistent delimiters (comma/hyphen/slash) and whitespace."""
    s = text.replace("'", "").replace("’", "")
    punct_choice = rng.choice(["hyphen", "slash", "spaced_comma", "clean", "double_space"])
    if punct_choice == "hyphen":
        s = re.sub(r",\s*", " - ", s, count=rng.randint(1, 2))
    elif punct_choice == "slash":
        s = re.sub(r",\s*", " / ", s, count=1)
    elif punct_choice == "spaced_comma":
        s = re.sub(r",\s*", " , ", s, count=2)
    elif punct_choice == "double_space":
        words = s.split()
        if len(words) > 2:
            ins = rng.randint(1, len(words) - 1)
            words[ins] = "  " + words[ins]
            s = " ".join(words)
    return s


def apply_landmark_noise(address: str, country: str, rng: random.Random) -> str:
    """Injects landmarks common in dense urban zones and directory listings."""
    india_landmarks = [
        "Near Cyber Hub", "Opposite Metro Station", "Behind City Center Mall",
        "Near Railway Station", "Opposite Bus Stand", "Next to State Bank of India",
        "Behind Forum Mall", "Near Manyata Tech Park", "Opposite Inorbit Mall",
        "Near DLF Phase 2", "Beside Apollo Pharmacy", "Near Wipro Circle"
    ]
    usa_landmarks = [
        "Near Times Square", "Across from Central Park", "Next to Grand Central",
        "Adjacent to Metro Station", "Near Financial District", "Across from City Hall",
        "Near Millennium Park", "Next to Washington Square", "Near Embarcadero Plaza"
    ]
    landmarks = india_landmarks if country == "India" else usa_landmarks
    lm = rng.choice(landmarks)
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) >= 2:
        insert_pos = rng.randint(1, len(parts) - 1)
        parts.insert(insert_pos, lm)
        return ", ".join(parts)
    return f"{address}, {lm}"


def apply_dropped_address_components(address: str, country: str, rng: random.Random) -> str:
    """Simulates omitted PIN/ZIP codes, states, and unit numbers."""
    s = address
    if rng.random() < 0.65:
        s = re.sub(r"\b\d{6}\b", "", s)
        s = re.sub(r"\b\d{5}(-\d{4})?\b", "", s)
    if rng.random() < 0.4:
        s = re.sub(r"^(Plot\s+[A-Za-z0-9\-\/]+|Shop\s+No\.\s*\d+|Unit\s+\d+|Level\s+\d+|Suite\s+\d+|Gala\s+No\.\s*\d+|Shed\s+No\.\s*\d+|Office\s+\d+|[0-9A-Z\-\/]+,)\s*,?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r",\s*,+", ",", s)
    return re.sub(r"\s+", " ", s).strip(" ,-")


def apply_word_ordering_noise(address: str, rng: random.Random) -> str:
    """Swaps address components (e.g. locality before street)."""
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) >= 3:
        idx = rng.randint(0, len(parts) - 2)
        parts[idx], parts[idx + 1] = parts[idx + 1], parts[idx]
        return ", ".join(parts)
    return address


def apply_name_variation(name: str, rng: random.Random) -> str:
    """Subtle business name modifications (omitted 'The', 'India', acronyms)."""
    s = name
    if s.startswith("The ") and rng.random() < 0.5:
        return s[4:]
    if " India " in s and rng.random() < 0.4:
        return s.replace(" India ", " ")
    acronyms = {
        "Tata Consultancy Services Limited": "TCS Ltd",
        "International Business Machines": "IBM Corp",
        "Amazon.com Inc.": "Amazon",
        "Meta Platforms Inc.": "Meta",
        "One97 Communications Limited": "Paytm One97",
        "Amazon Web Services Inc.": "AWS Inc",
        "JPMorgan Chase Bank NA": "JPMorgan Chase",
    }
    if s in acronyms and rng.random() < 0.5:
        return acronyms[s]
    return s


def generate_source2_record(base: dict, rng: random.Random) -> dict:
    """Source 2 noise: ERP / CRM style (abbreviations, legal variations, typos, spacing)."""
    name = apply_name_variation(base["name"], rng)
    name = apply_legal_suffix_noise(name, base["country"], rng)
    addr = base["address"]
    name, addr = apply_abbreviation_noise(name, addr, rng)
    addr = apply_road_street_noise(addr, rng)
    if rng.random() < 0.45:
        addr = apply_dropped_address_components(addr, base["country"], rng)
    if rng.random() < 0.35:
        addr = apply_spelling_and_typo_noise(addr, rng)
    if rng.random() < 0.25:
        name = apply_spelling_and_typo_noise(name, rng)
    name = apply_punctuation_and_space_noise(name, rng)
    addr = apply_punctuation_and_space_noise(addr, rng)
    if rng.random() < 0.3:
        name = apply_casing_noise(name, rng)
        addr = apply_casing_noise(addr, rng)
    return {
        "business_name": clean_tsv_value(name),
        "business_address": clean_tsv_value(addr),
        "country": base["country"],
    }


def generate_source3_record(base: dict, rng: random.Random) -> dict:
    """Source 3 noise: Scraped / Web directory (lowercase, landmarks, reordering, partial addrs)."""
    name = base["name"]
    addr = base["address"]
    country = base["country"]
    if rng.random() < 0.5:
        addr = apply_landmark_noise(addr, country, rng)
    if rng.random() < 0.45:
        addr = apply_word_ordering_noise(addr, rng)
    addr = apply_road_street_noise(addr, rng)
    if rng.random() < 0.55:
        addr = apply_dropped_address_components(addr, country, rng)
    name = apply_legal_suffix_noise(name, country, rng)
    name, addr = apply_abbreviation_noise(name, addr, rng)
    name = apply_name_variation(name, rng)
    if rng.random() < 0.35:
        name = apply_spelling_and_typo_noise(name, rng)
    if rng.random() < 0.35:
        addr = apply_spelling_and_typo_noise(addr, rng)
    name = apply_punctuation_and_space_noise(name, rng)
    addr = apply_punctuation_and_space_noise(addr, rng)

    casing_mode = rng.choice(["upper", "lower", "title", "preserve", "lower"])
    if casing_mode == "upper":
        name = name.upper()
        addr = addr.upper()
    elif casing_mode == "lower":
        name = name.lower()
        addr = addr.lower()
    elif casing_mode == "title":
        name = name.title()
        addr = addr.title()

    return {
        "business_name": clean_tsv_value(name),
        "business_address": clean_tsv_value(addr),
        "country": country,
    }


# =====================================================================
# 4. DATASET ASSEMBLY & GROUND TRUTH GENERATION
# =====================================================================

def build_all_datasets(seed: int = RANDOM_SEED):
    """
    Constructs Source 1, Source 2, Source 3, and Ground Truth datasets.

    Sizes:
    - Source 1: Exactly 1,000 businesses (S1-00001 to S1-01000)
    - Source 2: ~800 records (S2-00001 ...)
    - Source 3: ~750 records (S3-00001 ...)

    Matching Design:
    - 700 matched entities (70.0% of Source 1):
      - 380 matched in BOTH Source 2 and Source 3
      - 180 matched in Source 2 ONLY
      - 140 matched in Source 3 ONLY
    - 300 singletons (30.0% of Source 1, matched_entity_ids = "")
    - Multi-record matches: 70 entities have 2 records in S2, 60 entities have 2 records in S3
    - Distractors: 170 unlinked in S2, 170 unlinked in S3 (many are hard negatives)
    """
    rng = random.Random(seed)

    base_catalog = build_procedural_businesses(rng)
    assert len(base_catalog) == 1000, f"Expected 1,000, got {len(base_catalog)}"

    # Source 1 clean records
    s1_records = []
    for idx, b in enumerate(base_catalog, start=1):
        s1_id = f"S1-{idx:05d}"
        s1_records.append({
            "entity_id": s1_id,
            "business_name": clean_tsv_value(b["name"]),
            "business_address": clean_tsv_value(b["address"]),
            "country": b["country"],
            "_base_idx": idx - 1,
        })

    # Separate India (500) and USA (500) indices for balanced partitions
    india_indices = [i for i, b in enumerate(base_catalog) if b["country"] == "India"]
    usa_indices = [i for i, b in enumerate(base_catalog) if b["country"] == "USA"]

    # Stratified partitions:
    # 380 both (190 India, 190 USA)
    # 180 s2_only (90 India, 90 USA)
    # 140 s3_only (70 India, 70 USA)
    # 300 singletons (150 India, 150 USA)
    both_indices = india_indices[:190] + usa_indices[:190]
    s2_only_indices = india_indices[190:280] + usa_indices[190:280]
    s3_only_indices = india_indices[280:350] + usa_indices[280:350]
    singleton_indices = india_indices[350:500] + usa_indices[350:500]

    assert len(both_indices) == 380
    assert len(s2_only_indices) == 180
    assert len(s3_only_indices) == 140
    assert len(singleton_indices) == 300
    assert len(both_indices) + len(s2_only_indices) + len(s3_only_indices) + len(singleton_indices) == 1000

    # Multi-match candidates (subset of matched entities with a 2nd record)
    multi_s2_candidates = both_indices[:45] + s2_only_indices[:25]  # 70 entities
    multi_s3_candidates = both_indices[45:85] + s3_only_indices[:20]  # 60 entities

    # Build raw records for Source 2
    raw_s2 = []
    # (A) Matches from both
    for b_idx in both_indices:
        rec = generate_source2_record(base_catalog[b_idx], rng)
        rec["_s1_id"] = s1_records[b_idx]["entity_id"]
        raw_s2.append(rec)
    # (B) Matches from S2 only
    for b_idx in s2_only_indices:
        rec = generate_source2_record(base_catalog[b_idx], rng)
        rec["_s1_id"] = s1_records[b_idx]["entity_id"]
        raw_s2.append(rec)
    # (C) 2nd duplicate/branch match records in S2
    for b_idx in multi_s2_candidates:
        rec = generate_source2_record(base_catalog[b_idx], rng)
        rec["_s1_id"] = s1_records[b_idx]["entity_id"]
        raw_s2.append(rec)

    # (D) Distractors in Source 2 (170 records, including hard negative distractors)
    # Generate 85 India + 85 USA distractor records
    for i in range(85):
        # Derive from an existing entity with modified name or adjacent address (Hard Negative distractor)
        ref_b = base_catalog[india_indices[i % len(india_indices)]]
        dist_name = ref_b["name"].replace("Private Limited", "Ventures Limited").replace("Limited", "Corporation") + " Distractor"
        dist_name = dist_name.replace("Distractor Distractor", "Distractor")
        dist_rec = generate_source2_record({
            "name": dist_name,
            "address": ref_b["address"],
            "country": "India"
        }, rng)
        dist_rec["_s1_id"] = None
        raw_s2.append(dist_rec)

    for i in range(85):
        ref_b = base_catalog[usa_indices[i % len(usa_indices)]]
        dist_name = ref_b["name"].replace("Inc.", "Holdings Corp.").replace("LLC", "Capital LLC") + " Distractor"
        dist_name = dist_name.replace("Distractor Distractor", "Distractor")
        dist_rec = generate_source2_record({
            "name": dist_name,
            "address": ref_b["address"],
            "country": "USA"
        }, rng)
        dist_rec["_s1_id"] = None
        raw_s2.append(dist_rec)

    # Shuffle raw_s2
    rng.shuffle(raw_s2)

    # Assign S2 IDs: S2-00001, S2-00002, ...
    s2_records = []
    s1_to_s2_map = {}
    for idx, item in enumerate(raw_s2, start=1):
        s2_id = f"S2-{idx:05d}"
        s2_records.append({
            "entity_id": s2_id,
            "business_name": item["business_name"],
            "business_address": item["business_address"],
            "country": item["country"],
        })
        if item["_s1_id"] is not None:
            s1_to_s2_map.setdefault(item["_s1_id"], []).append(s2_id)

    # Build raw records for Source 3
    raw_s3 = []
    # (A) Matches from both
    for b_idx in both_indices:
        rec = generate_source3_record(base_catalog[b_idx], rng)
        rec["_s1_id"] = s1_records[b_idx]["entity_id"]
        raw_s3.append(rec)
    # (B) Matches from S3 only
    for b_idx in s3_only_indices:
        rec = generate_source3_record(base_catalog[b_idx], rng)
        rec["_s1_id"] = s1_records[b_idx]["entity_id"]
        raw_s3.append(rec)
    # (C) 2nd duplicate/scraped match records in S3
    for b_idx in multi_s3_candidates:
        rec = generate_source3_record(base_catalog[b_idx], rng)
        rec["_s1_id"] = s1_records[b_idx]["entity_id"]
        raw_s3.append(rec)

    # (D) Distractors in Source 3 (170 records, including hard negative distractors)
    for i in range(85):
        ref_b = base_catalog[india_indices[(i + 40) % len(india_indices)]]
        dist_name = "Global " + ref_b["name"] + " Scraped"
        dist_rec = generate_source3_record({
            "name": dist_name,
            "address": ref_b["address"],
            "country": "India"
        }, rng)
        dist_rec["_s1_id"] = None
        raw_s3.append(dist_rec)

    for i in range(85):
        ref_b = base_catalog[usa_indices[(i + 40) % len(usa_indices)]]
        dist_name = "American " + ref_b["name"] + " Scraped"
        dist_rec = generate_source3_record({
            "name": dist_name,
            "address": ref_b["address"],
            "country": "USA"
        }, rng)
        dist_rec["_s1_id"] = None
        raw_s3.append(dist_rec)

    # Shuffle raw_s3
    rng.shuffle(raw_s3)

    # Assign S3 IDs: S3-00001, S3-00002, ...
    s3_records = []
    s1_to_s3_map = {}
    for idx, item in enumerate(raw_s3, start=1):
        s3_id = f"S3-{idx:05d}"
        s3_records.append({
            "entity_id": s3_id,
            "business_name": item["business_name"],
            "business_address": item["business_address"],
            "country": item["country"],
        })
        if item["_s1_id"] is not None:
            s1_to_s3_map.setdefault(item["_s1_id"], []).append(s3_id)

    # Build Ground Truth
    ground_truth_records = []
    for s1 in s1_records:
        s1_id = s1["entity_id"]
        matched_ids = []
        if s1_id in s1_to_s2_map:
            matched_ids.extend(s1_to_s2_map[s1_id])
        if s1_id in s1_to_s3_map:
            matched_ids.extend(s1_to_s3_map[s1_id])

        ground_truth_records.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_ids),
        })

    # Prepare DataFrames
    df_s1 = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in s1_records])
    df_s2 = pd.DataFrame(s2_records)
    df_s3 = pd.DataFrame(s3_records)
    df_gt = pd.DataFrame(ground_truth_records)

    return df_s1, df_s2, df_s3, df_gt


# =====================================================================
# 5. EXPORT AND VERIFICATION
# =====================================================================

def save_and_verify_datasets(output_dirs: list[Path]):
    """
    Saves datasets to TSV files and verifies integrity using pandas.
    """
    df_s1, df_s2, df_s3, df_gt = build_all_datasets(seed=RANDOM_SEED)

    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        s1_path = out_dir / "source1.tsv"
        s2_path = out_dir / "source2.tsv"
        s3_path = out_dir / "source3.tsv"
        gt_path = out_dir / "ground_truth.tsv"

        df_s1.to_csv(s1_path, sep="\t", index=False)
        df_s2.to_csv(s2_path, sep="\t", index=False)
        df_s3.to_csv(s3_path, sep="\t", index=False)
        df_gt.to_csv(gt_path, sep="\t", index=False)
        print(f"-> Saved TSV files successfully to: {out_dir}")

    # Primary verification
    primary_dir = output_dirs[0]
    s1_loaded = pd.read_csv(primary_dir / "source1.tsv", sep="\t", keep_default_na=False)
    s2_loaded = pd.read_csv(primary_dir / "source2.tsv", sep="\t", keep_default_na=False)
    s3_loaded = pd.read_csv(primary_dir / "source3.tsv", sep="\t", keep_default_na=False)
    gt_loaded = pd.read_csv(primary_dir / "ground_truth.tsv", sep="\t", keep_default_na=False)

    s2_ids = set(s2_loaded["entity_id"])
    s3_ids = set(s3_loaded["entity_id"])

    matching_s1_count = 0
    singleton_s1_count = 0
    both_count = 0
    s2_only_count = 0
    s3_only_count = 0
    total_matching_relationships = 0

    for _, row in gt_loaded.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]).strip()
        if not matched_str:
            singleton_s1_count += 1
            continue

        matching_s1_count += 1
        m_ids = [m.strip() for m in matched_str.split(",")]
        total_matching_relationships += len(m_ids)

        has_s2 = any(m.startswith("S2-") for m in m_ids)
        has_s3 = any(m.startswith("S3-") for m in m_ids)

        if has_s2 and has_s3:
            both_count += 1
        elif has_s2:
            s2_only_count += 1
        elif has_s3:
            s3_only_count += 1

        for mid in m_ids:
            if mid.startswith("S2-"):
                assert mid in s2_ids, f"Error: Ground truth ID {mid} not found in Source 2!"
            elif mid.startswith("S3-"):
                assert mid in s3_ids, f"Error: Ground truth ID {mid} not found in Source 3!"
            else:
                raise ValueError(f"Unexpected ID prefix in ground truth: {mid}")

    c_s1 = s1_loaded["country"].value_counts().to_dict()
    c_s2 = s2_loaded["country"].value_counts().to_dict()
    c_s3 = s3_loaded["country"].value_counts().to_dict()

    print("\n========================================")
    print("BUSINESS ENTITY RESOLUTION DATASET")
    print("==================================")
    print(f"\nSource 1 records: {len(s1_loaded)}")
    print(f"Source 2 records: {len(s2_loaded)}")
    print(f"Source 3 records: {len(s3_loaded)}")
    print(f"\nTotal records: {len(s1_loaded) + len(s2_loaded) + len(s3_loaded)}")
    print(f"\nMatching Source 1 entities: {matching_s1_count}")
    print(f"Singleton Source 1 entities: {singleton_s1_count}")
    print(f"\nMatches in BOTH: {both_count}")
    print(f"Matches in Source 2 ONLY: {s2_only_count}")
    print(f"Matches in Source 3 ONLY: {s3_only_count}")
    print(f"\nTotal matching relationships: {total_matching_relationships}")
    print("\nCountry distribution:")
    print(f"Source 1: {c_s1}")
    print(f"Source 2: {c_s2}")
    print(f"Source 3: {c_s3}")
    print("========================================\n")

    print("First 5 records from Source 1:")
    print(s1_loaded.head(5).to_string(index=False))
    print("\nFirst 5 records from Source 2:")
    print(s2_loaded.head(5).to_string(index=False))
    print("\nFirst 5 records from Source 3:")
    print(s3_loaded.head(5).to_string(index=False))
    print("\n")


def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    target_dirs = [
        project_root / "dataset",
    ]

    workspace_root = project_root.parent
    if (workspace_root / "dataset") != (project_root / "dataset"):
        target_dirs.append(workspace_root / "dataset")

    save_and_verify_datasets(target_dirs)


if __name__ == "__main__":
    main()
