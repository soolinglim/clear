import argparse
import math
import time
import random
import sys
import json
import pickle
import numpy as np
import pandas as pd
import re
import os

from collections import defaultdict

base_path = "clear/apartment_images/"
gt_path = 'ground_truth.csv'

import base64
import requests

# OpenAI API Key
api_key = "xxxxx"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

MAX_LLM_RETRY = 6

EVALUATION_FEATURE = 'WINDOW' #'HEATING' # 'KWH' # 'LIGHTING' #'AGE', 'WINDOW'

print('evoatt fixed length:', EVALUATION_FEATURE)

RESULTS_DIRECTORY = 'results'
IMAGE_DIRECTORY = 'image/'

image_format = 'png'

# constants
RANDOM_SEED: int
NUM_POPULATION: int
NUM_GENERATION: int

REAL_LLM = True

NUMBER_OF_PARENTS = 5

AGE_TRAINING_SET_ID = [0, 1, 2, 4, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 24, 27, 28, 30, 34, 35, 36, 37, 39, 41, 43, 44, 45, 46]
WINDOW_TRAINING_SET_ID = [0, 1, 4, 7, 9, 10, 11, 12, 13, 14, 17, 19, 21, 22, 24, 25, 28, 29, 30, 31, 33, 34, 35, 37, 39, 40, 42, 43]
LIGHTING_TRAINING_SET_ID = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 26, 28, 29, 30, 31, 32, 36, 39, 40, 43, 45, 46]
KWH_TRAINING_SET_ID = [0, 1, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 20, 21, 29, 33, 35, 36, 38, 39, 42, 43, 44]
HEATING_TRAINING_SET_ID = [1, 3, 4, 6, 8, 9, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 31, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43]

OPTION_BLURB = "Finally, select one of these options:"
KWH_ESTIMATION_BLURB = 'Finally, give an estimate of the kwh. A highly efficient apartment might have a kwh/m2 value as low as 35 or better. An inefficient apartment might have a kwh/m2 value as high as 450 or worse'
WINDOW_ESTIMATION_BLURB = 'Finally, give an estimate of the U-value between 0.5 and 4.8. A highly efficient window might have a U-value as low as 0.5. An highly inefficient window might have a U-value of 4.8. Always provide an estimate, even if you are unable to view or analyze the features in the images.'

AGE_FINAL_INSTRUCTIONS = LIGHTING_FINAL_INSTRUCTIONS = HEATING_FINAL_INSTRUCTIONS = 'You can only use one of these, do not modify or invent your own options. Put the selected option in between ### and ###'
KWH_FINAL_INSTRUCTIONS = 'Put the estimated kwh in between ### and ###. Do not include any other text apart from the kwh values'
WINDOW_FINAL_INSTRUCTIONS = 'Always produce an estimate. Put the estimated U-value in between ### and ###. Do not include any other text apart from the U-value'

AGE_PROMPT_QUESTION = "What is the age of this apartment?"
WINDOW_PROMPT_QUESTION = "What type of windows does this apartment have?"
LIGHTING_PROMPT_QUESTION = "What type of lighting does this apartment have?"
KWH_PROMPT_QUESTION = "Estimate the energy consumption in kwh per metre squared for the following apartment."
HEATING_PROMPT_QUESTION = "What type of heating does this apartment have?"

AGE_CHOICE = ['before 1900', '1900-1930', '1930-1950', '1950-1970', '1970-1990', '1990-2020', '2020-now']
AGE_PROMPT_OPTIONS = ', '.join(AGE_CHOICE) # turn age_choice into a string

# WINDOW_CHOICE = ['single glazed', 'double glazed', 'high efficiency double or triple glazed']
# WINDOW_PROMPT_OPTIONS =  ', '.join(f"({i+1}) {choice}" for i, choice in enumerate(WINDOW_CHOICE))

WINDOW_CHOICE = []
WINDOW_PROMPT_OPTIONS = ''

LIGHTING_CHOICE = ['no low energy lighting', 'low energy in 20%', 'low energy in 40%', 'low energy in 60%', 'low energy in 80%', 'low energy in 100%']
LIGHTING_PROMPT_OPTIONS = ', '.join(LIGHTING_CHOICE) 

KWH_CHOICE = []
KWH_PROMPT_OPTIONS = '' # kwh don't have options

HEATING_CHOICE = ['underfloor heating', 'water radiators', 'electric heaters', 'electric storage heaters', 'warm air from vents']
HEATING_PROMPT_OPTIONS = ', '.join(HEATING_CHOICE) 

AGE_FEATURES = [
    [  # Architectural & Structural Features
        "Brickwork style",
        "Roof type and materials",
        "Chimney presence and style",
        "Gable ends",
        "Facade symmetry",
        "Building ornamentation",
        "String courses (horizontal bands)",
        "Parapet walls",
        "Use of stucco or render",
        "Stone or brick quoins",
        "External cornice details",
        "Façade materials (e.g., concrete, glass)",
        "External cladding",
        "Steel-reinforced concrete structures",
        "Reinforced concrete structure",
        "Insulated external walls",
        "Double-height lobby space",
    ],
    [  # Windows & Doors
        "Window design and shape",
        "Arched windows",
        "Door style and material",
        "Proportions of windows to walls",
        "Mouldings around windows",
        "Skylights or dormers",
        "Large glass windows",
        "UPVC doors",
        "Thermally efficient radiators",
        "Energy-efficient windows",
        "Double glazing",
        "Sliding glass doors",
        "Floor-to-ceiling windows",
        "Contemporary door styles",
    ],
    [  # Internal Features
        "Interior fireplace design",
        "Skirting board design",
        "Ceiling height and details (e.g., coving)",
        "Architraves around doors",
        "Staircase design",
        "Internal door style",
        "Loft conversions or additions",
        "Built-in cabinetry style",
        "Modern partition walls",
    ],
    [  # Modern & Minimalistic Design Elements
        "Minimalistic interior design",
        "Clean, straight lines",
        "Simple geometric shapes",
        "Monochrome color palette",
        "Neutral color palette",
    ],
    [  # Technological & Efficiency Features
        "Smart home technology",
        "Thermally efficient radiators",
        "Solar panels",
        "Energy performance certificates",
        "Automated building management systems",
        "Presence of digital infrastructures",
        "Digital connectivity features",
        "Photovoltaic cells",
        "Energy-efficient lighting",
    ],
    [  # Utilities & Technical Systems
        "Heating system (grills, radiators)",
        "Modern ventilation units",
        "Insulation types",
        "Fire sprinkler systems",
        "Central heating systems",
        "Motion sensor lighting",
        "Integrated waste disposal units",
        "High-efficiency water fixtures",
        "Underfloor heating",
        "Ventilation vents",
        "Modern ventilation systems",
    ],
    [  # Decorative & Ornamental Elements
        "Iron railings",
        "Balcony design",
        "Balconies with glass balustrades",
        "Steel railing designs",
        "Contemporary signage",
        "Contemporary light fixtures",
        "Modern blinds",
    ],
    [  # Community & Lifestyle Features
        "Communal outdoor areas",
        "Public access spaces",
        "Presence of gyms or fitness centers",
        "Shared amenity spaces",
        "Parking facilities",
        "Wheelchair accessibility",
        "Bicycle storage space",
        "Keyless entry systems",
    ],
]

WINDOW_FEATURES = [
    # Material and Construction
    [
        "Window Frame Material",
        "Window Thickness",
        "Type of Glazing Film",
        "Thickness of Glass",
        "Frame Insulation",
        "Frame Color",
        "Sealant Type and Quality",
        "Frame Profile Design",
        "Reinforcement in Frames",
        "Frame Condition and Maintenance",
        "Frame to Wall Connection",
        "Thermal Breaks in Frames",
        "Frame Design Complexity",
        "Depth of Window Frame"
    ],
    # Design and Aesthetics
    [
        "Window Style",
        "Frame Color",
        "Window Beading Style",
        "Overall Aesthetic Integration",
        "Window Sill Design and Detailing",
        "Historical Building Integration",
        "Neighboring Architectural Styles",
        "Recessed Window Placement",
        "Visual Consistency of the Glass Surface",
        "Window Orientation",
        "Shade Affixation",
        "Condition of Glazing Beads"
    ],
    # Functionality and Usability
    [
        "Presence of Gas Filling",
        "Number of Glass Layers",
        "Spacer Bar",
        "Integrated Blinds/Shades",
        "Locking Mechanisms",
        "Weatherstripping",
        "Window Ventilation Features",
        "Integrated Window Sensors",
        "Window Drainage Systems",
        "Inward/Outward Opening Features",
        "Sliding Mechanism Preparation",
        "Mounting Depth",
        "Presence of Micro-ventilation",
        "Presence of Mullions or Muntins",
        "Glazing Pattern and Divisions",
        "Style of Internal Muntins",
        "Internal/External Blinds or Coverings",
        "Style of Internal Blinds"
    ],
    # Performance and Metrics
    [
        "Window U-value",
        "Condensation Pattern",
        "Soundproofing Features",
        "Weather-resistant Coatings",
        "Edge Seal Quality",
        "Weatherproofing Details",
        "Acoustic Properties",
        "Sound Insulation Indicators",
        "Load-bearing Capacity Indicators",
        "Compliance with Fire-rated Standards",
        "UV Fading on Nearby Furnishings",
        "Window Weight"
    ],
    # Security and Safety
    [
        "Window Gaskets",
        "Sealant Type and Quality",
        "Use of Safety Glass",
        "Integrated Security Features",
        "Security Features Specific to Glazing",
        "Visible Documentation or Manufacturer’s Tags"
    ],
    # Compliance and Certification
    [
        "Window Glazing Certification",
        "Visible Brand or Certification Labels"
    ],
    # External Features and Maintenance
    [
        "Edge Finish of Glass",
        "External Capping or Cladding",
        "Cleaning Features",
        "Maintenance Accessibility",
        "Evidence of Recent Installation"
    ],
    # Specialized Features
    [
        "Heat-reflective Tints",
        "Low-emissivity (Low-E) Coating",
        "Anti-glare Features",
        "Visible Energy Labels",
        "Condition of Glazing Beads",
        "Presence of Storm Windows or Additional Layers",
        "Integrated Window Hardware",
        "Reflectivity and Tint of Glass",
        "Presence of Acoustic Glazing Features",
        "Visibility of Desiccant in Spacers"
    ]
]

LIGHTING_FEATURES = [
    # Cluster 1: Fixture Types & Placement
    [
        "Ceiling light fixtures",
        "Wall sconces",
        "Table lamps",
        "Floor lamps",
        "Pendant lights",
        "Recessed lighting",
        "Under-cabinet lighting",
        "Track lighting fixture",
        "Chandelier presence",
        "Ceiling spotlights",
        "Ceiling fan lights",
        "Wall light fixtures",
        "Built-in under-cabinet lighting",
        "Staircase lighting",
        "Kitchen under-shelf lighting"
    ],
    # Cluster 2: Bulb Types & Characteristics
    [
        "Light bulbs visible",
        "Bulb shape",
        "Bulb size",
        "LED presence / LED bulb usage",
        "CFL bulbs / CFL bulb usage",
        "Halogen bulbs / Halogen bulb usage",
        "Incandescent bulbs / Incandescent bulb usage",
        "Open filament bulbs",
        "Smart bulbs",
        "LED strip lighting"
    ],
    # Cluster 3: Light Control & Systems
    [
        "Smart lighting system",
        "Light switch design",
        "Dimmer switch / Dimmer switches",
        "Motion-sensor lights / Motion sensor lights",
        "Lighting control systems / Lighting control apps",
        "Smart home integration",
        "Presence of timers / Timer-based lighting system",
        "Adjustable light direction"
    ],
    # Cluster 4: Lighting Design & Style
    [
        "Lamp shade design",
        "Lamp base material",
        "Light fixture design",
        "Light fixture material",
        "Modern lighting design",
        "Retro lighting design",
        "Decorative lighting elements",
        "Lamp shades style",
        "Light fixture shape",
        "Articulated arms in fixtures",
        "Lamp shade material",
        "Lamp shade color"
    ],
    # Cluster 5: Energy Efficiency & Labels
    [
        "Energy-efficient design",
        "Energy star rating",
        "Energy label indication",
        "Energy usage markings",
        "Estimated bulb lifespan",
        "Energy consumption labels",
        "Energy-saving labels",
        "Bulb wattage indication",
        "Eco-friendly packaging"
    ],
    # Cluster 6: Light Distribution & Qualities
    [
        "Light distribution pattern",
        "Light intensity/brightness",
        "Ambient lighting",
        "Overhead lighting intensity",
        "Downlighting presence",
        "Uplighting presence",
        "Accent lighting / Accent lighting presence",
        "Task lighting / Task lighting presence",
        "Mood lighting / Mood lighting elements",
        "Colour temperature / Light color temperature",
        "Reflective surfaces",
        "Shadow formation",
        "Light spread angle",
        "Light reflection on surfaces",
        "Light beam angle",
        "Usage of frosted bulbs",
        "Colour rendering index (CRI)"
    ],
    # Cluster 7: Additional Design & Feature Attributes
    [
        "Fixture mounting type",
        "Fixture age or modernity",
        "Multiple light settings",
        "Ambient light level",
        "Emergency lighting fixtures",
        "Light fixture size / Number of bulbs per fixture",
        "Fixture mount type"
    ],
    # Cluster 8: Advanced & Specialized Features
    [
        "Skylights or natural light usage with artificial lighting",
        "Presence of chandeliers",
        "Light diffusers usage",
        "Glass fixtures",
        "Cove lighting",
        "Backlighting",
        "Sconce lighting",
        "Lighting in alcoves",
        "Usage of frosted bulbs",
        "Lighting zone plans",
        "Electric socket proximity to lighting",
        "Exterior lighting visibility"
    ]
]

KWH_FEATURES = [
    # 1. Insulation and Thermal Efficiency
    [
        "Airtightness of windows and doors",
        "Ceiling insulation",
        "Insulation in ceiling",
        "Insulation in walls",
        "Insulated attic",
        "Insulated doors",
        "Insulated floors",
        "Insulated pipes",
        "Loft insulation",
        "Roof insulation",
        "Wall insulation",
        "Wall thermostats",
        "Weather stripping on doors",
        "Weather stripping on windows",
        "Thermal blinds",
        "Thermally insulated doors"
    ],
    
    # 2. Heating and Ventilation
    [
        "Central heating system",
        "Condensing boiler",
        "Heat recovery ventilators",
        "High-efficiency boiler",
        "Infrared heating panels",
        "Radiant floor heating",
        "Radiator heating",
        "Radiators",
        "Smart radiator valves",
        "Thermostatic valve",
        "Underfloor heating controls",
        "Ventilated spaces",
        "Ventilation fan",
        "Ventilation system",
        "Zoned heating system"
    ],
    
    # 3. Windows and Doors
    [
        "Double-glazed windows",
        "Draught-proof doors",
        "Energy-efficient door seals",
        "Energy-efficient glazing",
        "Energy-efficient windows",
        "External door",
        "Reflective aluminum windows",
        "Sliding glass doors",
        "Thermal mass",
        "Window blinds"
    ],
    
    # 4. Renewable Energy and Solar
    [
        "Balcony with solar exposure",
        "Green roof",
        "Passive solar heating",
        "Solar panels"
    ],
    
    # 5. Smart and Programmable Systems
    [
        "Building energy management system",
        "Digital thermostat",
        "Energy monitors",
        "Motion sensor lighting",
        "Programmable thermostat",
        "Smart lighting system",
        "Smart meters",
        "Smart thermostat",
        "Wall-mounted thermostat"
    ],
    
    # 6. Lighting Solutions
    [
        "Ceiling spotlights",
        "Energy-efficient bulbs",
        "Energy-efficient lighting",
        "LED ceiling spotlights",
        "LED strip lighting",
        "LED table lamps",
        "LED outdoor lighting",
        "Pendant lighting",
        "Skylight",
        "Under-cabinet lighting"
    ],
    
    # 7. Appliances and Kitchen/Bathroom Fixtures
    [
        "Bathroom extractor fan",
        "Efficient water heater",
        "Efficient water heating systems",
        "Electric kettle",
        "Electric oven",
        "Energy-efficient appliances",
        "Energy-efficient bath",
        "Energy-efficient shower",
        "Energy-efficient toilet",
        "Extractor hood",
        "Microwave",
        "Modern faucets",
        "Modern shower head",
        "Refrigerator",
        "Shower curtain",
        "Shower rail",
        "Stovetop",
        "Tap fittings",
        "Towel rail heater",
        "Water heater",
        "Water-saving taps"
    ],
    
    # 8. Building Materials and Structure
    [
        "Balcony overhangs",
        "Balcony railings (glass)",
        "Brick exterior walls",
        "Composite decking",
        "External cladding",
        "Flat roofing",
        "Gable roof",
        "Hardwood floors",
        "Light shelves to enhance natural light",
        "Roof tiles",
        "Solid wood doors",
        "Thermoplastic flooring",
        "Wall-mounted mirrors",
        "Wall-mounted television"
    ]
]

HEATING_FEATURES = [
    # 1. Radiators and Components
    [
        "Radiators",
        "Radiator covers",
        "Radiator piping",
        "Radiator placement",
        "Radiator pipework",
        "Radiator design",
        "Radiator shielding",
        "Radiator fixing brackets",
        "Metal radiator fins"
    ],
    # 2. Thermostats and Controls
    [
        "Thermostatic radiator valves",
        "Digital thermostat",
        "Programmable thermostats",
        "Thermometer on wall",
        "Presence of thermostat control",
        "Temperature control access",
        "Wall-mounted thermostat",
        "Thermostat wiring",
        "Thermostat style (digital/analog)",
        "Heating system instruction manual"
    ],
    # 3. Boilers and Heating Systems
    [
        "Boiler controls",
        "Combi boiler",
        "Central heating boiler presence",
        "Boiler pressure gauges",
        "Boiler flue placement",
        "Central heating pump",
        "Hydronic heating manifolds"
    ],
    # 4. Heaters and Associated Features
    [
        "Baseboard heaters",
        "Electric heater units",
        "Wall-mounted heater units",
        "Electric wall-mounted heater",
        "Electric fireplace",
        "Infrared heaters",
        "Gas heaters",
        "Heat pump systems",
        "Wall-mounted heater",
        "Heater timers",
        "Electric panel heaters"
    ],
    # 5. Heating and Ventilation Infrastructure
    [
        "Pipes connected to radiators",
        "Heating vents",
        "Ducts for warm air",
        "Wall vent grilles",
        "Floor vents",
        "Vent openings",
        "Wall grills",
        "Ventilation systems",
        "Pipe insulation visible",
        "Heating ducts"
    ],
    # 6. Room and Fixture Design
    [
        "Absence of visible radiators",
        "Sleek heater design",
        "Compact heater design",
        "Heater integration with decor",
        "Modern interior design compatibility",
        "Minimalist heating design",
        "Modern fixture alignment",
        "Heater symmetry with room"
    ],
    # 7. Heating Augmentations and Accessories
    [
        "Underfloor heating controls",
        "Underfloor heating mats",
        "Heated towel racks",
        "Storage heater units",
        "Storage heater input/output dials",
        "Pellet stove exhausts",
        "Ceiling fans absent or present"
    ],
    # 8. Energy Efficiency and Safety
    [
        "Weatherproofing strips on windows",
        "Curtains as insulation",
        "Insulated walls",
        "Smoke detectors nearby heaters",
        "Heater safety grilles",
        "Insulation panels"
    ]
]

# globals
pop_list = []
genbest_list = []
llm_history = []
attributes_to_results = defaultdict(list) # duplication search dictionary
llm_repeated_call = 0
llm_call_count = 0

if EVALUATION_FEATURE == 'WINDOW':
    CHOICE_ARRAY = WINDOW_CHOICE
    TRAINING_SET_ID = WINDOW_TRAINING_SET_ID
    PROMPT_QUESTION = WINDOW_PROMPT_QUESTION
    # INSTRUCTIONS = f"""{OPTION_BLURB} {WINDOW_PROMPT_OPTIONS}"""
    INSTRUCTIONS = WINDOW_ESTIMATION_BLURB
    FINAL_INSTRUCTIONS = WINDOW_FINAL_INSTRUCTIONS
    FEATURE_LIST = WINDOW_FEATURES
elif EVALUATION_FEATURE == 'AGE':
    CHOICE_ARRAY = AGE_CHOICE
    TRAINING_SET_ID = AGE_TRAINING_SET_ID
    PROMPT_QUESTION = AGE_PROMPT_QUESTION
    INSTRUCTIONS = f"""{OPTION_BLURB} {AGE_PROMPT_OPTIONS}"""
    FINAL_INSTRUCTIONS = AGE_FINAL_INSTRUCTIONS
    FEATURE_LIST = AGE_FEATURES
elif EVALUATION_FEATURE == 'LIGHTING':
    CHOICE_ARRAY = LIGHTING_CHOICE
    TRAINING_SET_ID = LIGHTING_TRAINING_SET_ID
    PROMPT_QUESTION = LIGHTING_PROMPT_QUESTION
    INSTRUCTIONS = f"""{OPTION_BLURB} {LIGHTING_PROMPT_OPTIONS}"""
    FINAL_INSTRUCTIONS = LIGHTING_FINAL_INSTRUCTIONS
    FEATURE_LIST = LIGHTING_FEATURES  
elif EVALUATION_FEATURE == 'KWH':
    CHOICE_ARRAY = KWH_CHOICE
    TRAINING_SET_ID = KWH_TRAINING_SET_ID
    PROMPT_QUESTION = KWH_PROMPT_QUESTION
    INSTRUCTIONS = KWH_ESTIMATION_BLURB
    FINAL_INSTRUCTIONS = KWH_FINAL_INSTRUCTIONS
    FEATURE_LIST = KWH_FEATURES
elif EVALUATION_FEATURE == 'HEATING':
    CHOICE_ARRAY = HEATING_CHOICE
    TRAINING_SET_ID = HEATING_TRAINING_SET_ID
    PROMPT_QUESTION = HEATING_PROMPT_QUESTION
    INSTRUCTIONS = f"""{OPTION_BLURB} {HEATING_PROMPT_OPTIONS}"""
    FINAL_INSTRUCTIONS = HEATING_FINAL_INSTRUCTIONS
    FEATURE_LIST = HEATING_FEATURES   


NUM_FEATURES = len(FEATURE_LIST)


def get_key(attributes):
    # key = tuple(sorted(attributes.items()))  # Convert attributes to a sorted tuple of key-value pairs
    flattened_data = {k: ', '.join(v) for k, v in attributes.items()}
    key = tuple(sorted(flattened_data.items()))
    return key


def add_entry(attributes, result):
    """Add attributes and result to the dictionary."""
    key = get_key(attributes)
    attributes_to_results[key].append(result)


# convert list of images to path
def image_list(folder_id, image_string):
    images = []
    array = image_string.split(';')
    for item in array:
        image_url = base_path + str(folder_id) + '/image_' + item + '.jpg'
        images.append(image_url)
    return images


def save_obj(directory, obj, obj_name):
    # Construct the full file path
    file_path = os.path.join(directory, f"s{RANDOM_SEED}_{obj_name}.pkl")
    
    # Save the object to the file
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_feature_list(attributes):
    # Convert dictionary values to a comma-separated string
    return ', '.join(item for sublist in attributes.values() for item in sublist)


def clean_lighting_gt(gt):
    if gt == 'energy efficient spots':
        return 'low energy in 100%'
    elif gt == 'energy efficient lighting':
        return 'low energy in 100%'
    elif gt == 'no low energy lighting':
        return 'low energy in 0%'
    elif gt == 'some low energy in 75%':
        return 'low energy in 75%'
    else:
        return gt


def clean_lighting_prediction(pred):
    if pred == 'no low energy lighting':
        return 'low energy in 0%'
    else:
        return pred


def turn_into_number(item):
    item = item.replace('low energy in ', '')
    item = item.replace('%', '')
    return int(item.strip())


def clean_kwh(raw):
    substrings_to_remove = [
        'kWh/m² per year', 
        'kWh/m²', 
        'kwh/m²', 
        'kWh/m²/year', 
        'kWh/m2',
        '/year',
        'per year',
        'kWh/m',
        'kwh/m2',
        'kWh',
        'kwh',
        'Estimate:',
        'estimate:'
        '.'
    ]
    
    for substring in substrings_to_remove:
        raw = raw.replace(substring, '')
    
    raw = raw.replace(' and ', '-')
    raw = raw.replace(' to ', '-')
    raw = raw.replace(' - ', '-')
    raw = raw.strip()
    return raw


def clean_uval(raw):
    substrings_to_remove = [
        'u-value', 
        'U-Value', 
        'U-value', 
        'Estimate:',
        'estimate:'
        '.'
    ]
    
    for substring in substrings_to_remove:
        raw = raw.replace(substring, '')
    
    raw = raw.replace(' and ', '-')
    raw = raw.replace(' to ', '-')
    raw = raw.replace(' - ', '-')
    raw = raw.strip()
    return raw


def llm_evaluate_based_on_features(attributes, images, gt):
    global llm_repeated_call, llm_call_count
    feature_list = get_feature_list(attributes)

    prompt = f"""
    The images below belong to the same apartment. The building is located in UK. 
    
    {PROMPT_QUESTION} 
    
    Make your judgement focusing on the presence of the following features: {feature_list}
    
    For each feature, say yes if it is visible, no if it is not visible or n/a if it is not applicable, then provide a short explanation.
    
    {INSTRUCTIONS}. 

    {FINAL_INSTRUCTIONS}
    
    """
    try:
        if REAL_LLM:

            content = [
                    {
                      "type": "text",
                      "text": prompt
                    }
                  ]
            # print(images)
            for image in images:
                base64_image = encode_image(image)
                content.append({
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                      }
                    })

            headers = {
              "Content-Type": "application/json",
              "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
              "model": "gpt-4o",
              "messages": [
                {
                  "role": "user",
                  "content": content
                }
              ],
              "max_tokens": 500
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
            
            llm_response = response.json()['choices'][0]['message']['content']

        else:
            llm_response = '### ' + random.choice(CHOICE_ARRAY) + ' ###'

        match = re.search(r"###\s*(.*?)\s*###", llm_response)
        if not match:
            # If ### is not found, check for *** just in case
            match = re.search(r"\*\*\*\s*(.*?)\s*\*\*\*", llm_response)
        
        if match:
            if EVALUATION_FEATURE == 'AGE' or EVALUATION_FEATURE == 'LIGHTING' or EVALUATION_FEATURE == 'KWH' or EVALUATION_FEATURE == 'HEATING' or EVALUATION_FEATURE == 'WINDOW':
                result = match.group(1) # default no numbers
            result = re.sub(r"\.+$", "", result) # remove any trailing full stops
        else:
            raise Exception(f'"No match found: {llm_response}')

        # turn both gt and results to lower case for easier comparison
        result = result.lower().strip()

        if EVALUATION_FEATURE == 'AGE' or EVALUATION_FEATURE == 'LIGHTING' or EVALUATION_FEATURE == 'HEATING':
            gt = gt.lower().strip()
        elif EVALUATION_FEATURE == 'KWH' or EVALUATION_FEATURE == 'WINDOW':
            pass # gt for kwh is a float, no need cleaning

        if EVALUATION_FEATURE == 'AGE':
            result = result.replace('now', '2024') # replace now with 2024
            result = result.replace('before 1900', '1000-1899') # replace before 1900 with 1000-1899
            
            gt = gt.replace('19th century', '1801-1900') # replace 19th century with 1801-1900
            gt = gt.replace('before 1900', '1000-1899') # replace before 1900 with 1000-1899

            answer_range = result.split('-')
            gt_range = gt.split('-')
            
            startA = int(answer_range[0])
            endA = int(answer_range[1])
            startB = int(gt_range[0])
            
            if len(gt_range) < 2: # if ground truth is not a range, check if the year is in the result range
                gap = calculate_gap_point_interval(startA, endA, startB)
                    
            else: # ground truth is a range
                endB = int(gt_range[1])
                gap = calculate_gap(startA, endA, startB, endB)

        elif EVALUATION_FEATURE == 'WINDOW':
            result = clean_uval(result)
            uval = result.split('-')

            startA = float(uval[0])
            startB = float(gt)

            if len(uval) == 2: # if ground truth is not a range, check if the gt is in the result range
                endA = float(uval[1])
                gap = calculate_float_gap_point_interval(startA, endA, startB)
            else:
                gap = round(abs(startA - startB), 1)

        elif EVALUATION_FEATURE == 'LIGHTING':
            result = turn_into_number(clean_lighting_prediction(result))
            gt = turn_into_number(clean_lighting_gt(gt))
            gap = abs(result - gt)
        elif EVALUATION_FEATURE == 'KWH':
            result = clean_kwh(result)
            kwh = result.split('-')

            startA = int(kwh[0])
            startB = int(gt)

            if len(kwh) == 2: # if ground truth is not a range, check if the gt is in the result range
                endA = int(kwh[1])
                gap = calculate_gap_point_interval(startA, endA, startB)
            else:
                gap = abs(startA - startB)
        elif EVALUATION_FEATURE == 'HEATING':
            # clean result to match gt
            if result == 'underfloor heating':
                result = 'underfloor'
            elif result == 'water radiators':
                result = 'water rads'
            elif result == 'electric heaters':
                result = 'electric panels'
            elif result == 'electric storage heaters':
                result = 'electric storage'
            elif result == 'warm air from vents':
                result = 'warm air'
            else:
                raise Exception(f'No match found for heating: {result}, {gt}')

            if result == gt:
                gap = 0
            elif (result == 'warm air' and gt == 'underfloor') or (gt == 'warm air' and result == 'underfloor') or (result == 'electric storage' and gt == 'electric panels') or (gt == 'electric storage' and result == 'electric panels'):
                gap = 1
            else:
                gap = 2

    except Exception as e:
        print(f"An unexpected error occurred getting llm response: {e}")

        print("LLM call response status code:")
        try:
            print(f"Request failed with status code {response.status_code}: {response.text}")
        except:
            print("Can't get status code")

        if llm_repeated_call < MAX_LLM_RETRY: # allow for retry
            llm_repeated_call += 1
            print('LLM retry', llm_repeated_call)
            prompt, llm_response, result, gap = llm_evaluate_based_on_features(attributes, images, gt)
        else:
            save_results_to_file()
            exit("LLM error abort from multiple retries")


    return prompt, llm_response, result, gap


def calculate_gap(startA, endA, startB, endB):
    if endA < startB:
        return startB - endA  # B starts after A ends
    elif endB < startA:
        return startA - endB  # A starts after B ends
    else:
        return 0  # Intervals overlap


def calculate_gap_point_interval(startA, endA, pointB):
    if startA <= pointB <= endA:
        return 0  # pointB is within the interval
    else:
        return min(abs(pointB - startA), abs(pointB - endA))


def calculate_float_gap_point_interval(startA, endA, pointB):
    if startA <= pointB <= endA:
        return 0.0  # pointB is within the interval
    else:
        return round(min(abs(pointB - startA), abs(pointB - endA)), 1)


def evaluate_individual_based_on_features(g, p, attributes):
    global llm_repeated_call
    total_years_difference = 0
    df = pd.read_csv(gt_path)
    for index, row in df.iterrows():
        # if index in [0, 11, 22, 46]:
        if row['row id'] in TRAINING_SET_ID:
            if EVALUATION_FEATURE == 'AGE' or EVALUATION_FEATURE == 'KWH': # kwh also uses age image
                raw_image_list = row['age image']
            elif EVALUATION_FEATURE == 'WINDOW':
                raw_image_list = row['window image']
            elif EVALUATION_FEATURE == 'LIGHTING':
                raw_image_list = row['Lighting image']
            elif EVALUATION_FEATURE == 'HEATING':
                raw_image_list = row['heating image']

            if EVALUATION_FEATURE == 'AGE':
                actual_age = row['building age']
                if pd.isna(actual_age) or actual_age == '':
                    actual_age = row['raw age data']
                gt = actual_age
            elif EVALUATION_FEATURE == 'WINDOW':
                gt = row['window score']
            elif EVALUATION_FEATURE == 'LIGHTING':
                gt = row['Lighting']
            elif EVALUATION_FEATURE == 'KWH':
                gt = row['Energy KWh per square metre from EPC']
            elif EVALUATION_FEATURE == 'HEATING':
                gt = row['heating type']


            llm_repeated_call = 0 # reset repeated call
            llm_prompt, llm_response, result, gap = llm_evaluate_based_on_features(attributes, image_list(index, raw_image_list), gt)
            
            print('-row', index, ', result: ', result, ', gt: ', gt, ', gap: ', gap)
            llm_history.append({'row_id': index, 'address': row['Address'], 'url': row['Thesquare Building URL'], 'image list': raw_image_list, 'attributes': attributes, 'prompt': llm_prompt, 'response': llm_response, 'result': result, 'ground_truth': gt, 'gap': gap})

            total_years_difference += gap
    return total_years_difference


def evaluate_individual_add_to_pop(g, p, attributes):
    print('\n---\ngen', g, 'pop', p, ':', get_feature_list(attributes))
    evaluation_score = evaluate_individual_based_on_features(g, p, attributes)
    add_entry(
        attributes,
        evaluation_score,
    )
    query_key = get_key(attributes)
    results = attributes_to_results.get(query_key, [])

    # get the maximum value (worst score) from results as evaluation score
    worst_score = max(results)

    pop_list[g].append({'generation': g, 'population': p, 'attributes': attributes, 'result': worst_score})
    print('total_score', worst_score, '\n---')
    return evaluation_score


def sort_population(poplist):
    return sorted(poplist, key=lambda k: k['result'], reverse=True) # sort population by decreasing average fitness, less is better


def mutate_attributes(attributes):

    # mutation_action = random.choice(['swap', 'delete', 'add'])
    mutation_action = 'swap'

    # Randomly pick one of the attributes
    random_attribute = random.randrange(0, NUM_FEATURES)
    
    if mutation_action == 'swap' or mutation_action == 'delete':
        if len(attributes[random_attribute]) > 0:
            # randomly pick one item out of the attribute to mutate on
            random_item_id = random.randrange(0, len(attributes[random_attribute]))
        else:
            mutation_action = 'add' # if list is empty, change mutation to add back something

    if mutation_action == 'swap': # change an item to a different value
        attributes[random_attribute][random_item_id] = random.choice(FEATURE_LIST[random_attribute])
        attributes[random_attribute] = list(set(attributes[random_attribute]))
    elif mutation_action == 'delete': # remove an item
        del attributes[random_attribute][random_item_id]
    elif mutation_action == 'add': # add an item
        attributes[random_attribute].append(random.choice(FEATURE_LIST[random_attribute]))

    attributes[random_attribute] = list(set(attributes[random_attribute])) 

    return attributes


def crossover_attributes(parents):
    child_attributes = {}
    for key in parents[0]['attributes']:
        child_attributes[key] = []
        parent0_len = len(parents[0]['attributes'][key])
        parent1_len = len(parents[1]['attributes'][key])
        max_len = max(parent0_len, parent1_len)
        if max_len > 0:
            for i in range(0, max_len):
                chosen_parent = random.choice([0, 1])
                try:
                    parent_value = parents[chosen_parent]['attributes'][key][i]
                    child_attributes[key].append(parent_value)
                except Exception as e:
                    pass # list index out of range, i.e., the item doesn't exist for that parent
        child_attributes[key] = list(set(child_attributes[key])) # remove duplicates to ensure crossover doesn't create duplicates 
    return child_attributes


def get_parents_elites(g):

    sorted_population = sort_population(pop_list[g])

    # get the top NUMBER_OF_PARENTS results as potential parents
    selected_population = sorted_population[-NUMBER_OF_PARENTS:] # select the last NUMBER_OF_PARENTS items

    elites = sorted_population[-2:]

    genbest = sorted_population[-1]

    genbest_list.append(genbest)

    if genbest['result'] == 0:
        save_results_to_file()
        exit("---\nBest fitness of 0 achieved at generation %d" % g)

    return selected_population, elites


def save_results_to_file():
    print("write to file")
    # Create the directory if it doesn't exist
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    save_obj(RESULTS_DIRECTORY, pop_list, 'pop')
    save_obj(RESULTS_DIRECTORY, genbest_list, 'gen_best')
    save_obj(RESULTS_DIRECTORY, llm_history, 'llm_history')
    save_obj(RESULTS_DIRECTORY, attributes_to_results, 'attributes_to_results')

    print(genbest_list)


def run_ga():
    # make random population for first generation
    g = 0

    pop_list.append([])

    for p in range(NUM_POPULATION):
        attributes = {}

        for feature_id in range(0, NUM_FEATURES):
            attributes[feature_id] = [random.choice(FEATURE_LIST[feature_id])] # start with an array with one item
        result = evaluate_individual_add_to_pop(g, p, attributes)

    potential_parents, elites = get_parents_elites(g)

    # subsequent generations
    for g in range(1, NUM_GENERATION):
        
        print('\n-----\ngeneration', g, '\n-----')

        pop_list.append([])
        
        # use the parents to produce NUM_POPULATION-2 offsprings and then use elite (so two best from previous are part of new population)
        for p in range(NUM_POPULATION - 2):
            parents = random.sample(potential_parents, 2) # pick two random parent from potential parents
            
            child_attributes = crossover_attributes(parents)
            child_attributes = mutate_attributes(child_attributes)
            result = evaluate_individual_add_to_pop(g, p, child_attributes)

        # use elite (so best performing parents are part of new population)
        for elite in elites:
            p += 1
            result = evaluate_individual_add_to_pop(g, p, elite['attributes'])

        potential_parents, elites = get_parents_elites(g)

    save_results_to_file()


def parse_arguments():
    global RANDOM_SEED, NUM_POPULATION, NUM_GENERATION

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='enter seed number or enter -1 to use current time as seed number')
    parser.add_argument('-g', '--num_generation', help='enter generation number')
    parser.add_argument('-p', '--num_population', help='enter population number')

    args = parser.parse_args()

    # constants
    seed = int(args.seed)
    NUM_POPULATION = int(args.num_population)
    NUM_GENERATION = int(args.num_generation)
    return seed


def main():
    global RANDOM_SEED
    seed = parse_arguments()

    if seed < 0:
        RANDOM_SEED = int(time.time()) # use current time as random seed
    else:
        RANDOM_SEED = seed

    random.seed(RANDOM_SEED)
    run_ga()


if __name__ == "__main__":
    main()