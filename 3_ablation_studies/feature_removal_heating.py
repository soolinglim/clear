import base64
import requests

# OpenAI API Key
api_key = "xxxxx"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')



import pandas as pd

base_path = "clear/apartment_images/"
gt_path = 'ground_truth.csv'

# convert list of images to path
def image_list(folder_id, image_string):
    images = []
    array = image_string.split(';')
    for item in array:
        image_url = base_path + str(folder_id) + '/image_' + item + '.jpg'
        images.append(image_url)
    return images


def std(data):
    """ calculate the sample standard deviation. specify ddof=0 to compute the population standard deviation. """
    ddof = 1.0
    n = len(data)
    mean = sum(data)/float(n)
    if n < 2.0: # make it still work for n < 2
        std = 0.0
    else:
        ss = sum((x-mean)**2 for x in data)
        pvar = ss/(n-ddof)
        std = pvar**0.5
    return std


def mean(data):
    n = len(data)
    return sum(data)/float(n)


# evaluate ga outcome on test dataset

# get test set

def get_test_set_numbers(existing_numbers):
    MAX_ID = 46
    all_numbers = set(range(MAX_ID + 1))  # Create a set of numbers from 0 to full_range
    existing_set = set(existing_numbers)     # Convert the list to a set for faster operations
    missing_numbers = sorted(all_numbers - existing_set)  # Find the difference and sort the result
    return missing_numbers


HEATING_PROMPT_QUESTION = "What type of heating does this apartment have?"
OPTION_BLURB = "Finally, select one of these options:"
HEATING_FINAL_INSTRUCTIONS = 'You can only use one of these, do not modify or invent your own options. Put the selected option in between ### and ###'

HEATING_CHOICE = ['underfloor heating', 'water radiators', 'electric heaters', 'electric storage heaters', 'warm air from vents']
HEATING_PROMPT_OPTIONS = ', '.join(HEATING_CHOICE) 
PROMPT_OPTIONS = HEATING_PROMPT_OPTIONS
PROMPT_QUESTION = HEATING_PROMPT_QUESTION
INSTRUCTIONS = f"""{OPTION_BLURB} {HEATING_PROMPT_OPTIONS}"""
FINAL_INSTRUCTIONS = HEATING_FINAL_INSTRUCTIONS
EVALUATION_FEATURE = 'HEATING'
TRAINING_SET_ID = [1, 3, 4, 6, 8, 9, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 31, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43]
TEST_SET_ID = get_test_set_numbers(TRAINING_SET_ID)
print('test set id', TEST_SET_ID)
print('test set count', len(TEST_SET_ID))


import re

llm_repeated_call = 0
REAL_LLM = True
max_llm_retry = 20

def get_feature_list(attributes):
    # Convert dictionary values to a comma-separated string
    # return ', '.join(attributes.values())
    return ', '.join(item for sublist in attributes.values() for item in sublist)


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


def llm_evaluate_based_on_features(attributes, images, gt):
    global llm_repeated_call, llm_call_count
    feature_list = get_feature_list(attributes)
    prompt = f"""
    The images below belong to the same apartment. The building is located in UK. 
    
    {PROMPT_QUESTION} 
    
    Make your judgement focussing on the presence of the following features: {feature_list}
    
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
            # if len(genbest_list) == 0:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            # llm_call_count += 1
            
            llm_response = response.json()['choices'][0]['message']['content']

        else:
            llm_response = '### ' + random.choice(CHOICE_ARRAY) + ' ###'

        match = re.search(r"###\s*(.*?)\s*###", llm_response)

        if not match:
            # If ### is not found, check for *** just in case
            match = re.search(r"\*\*\*\s*(.*?)\s*\*\*\*", llm_response)
        
        if match:
            if EVALUATION_FEATURE == 'AGE' or EVALUATION_FEATURE == 'LIGHTING' or EVALUATION_FEATURE == 'KWH' or EVALUATION_FEATURE == 'HEATING':
                result = match.group(1) # default no numbers
            # result = re.sub(r"\(\d\)\s*", "", match.group(1)) # handling (1) text
            # result = re.sub(r"\(?\d\)?\)\s*", "", match.group(1)) # handling (1) text, 1) text, 1)text
            # result = re.sub(r"\(?\d\)?[.)]\s*", "", match.group(1))  # handling text, (1) text, 1) text, 1)text, 1.text, 1. text
            elif EVALUATION_FEATURE == 'WINDOW':
                result = re.sub(r"[\(\{\[]?\d[\)\}\]]?[.)]?\s*", "", match.group(1)) # handling text, (1) text, 1) text, 1)text, 1.text, 1. text, {1} text and {1}text
            result = re.sub(r"\.+$", "", result) # remove any trailing full stops
        else:
            raise Exception(f'"No match found: {llm_response}')

        # turn both gt and results to lower case for easier comparison
        result = result.lower().strip()
        
        if EVALUATION_FEATURE == 'AGE' or EVALUATION_FEATURE == 'WINDOW' or EVALUATION_FEATURE == 'LIGHTING' or EVALUATION_FEATURE == 'HEATING':
            gt = gt.lower().strip()
        elif EVALUATION_FEATURE == 'KWH':
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

            if startA >= 1970:
                recent = True
            else:
                recent = False
        elif EVALUATION_FEATURE == 'WINDOW':
            if gt == 'high efficiency double or triple glazed, pvc frames':
                gt = 'high efficiency double or triple glazed'

            # just in case results come back as just numbered options, convert back to text
            if result.isdigit():
                item_id = int(result) - 1
                result = WINDOW_CHOICE[item_id]

            if result == gt:
                gap = 0
            else:
                if (gt == 'high efficiency double or triple glazed' and result == 'double glazed') or (result == 'high efficiency double or triple glazed' and gt == 'double glazed'):
                    gap = 1
                elif (gt == 'double glazed' and result == 'single glazed') or (result == 'double glazed' and gt == 'single glazed'):
                    gap = 1
                elif (gt == 'high efficiency double or triple glazed' and result == 'single glazed') or (result == 'high efficiency double or triple glazed' and gt == 'single glazed'):
                    gap = 2
                else:
                    raise Exception(f'No match found for windows: {result}, {gt}')
        elif EVALUATION_FEATURE == 'LIGHTING':
            result = turn_into_number(clean_lighting_prediction(result))
            gt = turn_into_number(clean_lighting_gt(gt))
            gap = abs(result - gt)
        elif EVALUATION_FEATURE == 'KWH':
            result = clean_kwh(result)
            kwh = result.split('-')

            startA = int(kwh[0])
            startB = int(gt)

            if len(kwh) == 2: # if ground truth is not a range, check if the year is in the result range
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

        if llm_repeated_call < max_llm_retry:
            llm_repeated_call += 1
            print('LLM retry', llm_repeated_call)
            prompt, llm_response, result, gap = llm_evaluate_based_on_features(attributes, images, gt)
        else:
            save_results_to_file()
            exit("LLM error abort from multiple retries")
    return prompt, llm_response, result, gap


def evaluate_testing_data(attributes, test_set_ids):
    total_gap = 0
    df = pd.read_csv(gt_path)
    for index, row in df.iterrows():
        if index in test_set_ids:
            if EVALUATION_FEATURE == 'AGE' or EVALUATION_FEATURE == 'KWH':
                raw_image_list = row['age image']
            elif EVALUATION_FEATURE == 'WINDOW':
                raw_image_list = row['window image']
            elif EVALUATION_FEATURE == 'LIGHTING':
                raw_image_list = row['Lighting image']
            elif EVALUATION_FEATURE == 'HEATING':
                raw_image_list = row['heating image']
            # print('\n\n\n---row', index, row['Address'], row['Thesquare Building URL'])
            # print('---row', index)

            # print('---llm', result)
            if EVALUATION_FEATURE == 'AGE':
                actual_age = row['building age']
                if pd.isna(actual_age) or actual_age == '':
                    actual_age = row['raw age data']
                gt = actual_age
            elif EVALUATION_FEATURE == 'WINDOW':
                gt = row['window type']
            elif EVALUATION_FEATURE == 'LIGHTING':
                gt = row['Lighting']
            elif EVALUATION_FEATURE == 'KWH':
                gt = row['Energy KWh per square metre from EPC']
            elif EVALUATION_FEATURE == 'HEATING':
                gt = row['heating type']

            llm_repeated_call = 0 # reset repeated call
            llm_prompt, llm_response, result, gap = llm_evaluate_based_on_features(attributes, image_list(index, raw_image_list), gt)
            
            print('-row', index, ', result:', result, ', gt:', gt, ', gap:', gap)
            print('llm response', llm_response, '\n\n\n')
            total_gap += gap
    return total_gap

attributes = {0: ['Radiator pipework'], 1: [], 2: [], 3: ['Electric panel heaters'], 4: ['Wall grills'], 5: ['Modern interior design compatibility'], 6: [], 7: []}
# Combine all items into a single list
all_items = [item for sublist in attributes.values() for item in sublist]

total_testing_result_list = []

# Loop to create new dicts with one item removed in each iteration
for i in range(len(all_items)):
    new_dict = {0: [item for j, item in enumerate(all_items) if j != i]}
    print(f"Iteration {i+1}: {new_dict}")

    testing_results = evaluate_testing_data(new_dict, TEST_SET_ID)
    print('test results total gap', testing_results)
    total_testing_result_list.append(testing_results)
    print('==============')


print('heating final results', total_testing_result_list)
print('mean', mean(total_testing_result_list))
print('std', std(total_testing_result_list))
