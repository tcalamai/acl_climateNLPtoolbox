import os
import json
import re

def load_dict(filename):
    """
    Load and return the dictionary from a JSON file.
    
    Parameters:
    filename (str): The name of the file from which to load the dictionary.
    
    Returns:
    dict: The dictionary loaded from the file.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                print("Error: Could not decode the JSON file.")
                return {}
    else:
        print(f"Error: File '{filename}' does not exist.")
        return {}
    

import re

def extract_prompt(text):
    """
    Extracts the prompt between [STARTPROMPT] and [ENDPROMPT] from the given text.

    Parameters:
    text (str): The block of text from which to extract the prompt.

    Returns:
    str: The extracted prompt or a message if the markers are not found.
    """
    pattern = r'\[STARTPROMPT\](.*?)\[ENDPROMPT\]'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return "No prompt found between [STARTPROMPT] and [ENDPROMPT]."
    

def update_question(txt):
    updated_questions = {
    'Account for your organizationâ€™s Scope 3 emissions, disclosing and explaining any exclusions.': "Account for your organization's Scope 3 emissions, disclosing and explaining any exclusions. (Franchises, Investments, Other (upstream), Capital goods, Employee commuting, Other (downstream), Purchased goods and services, Business travel, Fuel-and-energy-related activities (not included in Scope 1 or 2), Upstream leased assets, Downstream transportation and distribution, Processing of sold products, Use of sold products, Downstream leased assets, Upstream transportation and distribution, Waste generated in operations, End of life treatment of sold products)",
     'Account for your organizationâ€™s gross global Scope 3 emissions, disclosing and explaining any exclusions.': "Account for your organization's gross global Scope 3 emissions, disclosing and explaining any exclusions. (Upstream transportation and distribution, Waste generated in operations, Upstream leased assets, Processing of sold products, End of life treatment of sold products, Downstream leased assets, Investments, Capital goods, Fuel-and-energy-related activities (not included in Scope 1 or 2), Employee commuting, Use of sold products, Purchased goods and services, Downstream transportation and distribution, Franchises, Other (upstream), Other (downstream), Business travel)",
     'Describe what your organization considers to be short-, medium- and long-term horizons.': 'Describe what your organization considers to be short-, medium- and long-term horizons. (Medium-term, Short-term, Long-term)',
     'How does your organization define short-, medium- and long-term time horizons?': 'How does your organization define short-, medium- and long-term time horizons? (Medium-term, Long-term, Short-term)',
     'Identify the total number of initiatives at each stage of development, and for those in the implementation stages, the estimated CO2e savings.': 'Identify the total number of initiatives at each stage of development, and for those in the implementation stages, the estimated CO2e savings. (Implemented*, Not to be implemented, Under investigation, To be implemented*, Implementation commenced*)',
     'Identify the total number of projects at each stage of development, and for those in the implementation stages, the estimated CO2e savings.': 'Identify the total number of projects at each stage of development, and for those in the implementation stages, the estimated CO2e savings. (To be implemented*, Implementation commenced*, Not to be implemented, Under investigation, Implemented*)',
     'Provide details on the electricity, heat, steam, and cooling your organization has generated and consumed in the reporting year.': 'Provide details on the electricity, heat, steam, and cooling your organization has generated and consumed in the reporting year. (Heat, Cooling, Steam, Electricity)',
     'Report your organizationâ€™s energy consumption totals (excluding feedstocks) in MWh.': "Report your organization's energy consumption totals (excluding feedstocks) in MWh. (Consumption of fuel (excluding feedstock), Consumption of purchased or acquired electricity, Consumption of purchased or acquired steam, Total energy consumption, Consumption of self-generated non-fuel renewable energy, Consumption of purchased or acquired heat, Consumption of purchased or acquired cooling)",
     'Select the applications of your organizationâ€™s consumption of fuel.': "Select the applications of your organization's consumption of fuel. (Consumption of fuel for the generation of steam, Consumption of fuel for co-generation or tri-generation, Consumption of fuel for the generation of electricity, Consumption of fuel for the generation of cooling, Consumption of fuel for the generation of heat)",
     'Select which energy-related activities your organization has undertaken.': 'Select which energy-related activities your organization has undertaken. (Consumption of fuel (excluding feedstocks), Consumption of purchased or acquired electricity , Generation of electricity, heat, steam, or cooling, Consumption of purchased or acquired heat, Consumption of purchased or acquired steam, Consumption of purchased or acquired cooling)',
     "Which of the following risk types are considered in your organization's climate-related risk assessments?": "Which of the following risk types are considered in your organization's climate-related risk assessments? (Current regulation, Market , Acute physical , Upstream , Emerging regulation, Technology , Legal , Reputation , Downstream , Chronic physical )",
     "Which risk types are considered in your organization's climate-related risk assessments?": "Which risk types are considered in your organization's climate-related risk assessments? (Technology , Acute physical , Chronic physical , Emerging regulation, Legal , Market , Reputation , Current regulation)"
    }
    if txt in updated_questions.keys():
        return updated_questions[txt]
    else:
        return txt

map_lobbymap_stance = {
        "alignment_with_ipcc_on_climate_action":'Alignment with IPCC on climate action',
        "carbon_tax":'Carbon tax',
        "communication_of_climate_science":'Communication of climate science',
        "disclosure_on_relationships":'Disclosure on relationships',
        "emissions_trading":'Emissions trading',
        "energy_and_resource_efficiency":'Energy and resource efficiency',
        "energy_transition_&_zero_carbon_technologies":'Energy transition & zero carbon technologies',
        "ghg_emission_regulation":'GHG emission regulation',
        "land_use":'Land use',
        "renewable_energy":'Renewable energy',
        "support_of_un_climate_process":'Support of UN climate process',
        "supporting_the_need_for_regulations":'Supporting the need for regulations',
         "transparency_on_legislation":'Transparency on legislation'
    }

def prepare_content(row, dataset_name, task_descriptions, prompts):  
    if dataset_name == "lobbympa_query_origin":
        dataset_name = "lobbympa_query"
    if dataset_name == "lobbympa_stance_origin":
        dataset_name = "lobbympa_stance"

    if dataset_name in ['lobbymap_stance']:
        return extract_prompt(prompts[dataset_name]).replace("[[Insert Text Here]]", row['clean_text']).replace("[[Insert Query Here]]", task_descriptions['lobbymap_query']['labels'][map_lobbymap_stance[row['query']]])
    elif dataset_name == "lobbymap":
        return extract_prompt(prompts[dataset_name]).replace("[[Insert List of Page Content Here]]", row['X'])
    elif dataset_name in ['climateFEVER_evidence']:
        return extract_prompt(prompts[dataset_name]).replace("[[Insert Text Here]]", row['clean_text']).replace("[[Insert Query Here]]", row['query'])
    elif dataset_name in ['climaQA']:
        return extract_prompt(prompts[dataset_name]).replace("[[Insert Text Here]]", row['clean_text']).replace("[[Insert Query Here]]", update_question(row['query']))
    else:
        return extract_prompt(prompts[dataset_name]).replace("[[Insert Text Here]]", row['clean_text'])
