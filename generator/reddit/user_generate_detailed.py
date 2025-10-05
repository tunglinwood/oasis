# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import json
import random
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os

from openai import OpenAI

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate detailed synthetic Reddit user profiles")
    parser.add_argument("--openai-api-key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--openai-base-url", type=str, default=None, help="OpenAI base URL (optional)")
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo", help="Model name to use")
    parser.add_argument("--max-workers", type=int, default=100, help="Maximum number of worker threads")
    parser.add_argument("--N-of-agents", type=int, default=10000, help="Number of user profiles to generate")
    parser.add_argument("--output-path", type=str, default="./experiment_dataset/user_data/generated_user_data_detailed_10000.json", help="Output file path")
    return parser.parse_args()


# Initialize client with CLI parameters
args = parse_arguments()
client = OpenAI(api_key=args.openai_api_key)
if args.openai_base_url:
    client.base_url = args.openai_base_url

# Gender ratio
female_ratio = 0.4
gender_ratio = [female_ratio, 1-female_ratio]
genders = ['female', 'male']

# Education level - based on typical distribution
education_ratio = [0.15, 0.25, 0.35, 0.20, 0.05]  # Elementary, High School, Bachelor's, Master's, PhD
education_levels = ['elementary', 'high school', 'bachelor', 'master', 'phd']

# Income bracket - based on US income distribution (can be adapted)
income_ratio = [0.20, 0.30, 0.25, 0.15, 0.10]  # Low, Lower-middle, Middle, Upper-middle, High
income_brackets = ['low', 'lower-middle', 'middle', 'upper-middle', 'high']

# Location type
location_ratio = [0.82, 0.10, 0.08]  # Urban, Suburban, Rural
location_types = ['urban', 'suburban', 'rural']

# Family status
family_ratio = [0.30, 0.40, 0.20, 0.05, 0.05]  # Single, Married-no-kids, Married-with-kids, Single-parent, Multi-generational
family_statuses = ['single', 'married-no-kids', 'married-with-kids', 'single-parent', 'multi-generational']

# Religious affiliation
religion_ratio = [0.65, 0.15, 0.05, 0.05, 0.10]  # No religion, Christian, Muslim, Other religion, Undeclared
religions = ['no religion', 'christian', 'muslim', 'other religion', 'undeclared']

# Age ratio
age_ratio = [0.44, 0.31, 0.11, 0.03, 0.11]
age_groups = ['18-29', '30-49', '50-64', '65-100', 'underage']

# Big Five Personality Traits - each trait will be scored 0-100
# These functions generate scores based on normal distributions for each trait
def get_openness_score():
    return min(100, max(0, int(random.normalvariate(50, 15))))

def get_conscientiousness_score():
    return min(100, max(0, int(random.normalvariate(50, 15))))

def get_extraversion_score():
    return min(100, max(0, int(random.normalvariate(50, 15))))

def get_agreeableness_score():
    return min(100, max(0, int(random.normalvariate(50, 15))))

def get_neuroticism_score():
    return min(100, max(0, int(random.normalvariate(50, 15))))

# Sexual orientation
sexual_orientation_ratio = [0.85, 0.04, 0.02, 0.02, 0.07]
sexual_orientations = ['heterosexual', 'homosexual', 'bisexual', 'pansexual', 'asexual']

# Gender identity (beyond binary)
gender_identity_ratio = [0.98, 0.01, 0.01]
gender_identities = ['cisgender', 'transgender', 'non-binary']

# Ethnicity/Race (US-focused, can be adapted)
ethnicity_ratio = [0.58, 0.19, 0.13, 0.04, 0.04, 0.02]
ethnicities = ['white', 'hispanic', 'black', 'asian', 'mixed', 'other']

# Disability status
disability_ratio = [0.85, 0.15]
disabilities = ['no disability', 'with disability']

# Country ratio
country_ratio = [0.4833, 0.0733, 0.0697, 0.0416, 0.0306, 0.3016]
countries = ["US", "UK", "Canada", "Australia", "Germany", "Other"]

# Profession ratio
p_professions = [1 / 16] * 16
professions = [
    "Agriculture, Food & Natural Resources", "Architecture & Construction",
    "Arts, Audio/Video Technology & Communications",
    "Business Management & Administration", "Education & Training", "Finance",
    "Government & Public Administration", "Health Science",
    "Hospitality & Tourism", "Human Services", "Information Technology",
    "Law, Public Safety, Corrections & Security", "Manufacturing", "Marketing",
    "Science, Technology, Engineering & Mathematics",
    "Transportation, Distribution & Logistics"
]

# Digital literacy levels
digital_literacy_ratio = [0.20, 0.30, 0.30, 0.20]  # Low, Medium, High, Very High
digital_literacy_levels = ['low', 'medium', 'high', 'very high']

# Consumption patterns
consumption_ratio = [0.25, 0.25, 0.25, 0.25]  # Essential-focused, Balanced, Luxury-oriented, Value-conscious
consumption_patterns = ['essential-focused', 'balanced', 'luxury-oriented', 'value-conscious']

# Information sources
info_source_ratio = [0.30, 0.25, 0.20, 0.15, 0.10]  # Social media, Traditional media, Online sources, Personal networks, Professional sources
info_sources = ['social media', 'traditional media', 'online sources', 'personal networks', 'professional sources']




def get_random_age():
    group = random.choices(age_groups, age_ratio)[0]
    if group == 'underage':
        return random.randint(10, 17)
    elif group == '18-29':
        return random.randint(18, 29)
    elif group == '30-49':
        return random.randint(30, 49)
    elif group == '50-64':
        return random.randint(50, 64)
    else:
        return random.randint(65, 100)


def adjust_education_for_age(age, original_education):
    """Adjust education based on age to ensure realistic life stage"""
    # Define typical education levels for age groups
    if age <= 12:  # under 13
        # Elementary age
        return 'elementary'
    elif 13 <= age <= 17:  # teens
        # Should be in high school or just finished
        possible_education = ['elementary', 'high school']
        return random.choice(possible_education)
    elif 18 <= age <= 22:  # traditional college age
        # Could be in college or just finished with high school or bachelor
        possible_education = ['high school', 'bachelor']
        return random.choice(possible_education)
    elif 23 <= age <= 25:  # recent graduates
        # Could have bachelor's, or master's if accelerated
        possible_education = ['high school', 'bachelor', 'master']
        return random.choice(possible_education)
    elif 26 <= age <= 35:  # early career
        # Could have bachelor's, master's, or even PhD by now
        possible_education = ['high school', 'bachelor', 'master', 'phd']
        return random.choice(possible_education)
    elif 36 <= age <= 50:  # mid career
        # Could have any education level, including PhD
        return original_education
    else:  # 50+
        # Could have any education level
        return original_education


def adjust_family_status_for_age(age, original_family_status):
    """Adjust family status based on age to ensure realistic life stage"""
    if age <= 17:  # under 18
        # Should likely be single or in multi-generational family
        possible_family = ['single', 'multi-generational']
        return random.choice(possible_family)
    elif 18 <= age <= 25:  # young adults
        # More likely to be single, could be single parent
        possible_family = ['single', 'single-parent', 'multi-generational']
        return random.choice(possible_family)
    elif 26 <= age <= 35:  # early 20s to mid 30s
        # Could be single, married (with or without kids), single parent
        return original_family_status
    else:  # 35+
        # Any family status possible
        return original_family_status


def adjust_profession_for_age(age, original_profession):
    """Adjust profession based on age to ensure realistic career stage"""
    if age <= 17:  # teens - should not have full-time professional careers
        # Students might have part-time jobs or be in education
        student_professions = [
            "Education & Training",  # For those in school
        ]
        return random.choice(student_professions)
    elif 18 <= age <= 22:  # traditional college age
        # Could be students or entry-level positions
        entry_level_professions = [
            "Agriculture, Food & Natural Resources",
            "Hospitality & Tourism", 
            "Human Services",
            "Transportation, Distribution & Logistics",
            "Manufacturing",
            "Education & Training"  # For teaching assistants, etc.
        ]
        return random.choice(entry_level_professions)
    else:  # 23+, could have any profession based on education and experience
        return original_profession


def adjust_digital_literacy_for_age_education(age, education, original_digital_literacy):
    """Adjust digital literacy based on age and education for realistic expectations"""
    education_index = education_levels.index(education)
    
    if age <= 12:  # Very young children
        # Children under 13 have limited but growing digital literacy
        return random.choices(digital_literacy_levels, [0.2, 0.4, 0.3, 0.1])[0]  # Mostly low-medium
    elif age <= 25:  # Teens and young adults typically have higher digital literacy 
        if education_index <= 1:  # But those with lower education may have varying levels
            return random.choices(digital_literacy_levels, [0.1, 0.2, 0.4, 0.3])[0]  # Skewed towards higher
        else:  # Higher education typically means better digital literacy
            return random.choices(digital_literacy_levels, [0.05, 0.15, 0.3, 0.5])[0]  # Highly skewed towards high/very high
    elif 26 <= age <= 35:
        if education_index <= 1:
            return random.choices(digital_literacy_levels, [0.15, 0.25, 0.4, 0.2])[0]
        else:
            return random.choices(digital_literacy_levels, [0.05, 0.15, 0.4, 0.4])[0]
    elif 36 <= age <= 50:
        if education_index <= 1:
            return random.choices(digital_literacy_levels, [0.25, 0.35, 0.3, 0.1])[0]
        else:
            return random.choices(digital_literacy_levels, [0.1, 0.2, 0.4, 0.3])[0]
    else:  # 50+
        if education_index <= 1:
            return random.choices(digital_literacy_levels, [0.4, 0.4, 0.15, 0.05])[0]  # Older with lower education typically has lower digital literacy
        else:
            return random.choices(digital_literacy_levels, [0.2, 0.4, 0.3, 0.1])[0]  # Still better if educated


def adjust_info_source_for_digital_literacy_age(digital_literacy, age, original_info_source):
    """Adjust information source preference based on digital literacy and age"""
    digital_level_index = digital_literacy_levels.index(digital_literacy)
    
    if digital_level_index >= 2:  # High or very high digital literacy
        if age <= 40:  # Younger people with high digital literacy prefer online sources
            return random.choices(info_sources, [0.1, 0.05, 0.6, 0.2, 0.05])[0]  # Heavy online sources
        else:  # Older people with high digital literacy might still prefer traditional
            return random.choices(info_sources, [0.2, 0.25, 0.35, 0.15, 0.05])[0]  # Balanced but online leaning
    elif digital_level_index == 1:  # Medium digital literacy
        if age <= 40:
            balanced_distribution = [0.25, 0.2, 0.3, 0.2, 0.05]  # More online than traditional
        else:
            balanced_distribution = [0.25, 0.3, 0.25, 0.15, 0.05]  # More traditional
        return random.choices(info_sources, balanced_distribution)[0]
    else:  # Low digital literacy
        if age <= 40:  # Younger people with low digital literacy still use some online
            return random.choices(info_sources, [0.2, 0.35, 0.25, 0.15, 0.05])[0]
        else:  # Older people with low digital literacy prefer traditional sources
            return random.choices(info_sources, [0.1, 0.4, 0.15, 0.3, 0.05])[0]  # Heavy traditional media


def adjust_income_for_age_education(age, education, original_income):
    """Adjust income based on age and education to ensure realistic expectations"""
    education_index = education_levels.index(education)
    age_group = None
    
    if age <= 17:
        age_group = 'minor'  # Under 18 - typically no significant income
    elif age <= 25:
        age_group = 'young'
    elif 26 <= age <= 35:
        age_group = 'early_career'
    elif 36 <= age <= 50:
        age_group = 'mid_career'
    else:
        age_group = 'experienced'
    
    # Based on age and education, adjust income distribution
    if age_group == 'minor':  # Under 18
        # Minors typically have very limited income (allowance, part-time jobs at minimum wage)
        # Even if from "high-income" families, their personal income is minimal
        return random.choices(income_brackets, [0.6, 0.35, 0.04, 0.01, 0.0])[0]  # Mostly low or lower-middle
    elif age_group == 'young':
        # Young people typically have lower incomes regardless of education
        if education_index <= 1:  # elementary or high school
            # Even lower income
            return random.choices(income_brackets, [0.4, 0.4, 0.15, 0.04, 0.01])[0]
        else:  # bachelor or higher
            # Slightly better but still entry level
            return random.choices(income_brackets, [0.3, 0.35, 0.2, 0.1, 0.05])[0]
    elif age_group == 'early_career':
        if education_index <= 1:  # elementary or high school
            return random.choices(income_brackets, [0.25, 0.35, 0.25, 0.1, 0.05])[0]
        elif education_index == 2:  # bachelor
            return random.choices(income_brackets, [0.15, 0.3, 0.35, 0.15, 0.05])[0]
        else:  # master/phd
            return random.choices(income_brackets, [0.05, 0.2, 0.4, 0.25, 0.1])[0]
    elif age_group == 'mid_career':
        if education_index <= 1:  # elementary or high school
            return random.choices(income_brackets, [0.15, 0.3, 0.35, 0.15, 0.05])[0]
        elif education_index == 2:  # bachelor
            return random.choices(income_brackets, [0.1, 0.2, 0.4, 0.2, 0.1])[0]
        else:  # master/phd
            return random.choices(income_brackets, [0.05, 0.15, 0.3, 0.35, 0.15])[0]
    else:  # experienced
        if education_index <= 1:  # elementary or high school
            return random.choices(income_brackets, [0.1, 0.25, 0.35, 0.2, 0.1])[0]
        elif education_index == 2:  # bachelor
            return random.choices(income_brackets, [0.05, 0.15, 0.35, 0.3, 0.15])[0]
        else:  # master/phd
            return random.choices(income_brackets, [0.02, 0.1, 0.25, 0.35, 0.28])[0]


def get_random_education():
    return random.choices(education_levels, education_ratio)[0]


def get_random_income():
    return random.choices(income_brackets, income_ratio)[0]


def get_random_location():
    return random.choices(location_types, location_ratio)[0]


def get_random_family_status():
    return random.choices(family_statuses, family_ratio)[0]


def get_random_religion():
    return random.choices(religions, religion_ratio)[0]


def get_random_sexual_orientation():
    return random.choices(sexual_orientations, sexual_orientation_ratio)[0]


def get_random_gender_identity():
    return random.choices(gender_identities, gender_identity_ratio)[0]


def get_random_ethnicity():
    return random.choices(ethnicities, ethnicity_ratio)[0]


def get_random_disability_status():
    return random.choices(disabilities, disability_ratio)[0]


def get_random_digital_literacy():
    return random.choices(digital_literacy_levels, digital_literacy_ratio)[0]


def get_random_consumption_pattern():
    return random.choices(consumption_patterns, consumption_ratio)[0]


def get_random_info_source():
    return random.choices(info_sources, info_source_ratio)[0]

def calculate_ses_score(education, income, profession):
    """
    Calculate a Socioeconomic Status (SES) score based on:
    - Education (0-4: elementary to PhD) multiplied by 0.4
    - Income (0-4: low to high) multiplied by 0.4
    - Profession (0-15: 16 profession types) normalized and multiplied by 0.2
    """
    education_idx = education_levels.index(education)
    income_idx = income_brackets.index(income)
    profession_idx = professions.index(profession)
    
    # Normalize profession index to 0-4 scale
    normalized_profession = (profession_idx / 15) * 4
    
    ses_score = round((education_idx * 0.4) + (income_idx * 0.4) + (normalized_profession * 0.2), 2)
    return min(4, max(0, ses_score))  # Clamp to 0-4 range

def get_random_gender():
    return random.choices(genders, gender_ratio)[0]

def adjust_ethnicity_for_country(country, original_ethnicity):
    """Adjust ethnicity based on country to ensure geographic and cultural consistency"""
    # Define major ethnicities by regions/countries
    north_american_ethnicities = ['white', 'hispanic', 'black', 'asian', 'mixed', 'other']
    european_ethnicities = ['white', 'asian', 'black', 'mixed', 'other']
    latin_american_ethnicities = ['hispanic', 'white', 'black', 'mixed', 'other']
    asian_ethnicities = ['asian', 'white', 'mixed', 'other']
    african_ethnicities = ['black', 'white', 'mixed', 'other']
    
    # Make country name case-insensitive for comparison
    country_lower = country.lower().strip()
    
    # North American countries
    if country_lower in ['us', 'usa', 'united states', 'canada', 'united states of america']:
        return random.choice(north_american_ethnicities)
    # European countries  
    elif country_lower in ['uk', 'united kingdom', 'great britain', 'england', 'scotland', 'france', 'germany', 'italy', 'spain', 'netherlands', 'belgium', 'austria', 'sweden', 'norway', 'denmark', 'finland', 'ireland', 'portugal', 'greece', 'poland', 'czech republic', 'hungary']:
        return random.choice(european_ethnicities) 
    # Latin American countries
    elif country_lower in ['mexico', 'brazil', 'argentina', 'chile', 'colombia', 'venezuela', 'peru', 'ecuador', 'bolivia', 'uruguay', 'paraguay', 'el salvador', 'guatemala', 'honduras', 'nicaragua', 'costa rica', 'panama']:
        return random.choice(latin_american_ethnicities)
    # Asian countries
    elif country_lower in ['china', 'japan', 'south korea', 'north korea', 'india', 'indonesia', 'thailand', 'vietnam', 'malaysia', 'philippines', 'singapore', 'taiwan', 'hong kong', 'pakistan', 'bangladesh', 'sri lanka', 'myanmar', 'cambodia', 'laos']:
        return random.choice(asian_ethnicities)
    # African countries
    elif country_lower in ['nigeria', 'egypt', 'south africa', 'kenya', 'ethiopia', 'ghana', 'morocco', 'uganda', 'sudan', 'algeria', 'tunisia', 'libya', 'zimbabwe', 'botswana', 'zambia', 'angola', 'cameroon', 'senegal', 'mali', 'ghana', 'ivory coast', 'burkina faso', 'niger', 'togo', 'benin']:
        return random.choice(african_ethnicities)
    else:
        # Default to original distribution for unknown countries
        return original_ethnicity


def adjust_gender_for_identity(gender, gender_identity):
    """Adjust gender based on gender identity for consistency"""
    # If gender identity is non-binary or transgender, the gender field should be more flexible
    if gender_identity == 'non-binary':
        # Non-binary individuals may identify as any gender, but let's keep it consistent
        # For simplicity in this context, we'll allow the original gender
        return gender
    elif gender_identity == 'transgender':
        # Transgender individuals may have a gender that differs from assigned gender
        # For this simulation, we'll maintain the original gender unless specifically adjusted
        return gender
    else:  # cisgender
        # For cisgender individuals, gender and identity should align
        return gender


def get_random_country():
    country = random.choices(countries, country_ratio)[0]
    if country == "Other":
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[{
                "role": "system",
                "content": "Select a real country name randomly. Respond with only the country name, nothing else."
            }])
        # Extract just the country name, removing any extra text like emojis or explanations
        full_response = response.choices[0].message.content.strip()
        # Get the first line and remove any emojis or extra text
        country_name = full_response.split('\n')[0].strip()
        # Remove any emojis and keep only the country name
        import re
        # Remove emojis using regex
        country_name = re.sub(r'[^\w\s,-]', '', country_name).strip()
        # If there are multiple words, take only the country part (usually the last word/phrase)
        lines = full_response.split('\n')
        for line in lines:
            line = re.sub(r'[^\w\s,-]', '', line).strip()
            if line and len(line.split()) <= 3:  # Most countries have 3 words or less
                country_name = line
                break
        return country_name
    return country


def adjust_profession_for_education(education, original_profession):
    """Adjust profession based on education level to ensure realistic combinations"""
    education_index = education_levels.index(education)
    
    # Define profession categories based on education requirements
    low_education_professions = [
        "Agriculture, Food & Natural Resources", 
        "Transportation, Distribution & Logistics",
        "Architecture & Construction",
        "Hospitality & Tourism",
        "Manufacturing",
        "Human Services"
    ]
    
    medium_education_professions = [
        "Business Management & Administration",
        "Finance",
        "Government & Public Administration",
        "Law, Public Safety, Corrections & Security",
        "Human Services",
        "Marketing"
    ]
    
    high_education_professions = [
        "Information Technology",
        "Health Science",
        "Education & Training",
        "Arts, Audio/Video Technology & Communications"
    ]
    
    # Very high education professions
    very_high_education_professions = [
        "Science, Technology, Engineering & Mathematics"
    ]
    
    if education_index == 0:  # elementary
        # Only low education professions are realistic
        return random.choice(low_education_professions)
    elif education_index == 1:  # high school
        # Most low/medium education professions, with some exceptions
        if original_profession in high_education_professions or original_profession in very_high_education_professions:
            return random.choice(medium_education_professions + low_education_professions)
        return original_profession
    elif education_index == 2:  # bachelor
        # Can do most professions but adjust if needed
        if original_profession in very_high_education_professions:
            # Some PhD-only positions might be too advanced
            return random.choice(high_education_professions + medium_education_professions)
        return original_profession
    else:  # master/phd
        # Can do any profession
        return original_profession


def get_random_profession():
    return random.choices(professions, p_professions)[0]


def get_interested_topics(age, gender, education, income, location, family_status, religion, 
                         big_five_traits, sexual_orientation, gender_identity, ethnicity, 
                         disability_status, country, profession):
    openness, conscientiousness, extraversion, agreeableness, neuroticism = big_five_traits
    prompt = f"""Based on the provided comprehensive demographic and psychological characteristics, please select 2-3 topics of interest from the given list.
    Input:
        Age: {age}
        Gender: {gender}
        Education: {education}
        Income Bracket: {income}
        Location Type: {location}
        Family Status: {family_status}
        Religion: {religion}
        Big Five Personality Traits - Openness: {openness}, Conscientiousness: {conscientiousness}, Extraversion: {extraversion}, Agreeableness: {agreeableness}, Neuroticism: {neuroticism}
        Sexual Orientation: {sexual_orientation}
        Gender Identity: {gender_identity}
        Ethnicity: {ethnicity}
        Disability Status: {disability_status}
        Country: {country}
        Profession: {profession}
    Available Topics:
        1. Economics: The study and management of production, distribution, and consumption of goods and services. Economics focuses on how individuals, businesses, governments, and nations make choices about allocating resources to satisfy their wants and needs, and tries to determine how these groups should organize and coordinate efforts to achieve maximum output.
        2. IT (Information Technology): The use of computers, networking, and other physical devices, infrastructure, and processes to create, process, store, secure, and exchange all forms of electronic data. IT is commonly used within the context of business operations as opposed to personal or entertainment technologies.
        3. Culture & Society: The way of life for an entire society, including codes of manners, dress, language, religion, rituals, norms of behavior, and systems of belief. This topic explores how cultural expressions and societal structures influence human behavior, relationships, and social norms.
        4. General News: A broad category that includes current events, happenings, and trends across a wide range of areas such as politics, business, science, technology, and entertainment. General news provides a comprehensive overview of the latest developments affecting the world at large.
        5. Politics: The activities associated with the governance of a country or other area, especially the debate or conflict among individuals or parties having or hoping to achieve power. Politics is often a battle over control of resources, policy decisions, and the direction of societal norms.
        6. Business: The practice of making one's living through commerce, trade, or services. This topic encompasses the entrepreneurial, managerial, and administrative processes involved in starting, managing, and growing a business entity.
        7. Fun: Activities or ideas that are light-hearted or amusing. This topic covers a wide range of entertainment choices and leisure activities that bring joy, laughter, and enjoyment to individuals and groups.
    Output:
    [list of topic numbers]
    Ensure your output could be parsed to **list**, don't output anything else."""  # noqa: E501

    response = client.chat.completions.create(model=args.model_name,
                                              messages=[{
                                                  "role": "system",
                                                  "content": prompt
                                              }])

    topics = response.choices[0].message.content.strip()
    
    # Try to extract JSON from the response if there's extra text
    try:
        return json.loads(topics)
    except json.JSONDecodeError:
        # If direct JSON parsing fails, try to extract JSON from the response
        import re
        # Look for JSON array pattern
        json_match = re.search(r'\[.*\]', topics, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If still can't parse, raise the original error
        raise json.JSONDecodeError(f"Could not parse topics from AI response: {topics}", topics, 0)


def generate_user_profile(age, gender, education, income, location, family_status, religion, 
                         big_five_traits, sexual_orientation, gender_identity, ethnicity, 
                         disability_status, digital_literacy, consumption_pattern, info_source, 
                         ses_score, country, profession, topics):
    openness, conscientiousness, extraversion, agreeableness, neuroticism = big_five_traits
    prompt = f"""
    Please generate a social media user profile based on the provided comprehensive personal information, including a real name, username, user bio, and a detailed user persona. The focus should be on creating a realistic background story and detailed interests based on their diverse characteristics.
    Generate ONLY a valid JSON object. Do NOT include any markdown, code fences, explanations, or extra text. Output must start with {{ and end with }} and be parseable by json.loads().

        Input:
            Age: {age}
            Gender: {gender}
            Education: {education}
            Income Bracket: {income}
            Location Type: {location}
            Family Status: {family_status}
            Religion: {religion}
            Big Five Personality Traits - Openness: {openness}, Conscientiousness: {conscientiousness}, Extraversion: {extraversion}, Agreeableness: {agreeableness}, Neuroticism: {neuroticism}
            Sexual Orientation: {sexual_orientation}
            Gender Identity: {gender_identity}
            Ethnicity: {ethnicity}
            Disability Status: {disability_status}
            Digital Literacy Level: {digital_literacy}
            Consumption Pattern: {consumption_pattern}
            Information Source Preference: {info_source}
            Socioeconomic Status Score: {ses_score}
            Country: {country}
            Profession: {profession}
            Interested Topics: {topics}

        Output:
        {{
            "realname": "str",
            "username": "str",
            "bio": "str",
            "persona": "str"
        }}"""

    response = client.chat.completions.create(model=args.model_name,
                                              messages=[{
                                                  "role": "system",
                                                  "content": prompt
                                              }])

    profile = response.choices[0].message.content.strip()
    
    # Try to extract JSON from the response if there's extra text
    try:
        return json.loads(profile)
    except json.JSONDecodeError:
        # If direct JSON parsing fails, try to extract JSON from the response
        import re
        # Look for JSON object pattern
        json_match = re.search(r'\{.*\}', profile, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If still can't parse, raise the original error
        raise json.JSONDecodeError(f"Could not parse profile from AI response: {profile}", profile, 0)


def index_to_topics(index_lst):
    topic_dict = {
        '1': 'Economics',
        '2': 'Information Technology',
        '3': 'Culture & Society',
        '4': 'General News',
        '5': 'Politics',
        '6': 'Business',
        '7': 'Fun'
    }
    result = []
    for index in index_lst:
        # Handle both string and integer indexes by converting to string
        index_str = str(index)
        # Remove any extra characters like periods or spaces
        index_str = index_str.strip().rstrip('.').strip()
        if index_str in topic_dict:
            topic = topic_dict[index_str]
            result.append(topic)
        else:
            # If index is not found, try to extract number from the index string
            import re
            number_match = re.search(r'\d+', index_str)
            if number_match:
                number = number_match.group(0)
                if number in topic_dict:
                    topic = topic_dict[number]
                    result.append(topic)
    return result


def calculate_risk_tolerance(big_five_traits, age, income):
    """Calculate risk tolerance based on personality, age, and income"""
    openness, conscientiousness, extraversion, agreeableness, neuroticism = big_five_traits
    
    # Risk tolerance is typically higher with high openness, extraversion, and low neuroticism
    # Younger people might be more risk-tolerant
    # Higher income might allow more risk-taking
    
    risk_base = (openness * 0.3) + (extraversion * 0.2) - (neuroticism * 0.2) + (conscientiousness * -0.1)
    
    # Younger people might be more risk-tolerant (up to a point)
    if age < 30:
        risk_base += 10
    elif age > 60:
        risk_base -= 15
    
    # Higher income allows more risk-taking
    income_index = income_brackets.index(income)
    risk_base += (income_index * 5)  # Higher income = higher risk tolerance
    
    # Normalize to 0-100 scale
    risk_tolerance = max(0, min(100, int(risk_base + 50)))  # Base around 50 plus adjustments
    return risk_tolerance


def calculate_conformity_score(big_five_traits, age, education, religion):
    """Calculate conformity to group pressure based on personality and background"""
    openness, conscientiousness, extraversion, agreeableness, neuroticism = big_five_traits
    
    # Higher agreeableness and conscientiousness = higher conformity
    # Lower openness = higher conformity
    # Higher neuroticism might lead to conformity for security
    conformity_base = (agreeableness * 0.3) + (conscientiousness * 0.2) - (openness * 0.2) + (neuroticism * 0.1)
    
    # Education level: higher education might lead to less conformity
    education_index = education_levels.index(education)
    conformity_base -= (education_index * 3)  # Higher education = less conformity
    
    # Religious people might show higher conformity in religious contexts
    if religion in ['christian', 'muslim', 'other religion']:
        conformity_base += 10
    
    conformity_score = max(0, min(100, int(conformity_base + 50)))  # Base around 50 plus adjustments
    return conformity_score


def calculate_social_trust_score(big_five_traits, age, income, ethnicity):
    """Calculate general trust in others and institutions"""
    openness, conscientiousness, extraversion, agreeableness, neuroticism = big_five_traits
    
    # Higher agreeableness and extraversion = higher trust
    # Higher neuroticism = lower trust
    trust_base = (agreeableness * 0.3) + (extraversion * 0.2) - (neuroticism * 0.3) + (openness * 0.1)
    
    # Higher income often means higher trust in institutions
    income_index = income_brackets.index(income)
    trust_base += (income_index * 4)
    
    trust_score = max(0, min(100, int(trust_base + 50)))  # Base around 50 plus adjustments
    return trust_score


def calculate_political_leaning(big_five_traits, age, education, religion, ethnicity, income):
    """Calculate political leaning based on various factors"""
    openness, conscientiousness, extraversion, agreeableness, neuroticism = big_five_traits
    
    # Higher openness typically correlates with more liberal views
    # Higher conscientiousness and agreeableness might correlate with conservative views
    political_base = (openness * 0.4) - (conscientiousness * 0.2) - (agreeableness * 0.15) + (neuroticism * 0.05)
    
    # Higher education often correlates with more liberal views
    education_index = education_levels.index(education)
    political_base += (education_index * 5)
    
    # Religious people often lean more conservative
    if religion in ['christian', 'muslim']:
        political_base -= 15
    elif religion == 'no religion':
        political_base += 10
        
    # Younger people often lean more liberal
    if age < 35:
        political_base += 10
    elif age > 60:
        political_base -= 10
        
    # Income: complex relationship, but higher income can be either way
    income_index = income_brackets.index(income)
    if income_index <= 1:  # Low income
        political_base += 5  # Might lean more liberal on social issues
    elif income_index >= 3:  # High income
        political_base -= 5  # Might lean more conservative on economic issues
    
    political_leaning = max(0, min(100, int(political_base + 50)))  # Base around 50 plus adjustments
    return political_leaning


profile_generation_failures = 0

def create_user_profile():
    global profile_generation_failures
    while True:
        try:
            # Basic demographics
            gender = get_random_gender()
            age = get_random_age()
            
            # Educational and socioeconomic characteristics
            education = get_random_education()
            # Adjust education based on age for realistic life stage
            education = adjust_education_for_age(age, education)
            
            income = get_random_income()
            # Adjust income based on age and education for realistic expectations
            income = adjust_income_for_age_education(age, education, income)
            
            location = get_random_location()
            family_status = get_random_family_status()
            # Adjust family status based on age for realistic life stage
            family_status = adjust_family_status_for_age(age, family_status)
            religion = get_random_religion()
            
            # Big Five personality traits
            big_five_traits = [
                get_openness_score(),
                get_conscientiousness_score(),
                get_extraversion_score(),
                get_agreeableness_score(),
                get_neuroticism_score()
            ]
            
            # Geographic and professional
            country = get_random_country()
            profession = get_random_profession()
            
            # Adjust profession based on education for realistic combinations
            profession = adjust_profession_for_education(education, profession)
            
            # Identity characteristics
            sexual_orientation = get_random_sexual_orientation()
            gender_identity = get_random_gender_identity()
            ethnicity = get_random_ethnicity()
            # Adjust ethnicity based on country for geographic consistency
            ethnicity = adjust_ethnicity_for_country(country, ethnicity)
            disability_status = get_random_disability_status()
            
            # Behavioral characteristics
            digital_literacy = get_random_digital_literacy()
            # Adjust digital literacy based on age and education for realistic expectations
            digital_literacy = adjust_digital_literacy_for_age_education(age, education, digital_literacy)
            
            consumption_pattern = get_random_consumption_pattern()
            
            info_source = get_random_info_source()
            # Adjust info source based on digital literacy and age for realistic preferences
            info_source = adjust_info_source_for_digital_literacy_age(digital_literacy, age, info_source)
            
            # Geographic and professional
            country = get_random_country()
            profession = get_random_profession()
            
            # Adjust profession based on education for realistic combinations
            profession = adjust_profession_for_education(education, profession)
            
            # Adjust profession based on age for realistic career stage
            profession = adjust_profession_for_age(age, profession)
            
            # Calculate SES score
            ses_score = calculate_ses_score(education, income, profession)
            
            # Get interested topics based on comprehensive profile
            topic_index_lst = get_interested_topics(age, gender, education, income, location, 
                                                  family_status, religion, big_five_traits, 
                                                  sexual_orientation, gender_identity, 
                                                  ethnicity, disability_status, country, 
                                                  profession)
            topics = index_to_topics(topic_index_lst)
            
            # Generate user profile with comprehensive characteristics
            profile = generate_user_profile(age, gender, education, income, location, 
                                          family_status, religion, big_five_traits, 
                                          sexual_orientation, gender_identity, ethnicity, 
                                          disability_status, digital_literacy, 
                                          consumption_pattern, info_source, ses_score, 
                                          country, profession, topics)
            
            # Calculate social context parameters for behavior prediction
            risk_tolerance = calculate_risk_tolerance(big_five_traits, age, income)
            conformity_score = calculate_conformity_score(big_five_traits, age, education, religion)
            social_trust_score = calculate_social_trust_score(big_five_traits, age, income, ethnicity)
            political_leaning = calculate_political_leaning(big_five_traits, age, education, religion, ethnicity, income)
            
            # Add all characteristics to the profile
            profile['age'] = age
            profile['gender'] = gender
            profile['education'] = education
            profile['income'] = income
            profile['location'] = location
            profile['family_status'] = family_status
            profile['religion'] = religion
            profile['big_five_traits'] = {
                'openness': big_five_traits[0],
                'conscientiousness': big_five_traits[1],
                'extraversion': big_five_traits[2],
                'agreeableness': big_five_traits[3],
                'neuroticism': big_five_traits[4]
            }
            profile['sexual_orientation'] = sexual_orientation
            profile['gender_identity'] = gender_identity
            profile['ethnicity'] = ethnicity
            profile['disability_status'] = disability_status
            profile['digital_literacy'] = digital_literacy
            profile['consumption_pattern'] = consumption_pattern
            profile['info_source'] = info_source
            profile['ses_score'] = ses_score
            profile['country'] = country
            profile['profession'] = profession
            profile['interested_topics'] = topics
            profile['risk_tolerance'] = risk_tolerance
            profile['conformity_score'] = conformity_score
            profile['social_trust_score'] = social_trust_score
            profile['political_leaning'] = political_leaning  # 0-100, where lower = more liberal, higher = more conservative
            
            return profile
        except Exception as e:
            profile_generation_failures += 1
            print(f"Profile generation failed: {e}.\n Cumulated failure counts: {profile_generation_failures} times...\nRetrying...")


def generate_user_data(n):
    user_data = []
    start_time = datetime.now()
    max_workers = args.max_workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_user_profile) for _ in range(n)]
        for i, future in enumerate(as_completed(futures)):
            profile = future.result()
            user_data.append(profile)
            elapsed_time = datetime.now() - start_time
            print(f"Generated {i+1}/{n} user profiles. Time elapsed: "
                  f"{elapsed_time}")
    return user_data


import os

def save_user_data(user_data, filename):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    N = args.N_of_agents
    user_data = generate_user_data(N)
    output_path = args.output_path
    save_user_data(user_data, output_path)
    error_rate = f"{profile_generation_failures/N:.2%}" if N > 0 else "0.00%"
    print(f"Generated {N} user profiles and saved to {output_path} Overall failure counts: {profile_generation_failures} times. Error rate: {error_rate}")
