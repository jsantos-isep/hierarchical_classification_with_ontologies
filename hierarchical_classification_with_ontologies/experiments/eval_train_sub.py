import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

load_dotenv()

DATASET = "the_guardian"
MODEL = "microsoft/deberta-base"


# 1. Load the CSV
input_dir = os.getenv(f"DATASETS_FOLDER_{str(DATASET).upper()}_RESULTS_FOLDER")
df = pd.read_csv(f"{input_dir}/{DATASET}_{str(MODEL).split('/')[-1]}_sub_results_val_dataset.csv")
# 2. Extract true and predicted labels
y_true = df['true_subcategory']
y_pred = df['pred_subcategory']

# 3. Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
f1_score_result = f1_score(y_true, y_pred, average='weighted', zero_division=0)

f1_score_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)

recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)


# Your list of categories
#categories = ["agricultural_and_biological_sciences", "arts_and_humanities", "biochemistry_genetics_and_molecular_biology", "business_management_and_accounting", "chemical_engineering", "chemistry", "computer_science", "decision_sciences", "dentistry", "earth_and_planetary_sciences", "economics_econometrics_and_finance", "energy", "engineering", "environmental_science", "health_professions", "immunology_and_microbiology", "materials_science", "mathematics", "medicine", "neuroscience", "nursing", "pharmacology_toxicology_and_pharmaceutics", "physics_and_astronomy", "psychology", "social_sciences", "veterinary"]
categories=["artanddesign", "australia-news", "books", "business", "childrens-books-site", "cities", "crosswords", "culture", "education", "environment", "fashion", "film", "food", "football", "games", "global", "global-development", "law", "lifeandstyle", "media", "media-network", "money", "music", "politics", "science", "small-business-network", "society", "sport", "stage", "sustainable-business", "teacher-network", "technology", "travel", "tv-and-radio", "uk-news", "us-news", "world"]


# Build file names based on a naming pattern
files_to_read = [f"{DATASET}_{str(MODEL).split('/')[-1]}_sub_results_{cat}_train.csv" for cat in categories]

# Total time accumulator
total_time = pd.to_timedelta(0)

# Process each file
for filename in files_to_read:
    filepath = os.path.join(input_dir, filename)

    try:
        df = pd.read_csv(filepath)


        if 'Time' in df.columns:
            df['Time'] = df['Time'].astype(str).str.strip()
            df['Time'] = pd.to_timedelta(df['Time'], errors='coerce')

            total_time += df['Time'].sum()

        else:
            print(f"'Time' column not found in {filename}")
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error reading {filename}: {e}")

total_time_str = str(total_time).split(" ")[-1] # removes microseconds if present

# Output total
print(f"Total time from generated files: {total_time_str}")

df_results = pd.DataFrame({'accuracy': f"{accuracy*100:.2f}%", "f1": f"{f1_score_result*100:.2f}%", "F1 macro": f"{f1_score_macro*100:.2f}%", "precision": f"{precision*100:.2f}%", "recall": f"{recall*100:.2f}%", "Time": f"{total_time_str}"}, index=[0])
df_results.to_csv(
            f"{os.getenv(f"DATASETS_FOLDER_{str(DATASET).upper()}_RESULTS_FOLDER")}/{DATASET}_{str(MODEL).split('/')[-1]}_sub_results_train_eval.csv",
            index=True, header=True)
